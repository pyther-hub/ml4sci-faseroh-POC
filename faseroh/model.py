"""Full FASeROH model: histogram encoder + symbolic decoder.

Architecture
------------
- HistogramEncoder: Linear → Conv1d×2 → PosEnc → TransformerEncoder → CrossAttention pooling
- SymbolicDecoder:  Embedding + mantissa proj → PosEnc → TransformerDecoder → symbol & const heads
- FASeROH:          Encoder + Decoder wrapper with forward() and encode()
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from faseroh.config import FASeROHConfig
from faseroh.tokenizer import get_vocab_size, is_constant_token, build_vocabulary


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding.

    Parameters
    ----------
    d_model : int   embedding dimension
    max_len : int   maximum sequence length
    dropout : float dropout rate
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to input.

        Parameters
        ----------
        x : Tensor  shape (B, T, d_model)

        Returns
        -------
        Tensor  shape (B, T, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class HistogramEncoder(nn.Module):
    """Encode variable-length normalised histograms into fixed-size latent.

    Parameters
    ----------
    config : FASeROHConfig
    """

    def __init__(self, config: FASeROHConfig) -> None:
        super().__init__()
        d = config.d_model
        k = config.conv_kernel
        pad = k // 2

        self.proj = nn.Linear(1, d)
        self.conv1 = nn.Conv1d(d, d, kernel_size=k, padding=pad)
        self.conv2 = nn.Conv1d(d, d, kernel_size=k, padding=pad)
        self.act = nn.GELU()
        self.pos_enc = PositionalEncoding(d, dropout=config.dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.n_heads,
            dim_feedforward=4 * d,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=config.n_enc_layers)

        # Cross-attention pooling: n_latent learnable queries
        self.latent_queries = nn.Parameter(torch.randn(1, config.n_latent, d) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d, config.n_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(d)

    def forward(
        self, histogram: Tensor, src_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Encode histogram to latent representation.

        Parameters
        ----------
        histogram : Tensor  shape (B, K, 1)
        src_key_padding_mask : Tensor | None  shape (B, K), True = padded

        Returns
        -------
        Tensor  shape (B, m, d_model)  where m = n_latent
        """
        h = self.proj(histogram)  # (B, K, d)

        # Conv layers expect (B, d, K)
        h = h.transpose(1, 2)
        h = self.act(self.conv1(h))
        h = self.act(self.conv2(h))
        h = h.transpose(1, 2)  # back to (B, K, d)

        h = self.pos_enc(h)
        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)

        # Cross-attention pooling
        B = h.size(0)
        queries = self.latent_queries.expand(B, -1, -1)
        pooled, _ = self.cross_attn(
            queries, h, h, key_padding_mask=src_key_padding_mask,
        )
        return self.cross_norm(pooled + queries)


class SymbolicDecoder(nn.Module):
    """Autoregressive decoder producing symbol logits and constant predictions.

    Parameters
    ----------
    config : FASeROHConfig
    """

    def __init__(self, config: FASeROHConfig) -> None:
        super().__init__()
        d = config.d_model
        V = get_vocab_size()

        self.tok_embed = nn.Embedding(V, d)
        self.mantissa_proj = nn.Linear(1, d)
        self.pos_enc = PositionalEncoding(d, max_len=config.max_seq_len, dropout=config.dropout)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=config.n_heads,
            dim_feedforward=4 * d,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(dec_layer, num_layers=config.n_dec_layers)

        self.symbol_head = nn.Linear(d, V)
        self.const_head = nn.Linear(d, 1)

        # Precompute which token ids are constant tokens
        vocab = build_vocabulary()
        self.register_buffer(
            "const_mask",
            torch.tensor(
                [is_constant_token(tok) for tok in sorted(vocab, key=vocab.get)],
                dtype=torch.bool,
            ),
        )

    def forward(
        self,
        tgt_ids: Tensor,
        memory: Tensor,
        mantissas: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Decode with teacher forcing.

        Parameters
        ----------
        tgt_ids : Tensor    shape (B, T)
        memory : Tensor     shape (B, m, d_model)
        mantissas : Tensor  shape (B, T) teacher-forced mantissa values

        Returns
        -------
        (logits, const_preds)
            logits : Tensor      shape (B, T, V)
            const_preds : Tensor shape (B, T, 1)
        """
        T = tgt_ids.size(1)
        h = self.tok_embed(tgt_ids)  # (B, T, d)

        # Add mantissa information at constant positions
        mant_emb = self.mantissa_proj(mantissas.unsqueeze(-1))  # (B, T, d)
        # Build mask: which positions have a C-token
        const_ids = self.const_mask[tgt_ids]  # (B, T) bool
        h = h + mant_emb * const_ids.unsqueeze(-1).float()

        h = self.pos_enc(h)

        # Causal mask
        causal = nn.Transformer.generate_square_subsequent_mask(T, device=h.device)
        h = self.transformer(h, memory, tgt_mask=causal, tgt_is_causal=True)

        logits = self.symbol_head(h)
        const_preds = self.const_head(h)
        return logits, const_preds


class FASeROH(nn.Module):
    """Full FASEROH model: histogram encoder + symbolic decoder.

    Parameters
    ----------
    config : FASeROHConfig
    """

    def __init__(self, config: FASeROHConfig) -> None:
        super().__init__()
        self.encoder = HistogramEncoder(config)
        self.decoder = SymbolicDecoder(config)

    def forward(
        self,
        histogram: Tensor,
        tgt_ids: Tensor,
        mantissas: Tensor,
        src_key_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Full forward pass with teacher forcing.

        Parameters
        ----------
        histogram : Tensor            (B, K, 1)
        tgt_ids : Tensor              (B, T)
        mantissas : Tensor            (B, T)
        src_key_padding_mask : Tensor | None  (B, K)

        Returns
        -------
        (logits, const_preds)
            logits : Tensor      (B, T, V)
            const_preds : Tensor (B, T, 1)
        """
        memory = self.encoder(histogram, src_key_padding_mask)
        logits, const_preds = self.decoder(tgt_ids, memory, mantissas)
        return logits, const_preds

    def encode(
        self, histogram: Tensor, src_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Encode histogram only (for inference).

        Parameters
        ----------
        histogram : Tensor            (B, K, 1)
        src_key_padding_mask : Tensor | None  (B, K)

        Returns
        -------
        Tensor  (B, m, d_model)
        """
        return self.encoder(histogram, src_key_padding_mask)
