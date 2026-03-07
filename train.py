"""Training loop for the FASEROH POC with teacher forcing.

Loss = CE(token logits, shifted targets) + lambda * MSE(const preds at C-positions, true mantissas)
Lambda warms up from 0 over the first lambda_warmup_epochs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tokenizer import token_to_id, id_to_token, is_constant_token, build_vocabulary


def _get_const_ids(device: torch.device) -> set[int]:
    """Return the set of token ids that correspond to C{ce} tokens."""
    vocab = build_vocabulary()
    return {
        vid for tok, vid in vocab.items() if is_constant_token(tok)
    }


def _compute_lambda(epoch: int, config) -> float:
    """Compute lambda_const for the current epoch (linear warmup).

    Parameters
    ----------
    epoch : int      current epoch (0-indexed)
    config : FASeROHConfig

    Returns
    -------
    float
    """
    if epoch < config.lambda_warmup_epochs:
        return 0.0
    ramp = min(
        (epoch - config.lambda_warmup_epochs + 1)
        / max(config.lambda_warmup_epochs, 1),
        1.0,
    )
    return config.lambda_const * ramp


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config,
    epoch: int,
) -> float:
    """Run one training epoch.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    optimizer : Optimizer
    config : FASeROHConfig
    epoch : int  current epoch (0-indexed)

    Returns
    -------
    float  average training loss for the epoch
    """
    model.train()
    pad_id = token_to_id(config.pad_token)
    const_ids = _get_const_ids(config.device)
    lam = _compute_lambda(epoch, config)

    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    mse_loss_fn = nn.MSELoss()
    total_loss = 0.0
    n_batches = 0

    for step, batch in enumerate(loader):
        hist = batch["histogram"].to(config.device)
        tgt = batch["tgt_ids"].to(config.device)
        mant = batch["mantissas"].to(config.device)
        mask = batch["src_key_padding_mask"].to(config.device)

        logits, const_preds = model(hist, tgt, mant, mask)

        # Token CE loss: predict tgt[:, 1:] from logits[:, :-1]
        token_loss = ce_loss_fn(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)),
            tgt[:, 1:].contiguous().view(-1),
        )

        # Constant MSE loss at C-token positions
        const_loss = torch.tensor(0.0, device=config.device)
        if lam > 0:
            const_mask = torch.zeros_like(tgt, dtype=torch.bool)
            for cid in const_ids:
                const_mask |= (tgt == cid)
            if const_mask.any():
                pred_m = const_preds.squeeze(-1)[const_mask]
                true_m = mant[const_mask]
                const_loss = mse_loss_fn(pred_m, true_m)

        loss = token_loss + lam * const_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if (step + 1) % config.log_every_n_steps == 0:
            print(
                f"    step {step + 1:>4}/{len(loader)}  "
                f"loss={loss.item():.4f}  tok={token_loss.item():.4f}  "
                f"const={const_loss.item():.4f}  lam={lam:.4f}"
            )

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, config,
) -> float:
    """Compute average validation loss.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    config : FASeROHConfig

    Returns
    -------
    float  average validation loss
    """
    model.eval()
    pad_id = token_to_id(config.pad_token)
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        hist = batch["histogram"].to(config.device)
        tgt = batch["tgt_ids"].to(config.device)
        mant = batch["mantissas"].to(config.device)
        mask = batch["src_key_padding_mask"].to(config.device)

        logits, _ = model(hist, tgt, mant, mask)
        loss = ce_loss_fn(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)),
            tgt[:, 1:].contiguous().view(-1),
        )
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def _print_sample(
    model: nn.Module, loader: DataLoader, config,
) -> None:
    """Print one random validation prediction vs ground truth."""
    model.eval()
    batch = next(iter(loader))
    hist = batch["histogram"].to(config.device)
    tgt = batch["tgt_ids"].to(config.device)
    mant = batch["mantissas"].to(config.device)
    mask = batch["src_key_padding_mask"].to(config.device)

    logits, _ = model(hist, tgt, mant, mask)
    pred_ids = logits[0].argmax(dim=-1).cpu().tolist()
    true_ids = tgt[0].cpu().tolist()
    pad_id = token_to_id(config.pad_token)

    pred_toks = [id_to_token(i) for i in pred_ids if i != pad_id]
    true_toks = [id_to_token(i) for i in true_ids if i != pad_id]
    print(f"    True : {' '.join(true_toks)}")
    print(f"    Pred : {' '.join(pred_toks)}")
    print(f"    Expr : {batch['expr_str'][0]}")


