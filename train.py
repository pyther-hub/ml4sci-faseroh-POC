"""Training loop for the FASEROH POC with teacher forcing.

Loss = CE(token logits, shifted targets)
Constants (C-tokens) are treated as classification targets; mantissa regression is not used.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tokenizer import token_to_id, id_to_token


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

    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    total_loss = 0.0
    n_batches = 0

    for step, batch in enumerate(loader):
        hist = batch["histogram"].to(config.device)
        tgt = batch["tgt_ids"].to(config.device)
        mant = batch["mantissas"].to(config.device)
        mask = batch["src_key_padding_mask"].to(config.device)

        logits, _ = model(hist, tgt, mant, mask)

        # CE loss only: predict tgt[:, 1:] from logits[:, :-1]
        loss = ce_loss_fn(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)),
            tgt[:, 1:].contiguous().view(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if (step + 1) % config.log_every_n_steps == 0:
            print(
                f"    step {step + 1:>4}/{len(loader)}  "
                f"loss={loss.item():.4f}"
            )

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, config,
) -> tuple[float, float]:
    """Compute average validation loss, sentence-level accuracy, and token accuracy.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    config : FASeROHConfig

    Returns
    -------
    tuple[float, float, float]  (average validation loss, sentence accuracy, token accuracy)
    """
    model.eval()
    pad_id = token_to_id(config.pad_token)
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    total_loss = 0.0
    n_batches = 0
    n_sent_correct = 0
    n_tok_correct = 0
    n_tok_total = 0
    n_total = 0

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

        pred_ids = logits[:, :-1].argmax(dim=-1)  # (B, T-1)
        true_ids = tgt[:, 1:]                       # (B, T-1)
        non_pad = true_ids != pad_id

        # Sentence accuracy: all non-pad tokens must match
        correct = ((pred_ids == true_ids) | ~non_pad).all(dim=-1)
        n_sent_correct += correct.sum().item()
        n_total += tgt.size(0)

        # Token accuracy: fraction of non-pad tokens predicted correctly
        n_tok_correct += ((pred_ids == true_ids) & non_pad).sum().item()
        n_tok_total += non_pad.sum().item()

    sentence_acc = n_sent_correct / n_total if n_total > 0 else 0.0
    token_acc = n_tok_correct / n_tok_total if n_tok_total > 0 else 0.0
    return total_loss / max(n_batches, 1), sentence_acc, token_acc


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


