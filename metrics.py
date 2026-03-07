"""Evaluation metrics for the FASEROH POC.

Three metrics:
1. R² score — coefficient of determination on a dense x-grid
2. Sentence accuracy — exact match of full token sequences
3. Prefix validity accuracy — syntactic validity of predicted prefix sequences
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference import run_inference, _eval_expr_on_grid  # noqa: E402


# ── Arity table (mirrors dataset_generation._ARITY) ──────────────────────────
_ARITY = {
    "+": 2, "mul": 2, "pow": 2,
    "sqrt": 1, "exp": 1, "log": 1, "sin": 1, "cos": 1, "tan": 1, "abs": 1,
}


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R² = 1 - SS_res / SS_tot.

    Parameters
    ----------
    y_true : np.ndarray  true function values on the grid
    y_pred : np.ndarray  predicted function values

    Returns
    -------
    float  R² value, or -inf if SS_tot == 0
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("-inf")
    return 1.0 - ss_res / ss_tot


def sentence_accuracy(
    pred_tokens: list[list[str]],
    true_tokens: list[list[str]],
) -> float:
    """Fraction of samples with exact full-sequence token match.

    Parameters
    ----------
    pred_tokens : list of predicted token lists (excluding <sos>/<eos>/<pad>)
    true_tokens : list of ground-truth token lists

    Returns
    -------
    float  accuracy in [0, 1]
    """
    if not pred_tokens:
        return 0.0
    skip = {"<sos>", "<eos>", "<pad>"}
    correct = 0
    for pred, true in zip(pred_tokens, true_tokens):
        p = [t for t in pred if t not in skip]
        t = [t for t in true if t not in skip]
        if p == t:
            correct += 1
    return correct / len(pred_tokens)


def _is_valid_prefix(tokens: list[str]) -> bool:
    """Check if a prefix token sequence forms a valid expression tree.

    Parameters
    ----------
    tokens : list[str]  token sequence (without <sos>/<eos>/<pad>)

    Returns
    -------
    bool
    """
    if not tokens:
        return False
    counter = 1  # expecting one expression
    for tok in tokens:
        if counter <= 0:
            return False
        counter -= 1
        arity = _ARITY.get(tok, None)
        if arity is not None:
            counter += arity
        # leaves (x, ints, C-tokens) have arity 0 → no addition
    return counter == 0


def prefix_validity_accuracy(pred_tokens_list: list[list[str]]) -> float:
    """Fraction of predictions that are syntactically valid prefix expressions.

    Parameters
    ----------
    pred_tokens_list : list of token lists (excluding <sos>/<eos>/<pad>)

    Returns
    -------
    float  accuracy in [0, 1]
    """
    if not pred_tokens_list:
        return 0.0
    valid = sum(_is_valid_prefix(toks) for toks in pred_tokens_list)
    return valid / len(pred_tokens_list)


def evaluate_predictions(
    model: torch.nn.Module,
    test_loader: DataLoader,
    config,
    numpy_fns: list,
) -> dict:
    """Run inference on the test set and compute all metrics.

    Parameters
    ----------
    model : nn.Module
    test_loader : DataLoader
    config : FASeROHConfig
    numpy_fns : list of callable  true numpy functions for R² computation

    Returns
    -------
    dict with keys: r2_mean, r2_median, sentence_acc, prefix_validity_acc
    """
    from tokenizer import id_to_token, token_to_id  # noqa

    model.eval()
    all_r2 = []
    all_pred_tokens = []
    all_true_tokens = []
    fn_idx = 0
    pad_id = token_to_id(config.pad_token)
    skip_ids = {token_to_id("<sos>"), token_to_id("<eos>"), pad_id}

    x_grid = np.linspace(1e-6, 1 - 1e-6, 200)

    for batch in test_loader:
        B = batch["tgt_ids"].size(0)
        for i in range(B):
            if fn_idx >= len(numpy_fns):
                break

            true_fn = numpy_fns[fn_idx]
            fn_idx += 1

            # Get true tokens
            true_ids = batch["tgt_ids"][i].tolist()
            true_toks = [
                id_to_token(tid) for tid in true_ids
                if tid not in skip_ids
            ]
            all_true_tokens.append(true_toks)

            # Run inference for this sample
            hist = batch["histogram"][i:i+1].to(config.device)
            mask = batch["src_key_padding_mask"][i:i+1].to(config.device)

            result = run_inference(model, hist, true_fn, config, mask)
            best = result["best"]
            all_pred_tokens.append(best["tokens"])

            # R² score
            y_pred = _eval_expr_on_grid(best["tokens"], best["mantissas"], n_points=200)
            if y_pred is not None:
                y_true = true_fn(x_grid)
                all_r2.append(r2_score(y_true, y_pred))
            else:
                all_r2.append(float("-inf"))

    r2_finite = [r for r in all_r2 if r > float("-inf")]
    r2_arr = np.array(r2_finite) if r2_finite else np.array([0.0])

    metrics = {
        "r2_mean": float(np.mean(r2_arr)),
        "r2_median": float(np.median(r2_arr)),
        "sentence_acc": sentence_accuracy(all_pred_tokens, all_true_tokens),
        "prefix_validity_acc": prefix_validity_accuracy(all_pred_tokens),
    }

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  R² mean   : {metrics['r2_mean']:.4f}")
    print(f"  R² median : {metrics['r2_median']:.4f}")
    print(f"  Sentence accuracy     : {metrics['sentence_acc']:.4f}")
    print(f"  Prefix validity acc   : {metrics['prefix_validity_acc']:.4f}")
    print(f"  Samples evaluated     : {len(all_r2)}")
    print(f"  Valid R² scores       : {len(r2_finite)}/{len(all_r2)}")
    print("=" * 50)

    return metrics
