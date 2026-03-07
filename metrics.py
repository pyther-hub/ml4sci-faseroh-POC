"""Evaluation metrics for the FASEROH POC.

Four metrics:
1. R² score — coefficient of determination on a dense x-grid
2. Sentence accuracy — exact match of full token sequences
3. Prefix validity accuracy — syntactic validity of predicted prefix sequences
4. Function validity accuracy — fraction of predictions that evaluate numerically
5. Goodness-of-fit (χ²/ndf) — for valid predictions against the true histogram
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_generation import goodness_of_fit, prefix_to_infix  # noqa: E402
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


def _pred_fn_from_tokens(tokens: list[str], mantissas: list[float]):
    """Build a numpy callable from predicted tokens+mantissas, or None on failure."""
    try:
        infix = prefix_to_infix(tokens, mantissas)
    except Exception:
        return None
    if not infix:
        return None
    _ns = {
        "exp": np.e, "sin": np.sin, "cos": np.cos, "sqrt": np.sqrt,
        "log": np.log, "abs": np.abs, "pi": np.pi, "__builtins__": {},
    }
    def fn(x):
        ns = dict(_ns)
        ns["x"] = np.asarray(x, dtype=float)
        result = eval(infix, ns)  # noqa: S307
        result = np.asarray(result, dtype=float)
        if result.ndim == 0:
            result = np.full_like(ns["x"], float(result))
        return result
    return fn


def evaluate_predictions(
    model: torch.nn.Module,
    test_loader: DataLoader,
    config,
    numpy_fns: list,
    raw_records: list[dict] | None = None,
) -> dict:
    """Run inference on the test set and compute all metrics.

    Parameters
    ----------
    model : nn.Module
    test_loader : DataLoader
    config : FASeROHConfig
        Must have an ``eval_metrics`` attribute — a tuple/list of metric names
        to compute.  Supported values:
          "r2"              – R² score against the true function
          "sentence_acc"    – exact token-sequence match
          "prefix_validity" – syntactic validity of prefix sequences
          "fn_validity"     – fraction of predictions that evaluate numerically
          "gof"             – goodness-of-fit χ²/ndf against the true histogram
        Example (disable GoF because it is crashing):
          config.eval_metrics = ("r2", "sentence_acc", "prefix_validity", "fn_validity")
    numpy_fns : list of callable  true numpy functions for R² computation
    raw_records : list[dict] | None  raw dataset records for GoF computation
                  (each record must have a 'histogram' key with {bins, N, K})

    Returns
    -------
    dict  — keys depend on ``config.eval_metrics``
    """
    from tokenizer import id_to_token, token_to_id  # noqa

    # Which metrics to compute (default to all if attribute missing)
    _active = set(getattr(config, "eval_metrics", (
        "r2", "sentence_acc", "prefix_validity", "fn_validity", "gof"
    )))

    run_r2 = "r2" in _active
    run_sent = "sentence_acc" in _active
    run_pfx = "prefix_validity" in _active
    run_fnv = "fn_validity" in _active
    run_gof = "gof" in _active

    model.eval()
    all_r2 = []
    all_pred_tokens = []
    all_true_tokens = []
    all_gof = []
    fn_valid_count = 0
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
            rec_idx = fn_idx
            fn_idx += 1

            # True tokens (needed for sentence_acc / prefix_validity)
            if run_sent or run_pfx:
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

            if run_sent or run_pfx:
                all_pred_tokens.append(best["tokens"])

            # Function validity + R²
            if run_fnv or run_r2 or run_gof:
                has_unk = "<unk>" in best["tokens"]
                if has_unk:
                    y_pred, fn_valid = None, False
                else:
                    y_pred, _ = _eval_expr_on_grid(best["tokens"], best["mantissas"], n_points=200)
                    fn_valid = y_pred is not None
                fn_valid_count += int(fn_valid)

                if run_r2:
                    if fn_valid:
                        y_true = true_fn(x_grid)
                        all_r2.append(r2_score(y_true, y_pred))
                    else:
                        all_r2.append(float("-inf"))

                # Goodness-of-fit against the true histogram
                if run_gof and fn_valid and raw_records is not None and rec_idx < len(raw_records):
                    raw_hist = raw_records[rec_idx]["histogram"]
                    pred_fn = _pred_fn_from_tokens(best["tokens"], best["mantissas"])
                    if pred_fn is not None:
                        try:
                            gof = goodness_of_fit(pred_fn, raw_hist)
                            all_gof.append(gof["X_per_ndf"])
                        except Exception:
                            pass

    n_total = fn_idx  # number of samples processed

    metrics: dict = {}

    if run_r2:
        r2_finite = [r for r in all_r2 if r > float("-inf")]
        r2_arr = np.array(r2_finite) if r2_finite else np.array([0.0])
        metrics["r2_mean"] = float(np.mean(r2_arr))
        metrics["r2_median"] = float(np.median(r2_arr))

    if run_sent:
        metrics["sentence_acc"] = sentence_accuracy(all_pred_tokens, all_true_tokens)

    if run_pfx:
        metrics["prefix_validity_acc"] = prefix_validity_accuracy(all_pred_tokens)

    if run_fnv or run_r2 or run_gof:
        metrics["fn_validity_acc"] = fn_valid_count / n_total if n_total > 0 else 0.0

    if run_gof:
        gof_arr = np.array(all_gof) if all_gof else np.array([float("nan")])
        metrics["gof_mean"] = float(np.nanmean(gof_arr))
        metrics["gof_median"] = float(np.nanmedian(gof_arr))

    print("\n" + "=" * 55)
    print("EVALUATION RESULTS")
    print(f"  Active metrics: {sorted(_active)}")
    print("=" * 55)
    print(f"  Samples evaluated     : {n_total}")
    if run_r2:
        r2_finite = [r for r in all_r2 if r > float("-inf")]
        print(f"  R² mean               : {metrics['r2_mean']:.4f}")
        print(f"  R² median             : {metrics['r2_median']:.4f}  (valid: {len(r2_finite)}/{n_total})")
    if run_sent:
        print(f"  Sentence accuracy     : {metrics['sentence_acc']:.4f}")
    if run_pfx:
        print(f"  Prefix validity acc   : {metrics['prefix_validity_acc']:.4f}")
    if run_fnv or run_r2 or run_gof:
        n_invalid = n_total - fn_valid_count
        print(f"  Function validity acc : {metrics['fn_validity_acc']:.4f}  ({fn_valid_count} valid, {n_invalid} invalid out of {n_total})")
    if run_gof:
        if all_gof:
            print(f"  GoF chi2/ndf mean     : {metrics['gof_mean']:.4f}  (≈1 is good fit)")
            print(f"  GoF chi2/ndf median   : {metrics['gof_median']:.4f}  ({len(all_gof)}/{fn_valid_count} valid fns)")
        else:
            print("  GoF chi2/ndf          : N/A (no raw_records or no valid predictions)")
    print("=" * 55)

    return metrics
