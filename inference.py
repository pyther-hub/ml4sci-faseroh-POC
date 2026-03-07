"""Top-K sampling inference for the FASEROH POC.

Generates candidate symbolic expressions autoregressively, then selects
the best candidate by MSE against the true histogram shape.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_generation import prefix_to_infix, encode_constant  # noqa: E402
from tokenizer import (  # noqa: E402
    token_to_id,
    id_to_token,
    is_constant_token,
    is_valid_prefix,
    build_vocabulary,
)


def _reconstruct_value(token: str, mantissa: float) -> float:
    """Reconstruct a float from a C{ce} token and its mantissa.

    Parameters
    ----------
    token : str      e.g. "C2"
    mantissa : float e.g. 0.17815

    Returns
    -------
    float  reconstructed value = mantissa * 10^ce
    """
    ce = int(token[1:])
    return mantissa * (10.0 ** ce)


def _format_prefix_with_constants(tokens: list[str], resolved: list[float]) -> str:
    """Format prefix token list with resolved constant values for display."""
    parts = []
    for tok, val in zip(tokens, resolved):
        if is_constant_token(tok):
            parts.append(f"{val:.4g}")
        else:
            parts.append(tok)
    return "[" + ", ".join(parts) + "]"


def _eval_expr_on_grid(
    tokens: list[str], mantissas: list[float], n_points: int = 100,
) -> tuple[np.ndarray | None, str | None]:
    """Evaluate a predicted prefix expression on a uniform grid.

    Parameters
    ----------
    tokens : list[str]      predicted token sequence (without <sos>/<eos>)
    mantissas : list[float] parallel mantissa values
    n_points : int          number of evaluation points

    Returns
    -------
    (y, error) where y is np.ndarray of shape (n_points,) or None,
    and error is a string description of the failure or None on success.
    """
    resolved_mantissas = []
    for tok, m in zip(tokens, mantissas):
        if is_constant_token(tok):
            resolved_mantissas.append(_reconstruct_value(tok, m))
        else:
            resolved_mantissas.append(m)

    try:
        infix = prefix_to_infix(tokens, resolved_mantissas)
    except Exception as e:
        return None, f"prefix_to_infix failed: {e}"

    if not infix:
        return None, "prefix_to_infix returned an empty string"

    x_grid = np.linspace(1e-6, 1 - 1e-6, n_points)
    try:
        safe_ns = {
            "x": x_grid, "exp": np.e, "sin": np.sin, "cos": np.cos,
            "sqrt": np.sqrt, "log": np.log, "abs": np.abs, "pi": np.pi,
            "__builtins__": {},
        }
        y = eval(infix, safe_ns)  # noqa: S307
        y = np.asarray(y, dtype=float)
        if not np.all(np.isfinite(y)):
            return None, f"expression evaluated to NaN/inf: {infix}"
        return y, None
    except Exception as e:
        return None, f"eval failed on '{infix}': {e}"


@torch.no_grad()
def sample_one(
    model: torch.nn.Module,
    memory: torch.Tensor,
    config,
) -> dict:
    """Generate one candidate via top-K sampling.

    Parameters
    ----------
    model : nn.Module     the FASeROH model
    memory : Tensor       encoder output (1, m, d_model)
    config : FASeROHConfig

    Returns
    -------
    dict with keys: tokens, mantissas, expr_str, y_pred
    """
    model.eval()
    device = config.device
    sos_id = token_to_id(config.sos_token)
    eos_id = token_to_id(config.eos_token)
    pad_id = token_to_id(config.pad_token)

    vocab = build_vocabulary()
    const_ids = {vid for tok, vid in vocab.items() if is_constant_token(tok)}

    seq = [sos_id]
    pred_mantissas = [0.0]
    max_len = config.max_seq_len

    for _ in range(max_len - 1):
        tgt = torch.tensor([seq], dtype=torch.long, device=device)
        mant = torch.tensor([pred_mantissas], dtype=torch.float32, device=device)

        logits, const_preds = model.decoder(tgt, memory, mant)
        next_logits = logits[0, -1]  # (V,)

        # Top-K filtering
        topk_vals, topk_ids = torch.topk(next_logits, config.top_k)
        probs = F.softmax(topk_vals, dim=-1)
        idx = torch.multinomial(probs, 1).item()
        next_id = topk_ids[idx].item()

        if next_id == eos_id:
            break

        seq.append(next_id)
        if next_id in const_ids:
            pred_mantissas.append(const_preds[0, -1, 0].item())
        else:
            pred_mantissas.append(0.0)

    # Strip <sos>
    out_tokens = [id_to_token(i) for i in seq[1:]]
    out_mantissas = pred_mantissas[1:]

    # Build resolved constants and prefix display string
    resolved = []
    for tok, m in zip(out_tokens, out_mantissas):
        if is_constant_token(tok):
            resolved.append(_reconstruct_value(tok, m))
        else:
            resolved.append(m)
    prefix_display = _format_prefix_with_constants(out_tokens, resolved)

    # ── Step 1: Validate prefix expression structure ───────────────────────────
    # Check arity constraints before any infix conversion or constant evaluation.
    prefix_error = None
    if not is_valid_prefix(out_tokens):
        prefix_error = "invalid prefix structure (arity constraint violated)"

    # ── Step 2: Prefix → infix conversion (only if structure is valid) ─────────
    if prefix_error is None:
        try:
            expr_str = prefix_to_infix(out_tokens, resolved)
            if not expr_str:
                prefix_error = "prefix_to_infix returned an empty string"
                expr_str = "(invalid)"
            elif "?" in expr_str:
                prefix_error = "incomplete expression (missing arguments for operator)"
                expr_str = "(invalid)"
        except Exception as e:
            prefix_error = str(e)
            expr_str = "(invalid)"
    else:
        expr_str = "(invalid)"

    # ── Step 3: Evaluate expression on grid (only if prefix is fully valid) ────
    if prefix_error is None:
        y_pred, eval_error = _eval_expr_on_grid(out_tokens, out_mantissas)
    else:
        y_pred, eval_error = None, None

    return {
        "tokens": out_tokens,
        "mantissas": out_mantissas,
        "resolved": resolved,
        "prefix_display": prefix_display,
        "expr_str": expr_str,
        "y_pred": y_pred,
        "prefix_error": prefix_error,
        "eval_error": eval_error,
    }


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    histogram_tensor: torch.Tensor,
    true_fn,
    config,
    src_key_padding_mask: torch.Tensor | None = None,
) -> dict:
    """Run inference: generate candidates and pick the best by MSE.

    Parameters
    ----------
    model : nn.Module
    histogram_tensor : Tensor  shape (1, K, 1)
    true_fn : callable         numpy function f(x) for comparison
    config : FASeROHConfig
    src_key_padding_mask : Tensor | None  shape (1, K)

    Returns
    -------
    dict with keys: best, all_candidates
    """
    model.eval()
    device = config.device
    hist = histogram_tensor.to(device)
    mask = src_key_padding_mask.to(device) if src_key_padding_mask is not None else None

    memory = model.encode(hist, mask)

    x_grid = np.linspace(1e-6, 1 - 1e-6, 100)
    try:
        y_true = true_fn(x_grid)
    except TypeError as e:
        print(f"[run_inference] TypeError evaluating true_fn: {e}")
        print(f"  true_fn = {true_fn}")
        return {
            "best": {"tokens": [], "mantissas": [], "expr_str": "(true_fn error)", "mse": float("inf"), "y_pred": None},
            "all_candidates": [],
        }

    n_invalid = 0
    candidates = []
    for _ in range(config.n_inference_samples):
        cand = sample_one(model, memory, config)
        has_unk = "<unk>" in cand["tokens"]
        if has_unk or cand["prefix_error"] is not None or cand["eval_error"] is not None:
            cand["mse"] = float("inf")
            n_invalid += 1
        elif cand["y_pred"] is not None:
            cand["mse"] = float(np.mean((cand["y_pred"] - y_true) ** 2))
        else:
            cand["mse"] = float("inf")
            n_invalid += 1
        candidates.append(cand)

    valid = [c for c in candidates if c["mse"] < float("inf")]
    if valid:
        best = min(valid, key=lambda c: c["mse"])
    else:
        best = candidates[0] if candidates else {"tokens": [], "mantissas": [], "expr_str": "(no valid)", "mse": float("inf")}

    return {"best": best, "all_candidates": candidates, "n_invalid": n_invalid}
