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

from faseroh.dataset_generation import prefix_to_infix, encode_constant  # noqa: E402
from faseroh.tokenizer import (  # noqa: E402
    token_to_id,
    id_to_token,
    is_constant_token,
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


def _eval_expr_on_grid(
    tokens: list[str], mantissas: list[float], n_points: int = 100,
) -> np.ndarray | None:
    """Evaluate a predicted prefix expression on a uniform grid.

    Parameters
    ----------
    tokens : list[str]     predicted token sequence (without <sos>/<eos>)
    mantissas : list[float] parallel mantissa values
    n_points : int          number of evaluation points

    Returns
    -------
    np.ndarray of shape (n_points,) or None if evaluation fails
    """
    # Resolve C-tokens to their float values for prefix_to_infix
    resolved_mantissas = []
    for tok, m in zip(tokens, mantissas):
        if is_constant_token(tok):
            resolved_mantissas.append(_reconstruct_value(tok, m))
        else:
            resolved_mantissas.append(m)

    try:
        infix = prefix_to_infix(tokens, resolved_mantissas)
    except Exception:
        return None

    x_grid = np.linspace(1e-6, 1 - 1e-6, n_points)
    try:
        # Safe eval with numpy
        safe_ns = {
            "x": x_grid, "exp": np.exp, "sin": np.sin, "cos": np.cos,
            "sqrt": np.sqrt, "log": np.log, "abs": np.abs, "pi": np.pi,
            "__builtins__": {},
        }
        y = eval(infix, safe_ns)
        y = np.asarray(y, dtype=float)
        if not np.all(np.isfinite(y)):
            return None
        return y
    except Exception:
        return None


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

    # Evaluate
    y_pred = _eval_expr_on_grid(out_tokens, out_mantissas)

    # Build infix string
    resolved = []
    for tok, m in zip(out_tokens, out_mantissas):
        if is_constant_token(tok):
            resolved.append(_reconstruct_value(tok, m))
        else:
            resolved.append(m)
    try:
        expr_str = prefix_to_infix(out_tokens, resolved)
    except Exception:
        expr_str = "(invalid)"

    return {
        "tokens": out_tokens,
        "mantissas": out_mantissas,
        "expr_str": expr_str,
        "y_pred": y_pred,
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
    y_true = true_fn(x_grid)

    candidates = []
    for _ in range(config.n_inference_samples):
        cand = sample_one(model, memory, config)
        if cand["y_pred"] is not None:
            cand["mse"] = float(np.mean((cand["y_pred"] - y_true) ** 2))
        else:
            cand["mse"] = float("inf")
        candidates.append(cand)

    valid = [c for c in candidates if c["mse"] < float("inf")]
    if valid:
        best = min(valid, key=lambda c: c["mse"])
    else:
        best = candidates[0] if candidates else {"tokens": [], "mantissas": [], "expr_str": "(no valid)", "mse": float("inf")}

    return {"best": best, "all_candidates": candidates}
