"""FASEROH POC — Jupyter-friendly training and inference script.

Sections
--------
1. Imports
2. Configuration
3. Helper functions
4. Data loading
5. Model setup
6. Training loop
7. Final evaluation
8. Custom function inference
"""

from __future__ import annotations

# ── 1. Imports ────────────────────────────────────────────────────────────────

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import integrate as sci_integrate
from torch.utils.data import DataLoader
from dataset import FASeROHDataset, collate_fn          # noqa: E402
from dataset_generation import generate_histogram, goodness_of_fit  # noqa: E402
from inference import run_inference                                  # noqa: E402
from metrics import evaluate_predictions, r2_score, _pred_fn_from_tokens  # noqa: E402
from model import FASeROH                               # noqa: E402
from  train import train_one_epoch, evaluate, _print_sample  # noqa: E402


# ── 2. Configuration ──────────────────────────────────────────────────────────

DATASET_JSON_PATH: str = "data/dataset_demo_1k.json"


@dataclass
class FASeROHConfig:
    # Dataset splits
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1

    # Vocabulary
    max_seq_len: int = 30
    pad_token: str = "<pad>"
    sos_token: str = "<sos>"
    eos_token: str = "<eos>"

    # Model architecture
    d_model: int = 128
    n_heads: int = 4
    n_enc_layers: int = 3
    n_dec_layers: int = 3
    n_latent: int = 16
    conv_kernel: int = 3
    dropout: float = 0.1

    # Training
    batch_size: int = 64
    lr: float = 1e-4
    n_epochs: int = 30
    evaluate_after: int = 1
    lambda_const: float = 0.1
    lambda_warmup_epochs: int = 5
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path: str = "checkpoints/best_model.pt"
    log_every_n_steps: int = 50

    # Inference
    top_k: int = 10
    n_inference_samples: int = 50

    # Evaluation — set which metrics to compute.
    # Available: "r2", "sentence_acc", "prefix_validity", "fn_validity", "gof"
    # Remove "gof" if goodness-of-fit is causing crashes / wrong outputs.
    eval_metrics: tuple = ("r2", "sentence_acc", "prefix_validity", "fn_validity", "gof")


config = FASeROHConfig()

print("\n" + "=" * 55)
print("  FASEROH POC — Configuration")
print("=" * 55)
for k, v in vars(config).items():
    print(f"  {k:30s} = {v}")
print("=" * 55 + "\n")


# ── 3. Helper functions ───────────────────────────────────────────────────────

def _load_json_records(path: str) -> list[dict]:
    with open(path) as f:
        raw = json.load(f)
    records = []
    for rec in raw:
        hist = rec["histogram"]
        hist["bins"] = np.array(hist["bins"], dtype=int)
        rec["histogram"] = hist
        records.append(rec)
    return records


def _split_records(
    records: list[dict], config: FASeROHConfig,
) -> tuple[list[dict], list[dict], list[dict]]:
    total = len(records)
    n_train = int(config.train_frac * total)
    n_val = int(config.val_frac * total)
    n_test = total - n_train - n_val
    if n_test <= 0:
        raise ValueError(
            f"Dataset has {total} records but train_frac={config.train_frac} "
            f"and val_frac={config.val_frac} leave no records for the test set."
        )
    return (
        records[:n_train],
        records[n_train : n_train + n_val],
        records[n_train + n_val :],
    )


def _build_dataloaders(
    train_recs: list[dict],
    val_recs: list[dict],
    test_recs: list[dict],
    config: FASeROHConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = FASeROHDataset(train_recs, config)
    val_ds = FASeROHDataset(val_recs, config)
    test_ds = FASeROHDataset(test_recs, config)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn,
    )
    return train_loader, val_loader, test_loader


def _numpy_fn_from_expr_str(expr_str: str):
    _safe_ns: dict = {
        "sin": np.sin, "cos": np.cos, "exp": np.e,
        "sqrt": np.sqrt, "log": np.log, "abs": np.abs,
        "pi": np.pi, "__builtins__": {},
    }
    def fn(x: np.ndarray) -> np.ndarray:
        ns = dict(_safe_ns)
        ns["x"] = x
        return eval(expr_str, ns)  # noqa: S307
    return fn


# ── 4. Data loading ───────────────────────────────────────────────────────────

print(f"Loading dataset from {DATASET_JSON_PATH!r} ...")
records = _load_json_records(DATASET_JSON_PATH)
train_recs, val_recs, test_recs = _split_records(records, config)
print(
    f"  Total={len(records)}  "
    f"train={len(train_recs)}  val={len(val_recs)}  test={len(test_recs)}"
)

train_loader, val_loader, test_loader = _build_dataloaders(
    train_recs, val_recs, test_recs, config,
)

numpy_fns = [_numpy_fn_from_expr_str(rec["expr_str"]) for rec in test_recs]


# ── 5. Model setup ────────────────────────────────────────────────────────────

model = FASeROH(config).to(config.device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters : {n_params:,}")
print(f"Device           : {config.device}")


# ── 6. Training loop ──────────────────────────────────────────────────────────

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
best_val_loss = float("inf")

ckpt_dir = Path(config.checkpoint_path).parent
ckpt_dir.mkdir(parents=True, exist_ok=True)

for epoch in range(config.n_epochs):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch + 1}/{config.n_epochs}")
    print(f"{'='*60}")

    train_loss = train_one_epoch(model, train_loader, optimizer, config, epoch)
    val_loss, val_sent_acc = evaluate(model, val_loader, config)

    saved = ""
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), config.checkpoint_path)
        saved = " [saved]"

    print(
        f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
        f"val_sent_acc={val_sent_acc:.4f}  best_val={best_val_loss:.4f}{saved}"
    )
    # _print_sample(model, val_loader, config)

    if (epoch + 1) % config.evaluate_after == 0:
        print(f"\n[Epoch {epoch + 1}] Running evaluation ...")
        state = torch.load(config.checkpoint_path, weights_only=True)
        model.load_state_dict(state)
        evaluate_predictions(model, test_loader, config, numpy_fns, test_recs)

print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


# ── 7. Final evaluation ───────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("  Final evaluation — best checkpoint")
print("=" * 55)
state = torch.load(config.checkpoint_path, weights_only=True)
model.load_state_dict(state)
final_metrics = evaluate_predictions(model, test_loader, config, numpy_fns, test_recs)
print(f"\nFinal metrics: {final_metrics}")


# ── 8. Custom function inference ──────────────────────────────────────────────
# For each function: normalize to a valid PDF over [0,1] following the same
# pipeline as dataset_generation (positive, integrates to 1), then generate
# a histogram and run inference.

functions = [
    lambda x: x**3 + 2*x*np.sin(np.pi*x),
    lambda x: np.exp(-x) * np.cos(2*np.pi*x),
    lambda x: np.log(x + 1) + x**2*np.sin(x),
    lambda x: np.sqrt(x + 0.1) * np.cos(np.pi*x),
    lambda x: (x**2 + 1) * np.sin(3*x),
]

print("\n" + "=" * 55)
print("  Custom function inference")
print("=" * 55)

_x_grid_r2 = np.linspace(1e-6, 1 - 1e-6, 100)
n_invalid_custom = 0

for i, fn in enumerate(functions):
    x_grid = np.linspace(1e-6, 1 - 1e-6, 500)
    ys = fn(x_grid)

    # Shift to non-negative (pipeline requires f(x) >= 0 over [0,1])
    min_y = float(ys.min())
    shift = -min_y if min_y < 0 else 0.0
    fn_pos = (lambda f, s: lambda x: f(x) + s)(fn, shift)

    # Normalize to integrate to 1 over [0,1]
    I, _ = sci_integrate.quad(fn_pos, 0.0, 1.0, limit=200, epsabs=1e-7, epsrel=1e-7)
    fn_normed = (lambda f, norm: lambda x: f(x) / norm)(fn_pos, I)

    # Generate histogram using the same pipeline (Algorithm 1)
    histogram = generate_histogram(fn_normed)

    # Build histogram tensor: shape (1, K, 1), normalized as bins / N
    bins = histogram["bins"].astype(float)
    N = float(histogram["N"])
    normed_bins = bins / N if N > 0 else bins
    hist_tensor = torch.tensor(normed_bins, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    # Run inference
    result = run_inference(model, hist_tensor, fn_normed, config)
    best = result["best"]

    is_invalid = (
        "<unk>" in best.get("tokens", [])
        or best.get("prefix_error") is not None
        or best.get("eval_error") is not None
        or best["expr_str"] in ("(invalid)", "(no valid)")
        or best.get("y_pred") is None
    )

    print(f"\nFunction {i + 1}:")
    if is_invalid:
        n_invalid_custom += 1
    else:
        # R² score
        y_true_r2 = fn_normed(_x_grid_r2)
        r2 = r2_score(y_true_r2, best["y_pred"])

        # Goodness-of-fit
        gof_val = None
        pred_fn = _pred_fn_from_tokens(best["tokens"], best["mantissas"])
        if pred_fn is not None:
            try:
                gof_result = goodness_of_fit(pred_fn, histogram)
                gof_val = gof_result["X_per_ndf"]
            except Exception:
                pass

        print(f"  predicted  : {best['expr_str']}")
        print(f"  MSE        : {best['mse']:.6f}")
        print(f"  R²         : {r2:.4f}")
        if gof_val is not None:
            print(f"  GoF χ²/ndf : {gof_val:.4f}  (≈1 is good fit)")
        else:
            print(f"  GoF χ²/ndf : N/A")

print(f"\n  Invalid: {n_invalid_custom}/{len(functions)} functions had no valid prediction")
