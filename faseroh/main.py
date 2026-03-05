"""FASEROH POC — main entry point.

Run: python -m faseroh.main

Toggle GENERATE_DATASET to switch between generating fresh data and loading
from a pre-existing JSON file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faseroh.dataset import FASeROHDataset, collate_fn
from faseroh.model import FASeROH
from faseroh.train import train
from faseroh.metrics import evaluate_predictions
from torch.utils.data import DataLoader

# ── User-controlled switches ───────────────────────────────────────────────────
GENERATE_DATASET: bool = not True          # True = generate fresh data; False = load from JSON
DATASET_JSON_PATH: str = "data/dataset_demo_5k.json"  # used only when GENERATE_DATASET = False
# ──────────────────────────────────────────────────────────────────────────────
"""Central configuration dataclass for the FASEROH POC.

Every hyperparameter lives here — no magic numbers anywhere else.
"""

from dataclasses import dataclass, field
import torch


@dataclass
class FASeROHConfig:
    """Single source of truth for all FASEROH hyperparameters."""

    # ── Dataset / Function Generation ─────────────────────────────────────────
    allowed_base_functions: list[str] = field(
        default_factory=lambda: ["sine_bump", "cosine_arch", "exponential", "polynomial"]
    )
    max_components: int = 2
    n_min: int = 500
    n_max: int = 5000
    k_min: int = 20
    k_max: int = 60
    n_total: int = 5500        # total samples for GENERATE_DATASET=True
    train_frac: float = 0.8   # fraction of n_total used for training
    val_frac: float = 0.1     # fraction of n_total used for validation
    test_frac: float = 0.1    # fraction of n_total used for testing
    seed: int = 42

    # computed from fractions in __post_init__ — do not set manually
    n_train: int = field(default=0, init=False)
    n_val: int = field(default=0, init=False)
    n_test: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.n_train = int(self.train_frac * self.n_total)
        self.n_val = int(self.val_frac * self.n_total)
        self.n_test = self.n_total - self.n_train - self.n_val

    # ── Vocabulary ────────────────────────────────────────────────────────────
    max_seq_len: int = 30
    pad_token: str = "<pad>"
    sos_token: str = "<sos>"
    eos_token: str = "<eos>"

    # ── Model Architecture ────────────────────────────────────────────────────
    d_model: int = 128
    n_heads: int = 4
    n_enc_layers: int = 3
    n_dec_layers: int = 3
    n_latent: int = 16
    conv_kernel: int = 3
    dropout: float = 0.1

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size: int = 64
    lr: float = 1e-4
    n_epochs: int = 30
    evaluate_after: int = 5   # run full evaluation every this many epochs
    lambda_const: float = 0.1
    lambda_warmup_epochs: int = 5
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path: str = "checkpoints/best_model.pt"
    log_every_n_steps: int = 50

    # ── Inference ─────────────────────────────────────────────────────────────
    top_k: int = 10
    n_inference_samples: int = 50


CONFIG = FASeROHConfig()


def _print_config(config: FASeROHConfig) -> None:
    """Print a formatted config summary."""
    print("\n" + "=" * 50)
    print("FASEROH POC Configuration")
    print("=" * 50)
    for k, v in vars(config).items():
        print(f"  {k:30s} = {v}")
    print("=" * 50 + "\n")


def _count_params(model: torch.nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _load_json_records(path: str) -> list[dict]:
    """Load dataset records from a JSON file and restore numpy arrays.

    The JSON is expected to be the direct serialisation of generate_dataset()
    output (list of record dicts). List fields that were numpy arrays are
    converted back to np.ndarray.

    Parameters
    ----------
    path : str  path to the JSON file

    Returns
    -------
    list[dict]
    """
    with open(path) as f:
        raw = json.load(f)

    records = []
    for rec in raw:
        hist = rec["histogram"]
        hist["bins"] = np.array(hist["bins"], dtype=int)
        hist["edges"] = np.array(hist["edges"], dtype=float)
        hist["means"] = np.array(hist["means"], dtype=float)
        rec["histogram"] = hist
        records.append(rec)

    return records


def _split_records(
    records: list[dict], config: FASeROHConfig,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split a flat list of records into train / val / test.

    Takes the first n_train as train, next n_val as val, next n_test as test.

    Parameters
    ----------
    records : list[dict]
    config : FASeROHConfig

    Returns
    -------
    (train_records, val_records, test_records)
    """
    total = len(records)
    n_train = int(config.train_frac * total)
    n_val = int(config.val_frac * total)
    n_test = int(config.test_frac * total)
    need = n_train + n_val + n_test
    if total < need:
        raise ValueError(
            f"JSON has {total} records but fractions require "
            f"{need} (train={n_train} + val={n_val} + test={n_test})"
        )
    a = n_train
    b = a + n_val
    c = b + n_test
    return records[:a], records[a:b], records[b:c]


def _build_dataloaders_from_records(
    train_recs: list[dict],
    val_recs: list[dict],
    test_recs: list[dict],
    config: FASeROHConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Wrap pre-loaded record lists into DataLoaders.

    Parameters
    ----------
    train_recs, val_recs, test_recs : list[dict]
    config : FASeROHConfig

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
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


def _get_test_numpy_fns(config: FASeROHConfig, n_test: int) -> list:
    """Regenerate test-set numpy callables (needed for R² computation).

    Parameters
    ----------
    config : FASeROHConfig
    n_test : int  number of test samples to regenerate functions for

    Returns
    -------
    list of callable
    """
    import random
    from faseroh.dataset_generation import generate_function, validate_function

    random.seed(config.seed + 20000)
    np.random.seed(config.seed + 20000)

    fns = []
    while len(fns) < n_test:
        result = generate_function(
            max_components=config.max_components,
            allowed_factories=config.allowed_base_functions,
        )
        if result is not None:
            ok, _ = validate_function(result)
            if ok:
                fns.append(result["numpy_fn"])
    return fns


def main() -> None:
    """Load or generate data, train, then evaluate."""
    config = CONFIG
    _print_config(config)

    # ── Data ──────────────────────────────────────────────────────────────────
    if GENERATE_DATASET:
        print("GENERATE_DATASET = True  →  generating fresh datasets")
        from faseroh.dataset import build_dataloaders
        train_loader, val_loader, test_loader = build_dataloaders(config)
        n_test = config.n_test
    else:
        print(f"GENERATE_DATASET = False  →  loading from {DATASET_JSON_PATH}")
        records = _load_json_records(DATASET_JSON_PATH)
        train_recs, val_recs, test_recs = _split_records(records, config)
        n_test = len(test_recs)
        print(
            f"  Loaded {len(records)} records  →  "
            f"train={len(train_recs)}  val={len(val_recs)}  test={n_test}"
        )
        train_loader, val_loader, test_loader = _build_dataloaders_from_records(
            train_recs, val_recs, test_recs, config,
        )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = FASeROH(config).to(config.device)
    print(f"Model parameters: {_count_params(model):,}")

    # ── Eval callback (called every evaluate_after epochs) ────────────────────
    print("Regenerating test numpy functions for R² evaluation...")
    numpy_fns = _get_test_numpy_fns(config, n_test)

    def eval_callback(epoch: int) -> None:
        print(f"\n[Epoch {epoch}] Running full evaluation (evaluate_after={config.evaluate_after})...")
        model.load_state_dict(torch.load(config.checkpoint_path, weights_only=True))
        evaluate_predictions(model, test_loader, config, numpy_fns)

    # ── Train ─────────────────────────────────────────────────────────────────
    train(model, train_loader, val_loader, config, eval_callback=eval_callback)


if __name__ == "__main__":
    main()
