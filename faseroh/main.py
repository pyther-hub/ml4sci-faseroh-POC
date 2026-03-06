"""FASEROH POC — training and validation entry point.

Always loads the dataset from ``DATASET_JSON_PATH`` (no on-the-fly generation).
Modify ``FASeROHConfig`` below to change any hyperparameter, then run:

    python -m faseroh.main

Flow
----
1. Load records from the JSON dataset and split into train / val / test.
2. Wrap each split in a ``FASeROHDataset`` and ``DataLoader``.
3. Instantiate the ``FASeROH`` model and move it to ``config.device``.
4. Build numpy callables from test-record ``expr_str`` fields for R² evaluation.
5. Train with cross-entropy + auxiliary constant-MSE loss; best val-loss is
   checkpointed to ``config.checkpoint_path``.
6. Every ``config.evaluate_after`` epochs, reload the best checkpoint and
   compute R², sentence accuracy, and prefix-validity accuracy on the test set.
7. After training finishes, run a final evaluation with the best checkpoint.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faseroh.dataset import FASeROHDataset, collate_fn  # noqa: E402
from faseroh.model import FASeROH  # noqa: E402
from faseroh.train import train  # noqa: E402
from faseroh.metrics import evaluate_predictions  # noqa: E402


# ── Path to the pre-generated dataset JSON ────────────────────────────────────
DATASET_JSON_PATH: str = "data/dataset_demo_5k.json"


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FASeROHConfig:
    """Single source of truth for all FASEROH hyperparameters.

    Edit the fields directly and re-run ``python -m faseroh.main``.
    No command-line interface is provided by design — all tuning is done here.
    The config instance is passed explicitly to every module; nothing is
    imported globally from this class outside of ``main.py``.

    Dataset splits
    --------------
    train_frac : float
        Fraction of total JSON records used for training (default 0.8).
    val_frac : float
        Fraction used for validation (default 0.1).
    test_frac : float
        Fraction used for testing (default 0.1).  The test slice is computed
        as the remainder so train + val + test always exhausts the dataset.

    Vocabulary
    ----------
    max_seq_len : int
        Maximum token-sequence length, including ``<sos>`` and ``<eos>``.
    pad_token / sos_token / eos_token : str
        Special token strings that must match the tokenizer vocabulary.

    Model architecture
    ------------------
    d_model : int
        Transformer embedding dimension.
    n_heads : int
        Number of attention heads (must evenly divide ``d_model``).
    n_enc_layers : int
        Depth of the Transformer encoder stack in ``HistogramEncoder``.
    n_dec_layers : int
        Depth of the Transformer decoder stack in ``SymbolicDecoder``.
    n_latent : int
        Number of learned latent query vectors for cross-attention pooling.
    conv_kernel : int
        Kernel size for the two Conv1d layers in the histogram encoder.
    dropout : float
        Dropout probability applied throughout the model.

    Training
    --------
    batch_size : int
        Mini-batch size passed to all DataLoaders.
    lr : float
        Adam optimiser learning rate.
    n_epochs : int
        Total number of training epochs.
    evaluate_after : int
        Full test-set evaluation is triggered every this many epochs.
    lambda_const : float
        Weight of the constant-MSE auxiliary loss once fully ramped up.
    lambda_warmup_epochs : int
        Epochs before ``lambda_const`` starts to ramp up (starts at 0).
    grad_clip : float
        Maximum gradient norm for clipping.
    device : str
        Torch device string.  Auto-detected (``"cuda"`` or ``"cpu"``).
    checkpoint_path : str
        File path where the best-val-loss model state dict is saved.
    log_every_n_steps : int
        Frequency (in optimiser steps) for printing per-step loss.

    Inference
    ---------
    top_k : int
        Number of top-logit tokens considered during top-K sampling.
    n_inference_samples : int
        Candidate expressions sampled per histogram at evaluation time.
    """

    # ── Dataset splits ────────────────────────────────────────────────────────
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1

    # ── Vocabulary ────────────────────────────────────────────────────────────
    max_seq_len: int = 30
    pad_token: str = "<pad>"
    sos_token: str = "<sos>"
    eos_token: str = "<eos>"

    # ── Model architecture ────────────────────────────────────────────────────
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
    evaluate_after: int = 5
    lambda_const: float = 0.1
    lambda_warmup_epochs: int = 5
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path: str = "checkpoints/best_model.pt"
    log_every_n_steps: int = 50

    # ── Inference ─────────────────────────────────────────────────────────────
    top_k: int = 10
    n_inference_samples: int = 50


# Singleton config — modify the fields in the class above, not here.
CONFIG = FASeROHConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_config(config: FASeROHConfig) -> None:
    """Print a human-readable summary of all config fields to stdout.

    Parameters
    ----------
    config : FASeROHConfig
        The active configuration instance.
    """
    print("\n" + "=" * 55)
    print("  FASEROH POC — Configuration")
    print("=" * 55)
    for k, v in vars(config).items():
        print(f"  {k:30s} = {v}")
    print("=" * 55 + "\n")


def _count_params(model: torch.nn.Module) -> int:
    """Return the total number of trainable parameters in *model*.

    Parameters
    ----------
    model : torch.nn.Module

    Returns
    -------
    int
        Sum of ``p.numel()`` for all parameters with ``requires_grad=True``.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _load_json_records(path: str) -> list[dict]:
    """Load dataset records from a JSON file and restore numpy arrays.

    Expects the JSON structure produced by ``generate_dataset()`` — a list of
    record dicts where histogram list fields (``bins``, ``edges``, ``means``)
    are stored as plain Python lists and must be converted back to
    ``np.ndarray`` for downstream use.

    Parameters
    ----------
    path : str
        Path to the JSON dataset file (e.g. ``"data/dataset_demo_5k.json"``).

    Returns
    -------
    list[dict]
        Each dict contains at minimum:

        * ``expr_str`` (str) — infix expression string of the true function.
        * ``histogram`` (dict) — with numpy arrays ``bins``, ``edges``,
          ``means`` and integers ``N``, ``K``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist on disk.
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
    records: list[dict],
    config: FASeROHConfig,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split a flat record list into contiguous train / val / test slices.

    Slice boundaries are determined by ``config.train_frac`` and
    ``config.val_frac``.  The test slice takes every remaining record so the
    three parts together always exhaust the full dataset.

    Parameters
    ----------
    records : list[dict]
        Full record list returned by ``_load_json_records``.
    config : FASeROHConfig

    Returns
    -------
    (train_records, val_records, test_records) : tuple[list[dict], ...]
        Three contiguous, non-overlapping slices of *records*.

    Raises
    ------
    ValueError
        If train + val fractions leave no records for the test set.
    """
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
    """Wrap record lists in ``FASeROHDataset`` and return three DataLoaders.

    Parameters
    ----------
    train_recs : list[dict]
        Training split records from ``_split_records``.
    val_recs : list[dict]
        Validation split records.
    test_recs : list[dict]
        Test split records.
    config : FASeROHConfig

    Returns
    -------
    (train_loader, val_loader, test_loader) : tuple[DataLoader, ...]
        The training loader is shuffled; val and test loaders are not.
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


def _numpy_fn_from_expr_str(expr_str: str):
    """Create a numpy callable from an infix expression string.

    Evaluates *expr_str* in a restricted namespace that exposes only numpy
    math functions and ``pi``.  The returned function accepts a numpy array
    ``x`` and returns a numpy array of the same shape.

    Parameters
    ----------
    expr_str : str
        Infix expression string as stored in the dataset JSON, e.g.
        ``"4.0*sin(pi*x)**2"``.  Supported names: ``x``, ``sin``, ``cos``,
        ``exp``, ``sqrt``, ``log``, ``abs``, ``pi``.

    Returns
    -------
    callable
        A function ``fn(x: np.ndarray) -> np.ndarray``.
    """
    _safe_ns: dict = {
        "sin": np.sin,
        "cos": np.cos,
        "exp": np.exp,
        "sqrt": np.sqrt,
        "log": np.log,
        "abs": np.abs,
        "pi": np.pi,
        "__builtins__": {},
    }

    def fn(x: np.ndarray) -> np.ndarray:
        ns = dict(_safe_ns)
        ns["x"] = x
        return eval(expr_str, ns)  # noqa: S307

    return fn


def _get_test_numpy_fns(test_records: list[dict]) -> list:
    """Build a numpy callable for every record in the test split.

    Derives each callable from the record's ``expr_str`` field instead of
    re-generating functions from scratch, so no random seed alignment is
    needed when loading data from a pre-built JSON file.

    Parameters
    ----------
    test_records : list[dict]
        Test split records; each must contain an ``"expr_str"`` key with a
        valid infix expression string.

    Returns
    -------
    list[callable]
        One callable per record, in the same order as *test_records*.
    """
    return [_numpy_fn_from_expr_str(rec["expr_str"]) for rec in test_records]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the full training and validation pipeline.

    Steps
    -----
    1. Print the active ``FASeROHConfig``.
    2. Load all records from ``DATASET_JSON_PATH`` and split into
       train / val / test using ``config.train_frac`` and ``config.val_frac``.
    3. Wrap each split in a ``FASeROHDataset`` and ``DataLoader``.
    4. Instantiate ``FASeROH`` and move it to ``config.device``.
    5. Build numpy callables from test-record ``expr_str`` fields for R²
       evaluation (no dataset re-generation required).
    6. Register an ``eval_callback`` that fires every
       ``config.evaluate_after`` epochs: reloads the best checkpoint and
       calls ``evaluate_predictions`` on the test set.
    7. Run ``train()`` (see ``faseroh/train.py``) which handles the
       epoch loop, loss computation, gradient clipping, and checkpointing.
    8. After training, reload the best checkpoint and run a final evaluation.
    """
    config = CONFIG
    _print_config(config)

    # ── Data ──────────────────────────────────────────────────────────────────
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

    # ── Model ─────────────────────────────────────────────────────────────────
    model = FASeROH(config).to(config.device)
    print(f"Model parameters : {_count_params(model):,}")
    print(f"Device           : {config.device}")

    # ── Numpy callables for R² (derived from expr_str, no re-generation) ──────
    numpy_fns = _get_test_numpy_fns(test_recs)

    # ── Eval callback (fires every config.evaluate_after epochs) ──────────────
    def eval_callback(epoch: int) -> None:
        """Load the best checkpoint and run full test-set evaluation.

        Parameters
        ----------
        epoch : int
            Current 1-indexed epoch number, passed in by the training loop.
        """
        print(
            f"\n[Epoch {epoch}] Running evaluation "
            f"(every {config.evaluate_after} epochs) ..."
        )
        state = torch.load(config.checkpoint_path, weights_only=True)
        model.load_state_dict(state)
        evaluate_predictions(model, test_loader, config, numpy_fns)

    # ── Train ─────────────────────────────────────────────────────────────────
    train(model, train_loader, val_loader, config, eval_callback=eval_callback)

    # ── Final evaluation ───────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  Final evaluation — best checkpoint")
    print("=" * 55)
    state = torch.load(config.checkpoint_path, weights_only=True)
    model.load_state_dict(state)
    final_metrics = evaluate_predictions(model, test_loader, config, numpy_fns)
    print(f"\nFinal metrics: {final_metrics}")


if __name__ == "__main__":
    main()
