"""PyTorch Dataset and DataLoader for the FASEROH POC.

Handles histogram normalisation, token sequence construction with
<sos>/<eos>/<pad>, variable-K batching, and dataset caching.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

# Allow importing from parent directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faseroh.dataset_generation import generate_dataset  # noqa: E402
from faseroh.tokenizer import (  # noqa: E402
    token_to_id,
    tokens_to_ids,
    is_constant_token,
)


class FASeROHDataset(Dataset):
    """Wraps records from generate_dataset() as a PyTorch Dataset.

    Parameters
    ----------
    records : list[dict]
        Output of generate_dataset().
    config : FASeROHConfig
    """

    def __init__(self, records: list[dict], config) -> None:
        self.records = records
        self.config = config
        self.pad_id = token_to_id(config.pad_token)
        self.sos_id = token_to_id(config.sos_token)
        self.eos_id = token_to_id(config.eos_token)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        """Return a single sample as a dict of tensors.

        Returns
        -------
        dict with keys: histogram, tgt_ids, mantissas, expr_str, K, T
        """
        rec = self.records[idx]
        hist = rec["histogram"]
        enc = rec["encoding"]

        # Normalise histogram: Nk / N
        bins = hist["bins"].astype(float)
        N = float(hist["N"])
        normed = bins / N if N > 0 else bins
        histogram = torch.tensor(normed, dtype=torch.float32).unsqueeze(-1)  # (K, 1)

        # Build token id sequence: <sos> + tokens + <eos> + <pad>...
        raw_tokens = enc["tokens"]
        raw_mantissas = enc["mantissas"]
        tok_ids = tokens_to_ids(raw_tokens)

        seq = [self.sos_id] + tok_ids + [self.eos_id]
        mant = [0.0] + list(raw_mantissas) + [0.0]

        max_len = self.config.max_seq_len
        T = len(seq)
        if T > max_len:
            seq = seq[:max_len]
            mant = mant[:max_len]
            T = max_len
        else:
            pad_len = max_len - T
            seq = seq + [self.pad_id] * pad_len
            mant = mant + [0.0] * pad_len

        return {
            "histogram": histogram,
            "tgt_ids": torch.tensor(seq, dtype=torch.long),
            "mantissas": torch.tensor(mant, dtype=torch.float32),
            "expr_str": rec["expr_str"],
            "K": int(hist["K"]),
            "T": T,
        }


def collate_fn(batch: list[dict]) -> dict:
    """Collate variable-K histograms into a padded batch.

    Parameters
    ----------
    batch : list[dict]
        Items from FASeROHDataset.__getitem__.

    Returns
    -------
    dict with keys: histogram, tgt_ids, mantissas, src_key_padding_mask,
                    expr_str, K_list, T_list
    """
    max_K = max(item["K"] for item in batch)
    B = len(batch)

    histograms = torch.zeros(B, max_K, 1)
    masks = torch.ones(B, max_K, dtype=torch.bool)  # True = padded

    for i, item in enumerate(batch):
        K = item["K"]
        histograms[i, :K, :] = item["histogram"]
        masks[i, :K] = False

    tgt_ids = torch.stack([item["tgt_ids"] for item in batch])
    mantissas = torch.stack([item["mantissas"] for item in batch])

    return {
        "histogram": histograms,
        "tgt_ids": tgt_ids,
        "mantissas": mantissas,
        "src_key_padding_mask": masks,
        "expr_str": [item["expr_str"] for item in batch],
        "K_list": [item["K"] for item in batch],
        "T_list": [item["T"] for item in batch],
    }


def _cache_key(config, split: str) -> str:
    """Compute a short hash key for dataset caching."""
    sizes = {"train": config.n_train, "val": config.n_val, "test": config.n_test}
    raw = f"{config.seed}_{sizes[split]}_{split}_{config.allowed_base_functions}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _generate_split(
    config, n: int, split: str, seed_offset: int,
) -> list[dict]:
    """Generate or load cached dataset split.

    Parameters
    ----------
    config : FASeROHConfig
    n : int  number of samples
    split : str  "train", "val", or "test"
    seed_offset : int  added to config.seed for split separation

    Returns
    -------
    list[dict]
    """
    cache_dir = Path("data")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{split}_{_cache_key(config, split)}.pt"

    if cache_file.exists():
        print(f"  Loading cached {split} set from {cache_file}")
        return torch.load(cache_file, weights_only=False)

    print(f"  Generating {split} set ({n} samples)...")
    t0 = time.perf_counter()
    records = generate_dataset(
        n=n,
        max_components=config.max_components,
        verbose=False,
        seed=config.seed + seed_offset,
        n_min=config.n_min,
        n_max=config.n_max,
        k_min=config.k_min,
        k_max=config.k_max,
        allowed_factories=config.allowed_base_functions,
    )
    elapsed = time.perf_counter() - t0
    print(f"  {split} set generated in {elapsed:.1f}s")
    torch.save(records, cache_file)
    return records


def build_dataloaders(
    config,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Generate train/val/test datasets and return their DataLoaders.

    Parameters
    ----------
    config : FASeROHConfig

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    print("Building dataloaders...")
    train_recs = _generate_split(config, config.n_train, "train", seed_offset=0)
    val_recs = _generate_split(config, config.n_val, "val", seed_offset=10000)
    test_recs = _generate_split(config, config.n_test, "test", seed_offset=20000)

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

    print(
        f"  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}  "
        f"batch_size={config.batch_size}"
    )
    return train_loader, val_loader, test_loader
