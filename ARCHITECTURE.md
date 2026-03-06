# FASEROH POC — Architecture & File Interaction Map

**FASEROH** = *Fast Accurate Symbolic Empirical Representation Of Histograms*

The POC trains a seq2seq neural network that reads a normalised histogram and
outputs a symbolic mathematical expression (in prefix notation) that best
describes the underlying probability density function.

---

## Directory Layout

```
faseroh-poc/
├── faseroh/
│   ├── __init__.py            # Package marker
│   ├── main.py                # Entry point: config, training loop, evaluation
│   ├── model.py               # FASeROH neural network (encoder + decoder)
│   ├── train.py               # Training loop, validation loss, checkpointing
│   ├── dataset.py             # FASeROHDataset, collate_fn, DataLoader builders
│   ├── dataset_generation.py  # Symbolic function generator (Algorithm 1 & 2)
│   ├── tokenizer.py           # Vocabulary, token↔id conversion utilities
│   ├── inference.py           # Top-K autoregressive sampling, expression eval
│   └── metrics.py             # R², sentence accuracy, prefix-validity metrics
├── data/
│   ├── dataset_demo_5k.json   # Pre-built dataset (5 000 samples, used by default)
│   ├── dataset_demo_100.json  # Tiny dataset for quick smoke-tests
│   └── dataset_demo_1M.json   # Large dataset for full training runs
├── checkpoints/
│   └── best_model.pt          # Saved when a new best val-loss is achieved
└── ARCHITECTURE.md            # This file
```

---

## File Responsibilities & Interaction

### `faseroh/main.py` — Entry Point & Configuration

**The single file you need to edit and run.**

- Defines `FASeROHConfig` — a `@dataclass` that is the single source of truth
  for every hyperparameter (model size, training schedule, inference settings,
  dataset split fractions).  There is no separate `config.py`.
- Creates the singleton `CONFIG = FASeROHConfig()`.
- Defines `DATASET_JSON_PATH` pointing to the JSON dataset to load.
- Provides helper functions:
  - `_load_json_records` — parses the JSON and restores `np.ndarray` fields.
  - `_split_records` — slices records into train / val / test by fraction.
  - `_build_dataloaders` — wraps splits in `FASeROHDataset` + `DataLoader`.
  - `_numpy_fn_from_expr_str` — builds a numpy callable from a record's
    `expr_str` (used for R² evaluation without re-generating functions).
  - `_get_test_numpy_fns` — maps `_numpy_fn_from_expr_str` over all test records.
- `main()` orchestrates the full pipeline: load → split → build loaders →
  instantiate model → train → periodic eval callback → final evaluation.

**Imports from:** `dataset.py`, `model.py`, `train.py`, `metrics.py`

**The config object is passed explicitly** to every function that needs it;
no other module imports `FASeROHConfig`.

---

### `faseroh/model.py` — Neural Network

Defines three `nn.Module` classes:

| Class | Role |
|---|---|
| `PositionalEncoding` | Standard sinusoidal PE added to sequence embeddings |
| `HistogramEncoder` | Linear → Conv1d×2 → PosEnc → TransformerEncoder → cross-attention pooling → latent `(B, n_latent, d_model)` |
| `SymbolicDecoder` | Token embedding + mantissa projection → PosEnc → causal TransformerDecoder → symbol logits + constant predictions |
| `FASeROH` | Wrapper: `encoder` + `decoder`; exposes `forward()` (teacher forcing) and `encode()` (encoder-only, for inference) |

**Inputs to `FASeROH.forward`:**

| Tensor | Shape | Description |
|---|---|---|
| `histogram` | `(B, K, 1)` | Normalised bin counts |
| `tgt_ids` | `(B, T)` | Teacher-forced target token ids |
| `mantissas` | `(B, T)` | Teacher-forced mantissa values at constant positions |
| `src_key_padding_mask` | `(B, K)` | `True` = padded histogram bin |

**Outputs:** `(logits (B,T,V), const_preds (B,T,1))`

**Imports from:** `tokenizer.py`
**Receives config from:** `main.py` (passed as argument)

---

### `faseroh/train.py` — Training Loop

Key functions:

| Function | Description |
|---|---|
| `train_one_epoch` | One forward pass over the train loader; computes CE token loss + MSE constant loss with linear lambda warm-up; clips gradients; logs every `config.log_every_n_steps` steps |
| `evaluate` | Validation loss (CE only, no grad) |
| `_print_sample` | Prints one predicted vs ground-truth sequence per epoch |
| `train` | Full epoch loop; checkpoints best val-loss model; calls optional `eval_callback` every `config.evaluate_after` epochs |

**Loss formula:**

```
loss = CrossEntropy(logits[:, :-1], tgt[:, 1:]) + λ(epoch) × MSE(const_preds, mantissas)
```

`λ(epoch)` is 0 for the first `lambda_warmup_epochs` epochs, then ramps
linearly to `lambda_const`.

**Imports from:** `tokenizer.py`
**Receives config from:** `main.py` (passed as argument)

---

### `faseroh/dataset.py` — Dataset & DataLoaders

| Component | Description |
|---|---|
| `FASeROHDataset` | `torch.utils.data.Dataset` wrapping a list of records; normalises histogram bins (`Nk / N`); builds token-id sequences `[<sos>, tok…, <eos>, <pad>…]`; returns dict of tensors |
| `collate_fn` | Pads variable-K histograms to the max K in the batch; stacks token and mantissa tensors; builds `src_key_padding_mask` |
| `build_dataloaders` | Generates fresh data via `dataset_generation.py` (not used in `main.py`; available for standalone generation scripts) |

Each `__getitem__` returns:

```python
{
  "histogram":            Tensor (K, 1),
  "tgt_ids":              Tensor (max_seq_len,),
  "mantissas":            Tensor (max_seq_len,),
  "expr_str":             str,
  "K":                    int,   # actual (un-padded) histogram length
  "T":                    int,   # actual (un-padded) sequence length
}
```

**Imports from:** `dataset_generation.py`, `tokenizer.py`
**Receives config from:** `main.py` (passed as argument)

---

### `faseroh/dataset_generation.py` — Symbolic Function Generator

Implements **Algorithm 1** (histogram sampling) and **Algorithm 2** (dataset
generation) from the FASEROH paper.

| Function | Description |
|---|---|
| `generate_function` | Randomly composes base functions (sine bump, cosine arch, exponential, polynomial) into a normalised PDF over `[0, 1]` |
| `validate_function` | Checks positivity, finiteness, and normalisation |
| `generate_histogram` | Samples `N` data points and bins them into `K` bins |
| `infix_to_prefix` | SymPy expression → pre-order token list |
| `prefix_to_infix` | Pre-order tokens + resolved mantissas → infix string |
| `encode_constants` | Replaces float tokens with `C{ce}` exponent tokens + mantissa |
| `encode_expression` | End-to-end: SymPy expr → `{tokens, mantissas}` |
| `generate_dataset` | Batch generation with caching and timing |

**Used by:** `dataset.py` (for generation mode), `inference.py` (for
`prefix_to_infix` during candidate evaluation)
**Not used in:** `main.py` directly (dataset is always loaded from JSON)

---

### `faseroh/tokenizer.py` — Vocabulary

Defines the fixed token vocabulary used by both encoder and decoder:

| Group | Tokens |
|---|---|
| Special | `<pad>`, `<sos>`, `<eos>`, `<unk>` |
| Operators | `+`, `mul`, `pow`, `sqrt`, `exp`, `sin`, `cos` |
| Variable | `x` |
| Small integers | `-5` … `5` |
| Constant exponents | `C-4` … `C4` |

Key functions: `token_to_id`, `id_to_token`, `tokens_to_ids`, `ids_to_tokens`,
`is_constant_token`, `is_operator_token`, `is_leaf_token`, `build_vocabulary`,
`get_vocab_size`.

**Imported by:** `model.py`, `train.py`, `dataset.py`, `inference.py`, `metrics.py`
**No external project dependencies.**

---

### `faseroh/inference.py` — Autoregressive Sampling

| Function | Description |
|---|---|
| `sample_one` | Generates one candidate expression via top-K sampling from the decoder; returns tokens, mantissas, infix string, and evaluated `y_pred` |
| `run_inference` | Encodes one histogram; samples `config.n_inference_samples` candidates; returns the best by MSE against the true function |
| `_eval_expr_on_grid` | Evaluates a prefix token sequence on a uniform `x` grid; returns `None` if evaluation fails |
| `_reconstruct_value` | Reconstructs a float from a `C{ce}` token and its mantissa: `mantissa × 10^ce` |

**Imports from:** `dataset_generation.py` (for `prefix_to_infix`), `tokenizer.py`
**Receives config from:** `main.py` → `metrics.py` → `run_inference`

---

### `faseroh/metrics.py` — Evaluation Metrics

| Function | Description |
|---|---|
| `r2_score` | Coefficient of determination R² = 1 − SS_res / SS_tot |
| `sentence_accuracy` | Fraction of test samples with an exact token-sequence match |
| `prefix_validity_accuracy` | Fraction of predictions that form a syntactically valid prefix expression tree |
| `evaluate_predictions` | Iterates the test loader; for each sample calls `run_inference`, computes all three metrics, and prints a summary table |

**Imports from:** `inference.py`, `tokenizer.py`
**Receives config from:** `main.py` (passed as argument)

---

## End-to-End Data Flow

```
data/dataset_demo_5k.json
        │
        ▼
main.py: _load_json_records()
        │  restores np.ndarray fields
        ▼
main.py: _split_records()
        │  train 80% / val 10% / test 10%
        ▼
main.py: _build_dataloaders()
        │  FASeROHDataset + collate_fn → DataLoader ×3
        ▼
train.py: train()  ←── model.py: FASeROH
        │  epoch loop
        │  ├─ train_one_epoch()  → CrossEntropy + λ·MSE
        │  ├─ evaluate()         → val CrossEntropy
        │  ├─ checkpoint if best val loss
        │  └─ eval_callback (every evaluate_after epochs)
        │          │
        │          ▼
        │    metrics.py: evaluate_predictions()
        │          │  for each test sample:
        │          │  ├─ inference.py: run_inference()
        │          │  │     └─ sample_one() ×n_inference_samples
        │          │  ├─ r2_score()
        │          │  ├─ sentence_accuracy()
        │          │  └─ prefix_validity_accuracy()
        ▼
main.py: final evaluate_predictions() with best checkpoint
```

---

## How to Run

```bash
# activate environment
source .venv/bin/activate

# edit FASeROHConfig in faseroh/main.py if needed, then:
python -m faseroh.main
```

Checkpoints are saved to `checkpoints/best_model.pt` whenever validation loss
improves.  Evaluation metrics are printed to stdout every `evaluate_after`
epochs and once more at the end of training.
