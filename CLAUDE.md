# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**FASEROH** = *Fast Accurate Symbolic Empirical Representation Of Histograms*

A seq2seq neural network POC that reads a normalised histogram and outputs a symbolic mathematical expression (in prefix notation) representing the underlying probability density function.

## Dependencies

Requires Python 3.13+ with: `torch`, `numpy`, `scipy`, `sympy`

## Running

```bash
# Activate the virtual environment first
source .venv/bin/activate

# Run training + evaluation (edit FASeROHConfig in main.py first if needed)
python main.py
```

All configuration lives in the `FASeROHConfig` dataclass at the top of `main.py`. Change `DATASET_JSON_PATH` to switch datasets (`data/dataset_demo_1k.json`, `data/dataset_demo_10k.json`, or `data/dataset_demo_100k.json`).

The `OPTIMISE_FOR_FLOAT` flag (top of `main.py`) controls whether mantissa MSE loss is added alongside CE loss.

Checkpoints are saved to `checkpoints/best_model.pt` whenever validation loss improves.

There is no test suite — validation is done via the training loop's evaluation and the notebooks (`model-test.ipynb`, `dataset-gen.ipynb`, `data/eda.ipynb`).

## Architecture

All modules are flat in the project root (not in a package subdirectory despite ARCHITECTURE.md describing a `faseroh/` layout — the actual files are at root level).

**Data flow:**
```
data/*.json -> main.py (load/split/DataLoaders) -> train.py (epoch loop) -> metrics.py (evaluate_predictions) -> inference.py (run_inference -> sample_one)
```

### Key files

| File | Role |
|---|---|
| `main.py` | Entry point. Defines `FASeROHConfig` (single source of truth for all hyperparameters). Runs training loop inline (not via a `main()` function — it's script-style). |
| `model.py` | `FASeROH` = `HistogramEncoder` + `SymbolicDecoder`. Encoder: Linear→Conv1d×2→TransformerEncoder→cross-attention pooling→latent `(B, n_latent, d_model)`. Decoder: causal TransformerDecoder with dual heads (symbol logits + mantissa regression). |
| `train.py` | `train_one_epoch` and `evaluate`. Loss = CE(tokens) + λ·MSE(mantissas at C-token positions). λ warms up linearly over `lambda_warmup_epochs`. |
| `dataset.py` | `FASeROHDataset` wraps JSON records; `collate_fn` pads variable-K histograms. `build_dataloaders` generates data on-the-fly (not used by main.py which always loads from JSON). |
| `dataset_generation.py` | Symbolic function generator. Key functions: `generate_function`, `generate_histogram`, `infix_to_prefix`, `prefix_to_infix`, `encode_constants`, `goodness_of_fit`. |
| `tokenizer.py` | Fixed vocabulary: special tokens, operators (`+`, `mul`, `pow`, `sqrt`, `log`, `sin`, `cos`), `x`, math constants (`pi`, `exp`), integers -5..5, and `C-4`..`C4` (scientific-notation exponent tokens). |
| `inference.py` | `run_inference`: encodes histogram, samples `n_inference_samples` candidates via top-K, picks best by MSE. `sample_one`: autoregressive top-K decoding. |
| `metrics.py` | `evaluate_predictions` computes R², sentence accuracy, prefix validity, function validity, and goodness-of-fit (χ²/ndf). Configurable via `config.eval_metrics` tuple. |

### Constant encoding scheme

Float constants are encoded as `C{ce}` token (exponent, range -4..4) + a mantissa float. At inference, value = `mantissa × 10^ce`. This lets the model handle arbitrary floats using a fixed vocabulary.

### Import note

`main.py` imports modules directly by name (`from dataset import ...`) rather than as a package, so run it from the project root directory.

---

## Dataset Generation Approach

### Overview

The FASEROH dataset generation pipeline creates synthetic training pairs of (histogram, symbolic expression) by synthesizing random probability density functions (PDFs) and sampling them according to a histogram-based empirical distribution. This approach enables controlled generation of diverse, ground-truth labeled data without manual annotation.

### Generation Pipeline

#### 1. **Symbolic Function Synthesis**

The first step generates a random symbolic mathematical expression using a compositional approach:

- **Base Functions**: The generator randomly selects from a set of elementary base functions:
  - `Uniform`: constant function f(x) = 1
  - `Linear`: f(x) = a + bx with parameters sampled from [0.1, 3.0]
  - `Exponential`: f(x) = exp(λx) with λ ∈ {-3, -2, -1, 1, 2, 3} (80% chance) or random float (20% chance)
  - `Power`: f(x) = x^p with p ∈ {0.5, 1.5, 2.0, 3.0}
  - `Polynomial`: f(x) = Σ a_i·x^i with random coefficients
  - `Trigonometric`: sin(x), cos(x), or combinations thereof
  - `Square root`: f(x) = √x

- **Composition Strategy**: Base functions are recursively combined using binary operations (addition, multiplication) with a depth limit to control expression complexity. This creates diverse expressions of varying sizes and mathematical structure.

- **Normalization**: Generated expressions are normalized so that ∫₀¹ f(x)dx = 1, ensuring valid probability density functions. Normalization is done numerically using scipy's quadrature integration.

- **Validation**: Each generated function undergoes validation checks:
  - Positivity: f(x) > 0 for all x ∈ [0, 1]
  - Finiteness: No infinite or NaN values
  - Proper normalization: Integral ≈ 1
  - Numerical stability: Function values within reasonable bounds

#### 2. **Histogram Generation (Algorithm 1)**

Once a valid symbolic function f(x) is obtained, it is empirically sampled into a histogram:

- **Binning**: The domain [0, 1] is divided into K bins (default: K=32). Bin width = 1/K.

- **Integration**: For each bin i, the algorithm numerically integrates:
  ```
  λᵢ = ∫[bin_i] f(x)dx
  ```
  This gives the expected probability mass in each bin.

- **Poisson Sampling**: The number of samples n_i in bin i is drawn from a Poisson distribution with rate parameter λᵢ:
  ```
  n_i ~ Poisson(λᵢ)
  ```
  This creates realistic histogram noise typical of empirical sampling.

- **Histogram Representation**: The final histogram is a vector h = [n₀, n₁, ..., n_{K-1}] representing counts per bin. The histogram is normalized so that Σ n_i = total sample count.

#### 3. **Expression Encoding**

The symbolic expression must be converted into a sequence of discrete tokens that the neural network can process:

- **Infix to Prefix Conversion**: The symbolic expression is converted from infix notation (human-readable) to prefix (Lisp-like) notation using preorder tree traversal:
  - Example: `a + b*x` (infix) → `['+', 'mul', 'a', 'b', 'x']` (prefix)
  - Prefix enables left-to-right decoding without parentheses

- **Tokenization**: Each element is mapped to a fixed vocabulary token:
  - **Special tokens**: `<START>`, `<END>`, `<PAD>`
  - **Operators**: `+`, `mul`, `pow`, `sqrt`, `log`, `sin`, `cos`
  - **Variables**: `x`
  - **Constants**: `pi`, `exp`
  - **Literals**: integers -5 to 5

- **Constant Encoding**: Floating-point coefficients are encoded in scientific notation to maintain a fixed vocabulary:
  - Each float is decomposed as: value = mantissa × 10^(exponent)
  - The exponent (range -4 to 4) becomes a token: `C-4, C-3, ..., C4`
  - The mantissa is stored as a regression target
  - Example: 0.00123 → `C-3` token + mantissa 1.23

#### 4. **Dataset Characteristics**

- **Variable Sequence Lengths**: Different expressions produce token sequences of different lengths (typically 5–50 tokens)
- **Variable Histogram Sizes**: K bins can be adjusted; default is K=32
- **Ground Truth Pairs**: Each record is (histogram, prefix_tokens, mantissas), where:
  - `histogram`: the empirical distribution (K-dimensional)
  - `prefix_tokens`: the encoded expression sequence
  - `mantissas`: the regression targets for floating-point constants

- **Distribution Diversity**: The compositional generation ensures diverse mathematical functions (polynomials, exponentials, trigonometric, hybrid combinations) across the dataset

### Key Design Decisions

1. **Poisson Sampling**: Creates realistic noise consistent with empirical histogram estimation from finite samples
2. **Prefix Notation**: Enables autoregressive decoding and simplifies grammar (no parentheses required)
3. **Constant Encoding**: Keeps vocabulary size fixed while supporting arbitrary float magnitudes
4. **Compositional Synthesis**: Generates complex expressions from simple primitives, creating natural complexity distribution

---

## Model Architecture

### High-Level Overview

FASEROH is a sequence-to-sequence neural network with an asymmetric encoder-decoder architecture designed to map empirical histograms to symbolic mathematical expressions.

### Core Components

#### **1. Histogram Encoder**

The encoder processes the input histogram and extracts a learned latent representation:

- **Linear Projection**: Input histogram (K-dimensional) is projected to d_model dimensions
- **Convolutional Layers**: Two 1D convolutional layers with kernel size 3 capture local bin patterns and relationships
- **Transformer Encoder**: A stack of transformer encoder blocks with multi-head self-attention and feed-forward networks
  - Allows the model to learn long-range dependencies across bins
  - Standard attention mechanisms enable focus on important histogram regions
- **Cross-Attention Pooling**: The encoder output is aggregated into a fixed-size latent vector via cross-attention pooling
  - Learns which histogram regions are most relevant for expression generation
  - Output shape: (batch_size, n_latent, d_model)

#### **2. Symbolic Decoder**

The decoder generates the output expression token-by-token using an autoregressive approach:

- **Transformer Decoder**: Causal transformer decoder with masked self-attention
  - Ensures each token only attends to previous tokens (autoregressive property)
  - Allows the model to incorporate full encoder context via cross-attention

- **Dual-Head Output**:
  - **Symbol Head**: Predicts the next token from the fixed vocabulary (softmax over ~50 tokens)
  - **Mantissa Head**: Predicts the floating-point value for constants that require regression
  - Both heads operate simultaneously but are independently trained

- **Autoregressive Sampling**: At inference, tokens are generated one at a time, with each new token conditioned on all previous tokens and the encoder output

#### **3. Loss Function & Training**

- **Token Prediction Loss**: Cross-entropy loss on symbol token predictions
- **Mantissa Regression Loss**: Mean squared error (MSE) on floating-point mantissas, but only evaluated at positions where a `C{ce}` token is predicted
- **Loss Weighting**: A scalar weight λ balances the two losses:
  ```
  Total Loss = CE_loss + λ × MSE_loss
  ```
  - λ is gradually increased from 0 to its target value over `lambda_warmup_epochs` to prioritize sequence learning first, then refine numeric accuracy

### Model Capacity Control

Configurable hyperparameters (in `FASeROHConfig`):
- `d_model`: Embedding dimension (default: 360)
- `n_heads`: Number of attention heads (default: 8)
- `n_enc_layers`: Transformer encoder depth (default: 4)
- `n_dec_layers`: Transformer decoder depth (default: 6)
- `n_latent`: Size of latent bottleneck (default: 32)
- `vocab_size`: Fixed vocabulary size (~50 tokens)

---

## Evaluation Metrics

### Overview

The evaluation framework assesses multiple aspects of model predictions: fidelity to the empirical data, structural correctness, and semantic validity. Each metric provides different insights into model performance.

### Metric Definitions & Interpretation

#### **1. R² (Coefficient of Determination)**

**Definition**: Measures how well the predicted symbolic function fits the original histogram data:
```
R² = 1 - (SS_res / SS_tot)
```
where SS_res is the sum of squared residuals between predicted and observed bin values, and SS_tot is the total variance of the observed histogram.

**Range**: (-∞, 1], where 1.0 = perfect fit, 0.0 = as good as mean baseline, <0 = worse than baseline

**Information**: Indicates how accurately the predicted expression captures the empirical distribution shape. High R² means the expression is numerically close to the histogram.

#### **2. Sentence Accuracy (Exact Match)**

**Definition**: Fraction of predictions whose token sequence exactly matches the ground truth, token-by-token, ignoring floating-point mantissa values.

**Range**: [0, 1], where 1.0 = all predictions exactly correct

**Information**: Measures whether the model recovered the exact symbolic structure. This is the strictest metric—minor token differences cause failure. Useful for understanding complete recovery rate.

#### **3. Prefix Validity**

**Definition**: Fraction of predictions that form syntactically valid prefix expressions—i.e., the token sequence can be parsed as a valid expression tree without mismatches in operator arity.

**Range**: [0, 1], where 1.0 = all predictions are syntactically valid

**Information**: Assesses whether the model respects the grammar constraints of the prefix notation. Invalid predictions indicate the model violated fundamental structural rules (e.g., wrong number of operands for an operator). A high validity score shows the model learned implicit grammar.

#### **4. Function Validity**

**Definition**: Fraction of predictions where the generated expression is mathematically valid and evaluable over [0, 1]—i.e., no division by zero, log of negative numbers, etc.

**Range**: [0, 1], where 1.0 = all functions are mathematically valid

**Information**: Measures whether predicted expressions are executable and numerically stable. Invalid functions cannot be evaluated to check R². A gap between "Prefix Valid" and "Function Valid" reveals domain-specific errors.

#### **5. Goodness of Fit (χ² / ndf)**

**Definition**: Chi-squared test statistic normalized by degrees of freedom:
```
χ²/ndf = (1/ndf) × Σ [(n_i - μ_i)² / μ_i]
```
where n_i is the observed count in bin i, μ_i is the expected count from the predicted function, and ndf = K - 1 - p (K bins, p fit parameters).

**Range**: [0, ∞), where values near 1.0 indicate good agreement, large values indicate poor fit

**Information**: Provides a statistical test of whether observed and predicted distributions are consistent. A value close to 1 suggests the predicted function explains the histogram well in a statistical sense. Values >> 1 suggest systematic mismatch; values << 1 suggest overfitting to noise.

### Combined Interpretation

- **Full Recovery**: Sentence Accuracy = 1.0 → the model perfectly reconstructed the symbolic expression
- **Robust Prediction**: High R² AND Prefix Valid AND Function Valid → the model generated an incorrect but still meaningful expression
- **Statistical Significance**: χ²/ndf ≈ 1 → the prediction is statistically indistinguishable from the true distribution
- **Structural Understanding**: Prefix Valid > Sentence Accuracy → the model understands expression grammar but makes minor token errors

### Configuration

Metrics to compute are specified in `config.eval_metrics` as a tuple in `FASeROHConfig`. The evaluation pipeline outputs all requested metrics for each prediction and aggregates them over the test set.
