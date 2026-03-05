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
