"""
FASEROH Dataset Generator — Algorithm 1
========================================
Pipeline
--------
1.  generate_function()   – build + normalise a random symbolic f(x) over [0,1]
2.  validate_function()   – check positivity, finiteness, normalisation
3.  generate_histogram()  – Algorithm 1: integrate f(x) over K bins, Poisson sample
4.  infix_to_prefix()     – convert SymPy expr → preorder token list
5.  prefix_to_infix()     – reconstruct infix string from preorder tokens
6.  encode_constants()    – replace numeric tokens with C{ce} + mantissa
7.  encode_expression()   – end-to-end: SymPy expr → {tokens, mantissas}
8.  generate_dataset()    – batch generation
"""

import random
import math
import signal
import time
from typing import Optional

import numpy as np
from scipy import integrate as sci_integrate
import sympy as sp
from sympy import symbols

# ── Symbolic variable ──────────────────────────────────────────────────────────
x_sym = symbols("x")


# ══════════════════════════════════════════════════════════════════════════════
# Base function classes
# ══════════════════════════════════════════════════════════════════════════════

class BaseFunc:
    """Holds a callable numpy function and its SymPy skeleton."""

    def __init__(self, np_fn, sympy_expr):
        self.np_fn = np_fn          # np_fn(x: ndarray) -> ndarray
        self.sympy_expr = sympy_expr

    def __call__(self, x):
        return self.np_fn(x)


# ── Individual base function factories ────────────────────────────────────────

def make_uniform():
    return BaseFunc(
        np_fn=lambda x: np.ones_like(np.asarray(x, float)),
        sympy_expr=sp.Integer(1),
    )


def make_linear():
    a = sample_float_param(0.1, 3.0)
    b = sample_float_param(0.0, 3.0)
    return BaseFunc(
        np_fn=lambda x, a=a, b=b: a + b * np.asarray(x, float),
        sympy_expr=a + b * x_sym,
    )


def make_exponential():
    if random.random() < 0.80:
        lam = float(random.choice([-3, -2, -1, 1, 2, 3]))
    else:
        lam = round(random.uniform(-3.0, 3.0), 3)
        while abs(lam) < 0.1:
            lam = round(random.uniform(-3.0, 3.0), 3)
    return BaseFunc(
        np_fn=lambda x, lam=lam: np.exp(lam * np.asarray(x, float)),
        sympy_expr=sp.exp(lam * x_sym),
    )


def make_gaussian():
    mu = sample_float_param(0.15, 0.85)
    sigma = sample_float_param(0.10, 0.40)
    return BaseFunc(
        np_fn=lambda x, mu=mu, s=sigma: np.exp(
            -0.5 * ((np.asarray(x, float) - mu) / s) ** 2
        ),
        sympy_expr=sp.exp(-sp.Rational(1, 2) * ((x_sym - mu) / sigma) ** 2),
    )


def make_power_law():
    alpha = sample_float_param(-0.5, 3.0)
    return BaseFunc(
        np_fn=lambda x, a=alpha: np.where(
            np.asarray(x, float) > 0,
            np.abs(np.asarray(x, float)) ** a,
            0.0,
        ),
        sympy_expr=x_sym ** alpha,
    )


def make_sine_bump():
    if random.random() < 0.80:
        n = random.choice([1, 2, 3])
    else:
        n = max(1, round(abs(random.uniform(0.5, 3.0))))
    return BaseFunc(
        np_fn=lambda x, n=n: np.sin(n * math.pi * np.asarray(x, float)) ** 2,
        sympy_expr=sp.sin(n * sp.pi * x_sym) ** 2,
    )


def make_cosine_arch():
    if random.random() < 0.80:
        n = random.choice([1, 2])
    else:
        n = max(1, round(abs(random.uniform(0.5, 2.5))))
    return BaseFunc(
        np_fn=lambda x, n=n: 1.0 + np.cos(n * math.pi * np.asarray(x, float)),
        sympy_expr=1 + sp.cos(n * sp.pi * x_sym),
    )


def make_beta_kernel():
    a = sample_float_param(0.5, 2.5)
    b = sample_float_param(0.5, 2.5)

    def np_fn(x, a=a, b=b):
        xv = np.asarray(x, float)
        return np.where(
            (xv > 0) & (xv < 1),
            xv ** (a - 1) * (1 - xv) ** (b - 1),
            0.0,
        )

    return BaseFunc(
        np_fn=np_fn,
        sympy_expr=x_sym ** (a - 1) * (1 - x_sym) ** (b - 1),
    )


def make_polynomial():
    degree = random.choice([2, 3])
    coeffs = [sample_float_param(-2.0, 2.0) for _ in range(degree + 1)]
    while abs(coeffs[0]) < 0.1:
        coeffs[0] = sample_float_param(-2.0, 2.0)

    def np_fn(x, c=coeffs):
        xv = np.asarray(x, float)
        return sum(ci * xv ** i for i, ci in enumerate(c))

    return BaseFunc(
        np_fn=np_fn,
        sympy_expr=sum(c * x_sym ** i for i, c in enumerate(coeffs)),
    )


def make_breit_wigner():
    x0 = sample_float_param(0.2, 0.8)
    gam = sample_float_param(0.15, 0.40)
    return BaseFunc(
        np_fn=lambda x, x0=x0, g=gam: g
        / ((np.asarray(x, float) - x0) ** 2 + (g / 2) ** 2),
        sympy_expr=gam / ((x_sym - x0) ** 2 + (gam / 2) ** 2),
    )


def make_log_bump():
    mu = sample_float_param(-1.0, 0.5)
    sigma = sample_float_param(0.3, 0.8)
    eps = 0.001

    def np_fn(x, mu=mu, sigma=sigma, eps=eps):
        xv = np.asarray(x, float)
        return np.exp(-0.5 * ((np.log(xv + eps) - mu) / sigma) ** 2)

    return BaseFunc(
        np_fn=np_fn,
        sympy_expr=sp.exp(
            -sp.Rational(1, 2) * ((sp.log(x_sym + eps) - mu) / sigma) ** 2
        ),
    )


# ── Constant vocabulary ────────────────────────────────────────────────────────

SMALL_INTS: set = set(range(-5, 6))   # -5 … 5  (own vocab tokens)


def sample_unary_multiplier(allow_zero: bool = False) -> float:
    if random.random() < 0.80:
        candidates = [i for i in range(-5, 6) if (allow_zero or i != 0)]
        return float(random.choice(candidates))
    else:
        val = round(random.uniform(-5.0, 5.0), 3)
        if not allow_zero and abs(val) < 1e-9:
            val = round(random.uniform(0.1, 5.0), 3)
        return val


def sample_float_param(low: float, high: float) -> float:
    return round(random.uniform(low, high), 3)


def sample_weight() -> float:
    val = round(random.uniform(-5.0, 5.0), 3)
    while abs(val) < 0.1:
        val = round(random.uniform(-5.0, 5.0), 3)
    return val


def make_constant():
    c = sample_float_param(-5.0, 5.0)
    while abs(c) < 1e-9:
        c = sample_float_param(-5.0, 5.0)
    return BaseFunc(
        np_fn=lambda x, c=c: c * np.ones_like(np.asarray(x, float)),
        sympy_expr=sp.Float(c),
    )


BASE_FACTORIES = [
    make_uniform,
    make_linear,
    make_exponential,
    make_gaussian,
    make_power_law,
    make_sine_bump,
    make_cosine_arch,
    make_beta_kernel,
    make_polynomial,
    make_breit_wigner,
    make_log_bump,
    make_constant,
]

FACTORY_WEIGHTS = [
    1,   # uniform
    1,   # linear
    3,   # exponential
    3,   # gaussian
    4,   # power_law
    2,   # sine_bump
    2,   # cosine_arch
    2,   # beta_kernel
    4,   # polynomial
    1,   # breit_wigner
    2,   # log_bump
    3,   # constant
]

FACTORY_NAME_MAP: dict = {
    "uniform":      (make_uniform,      1),
    "linear":       (make_linear,       1),
    "exponential":  (make_exponential,  3),
    "gaussian":     (make_gaussian,     3),
    "power_law":    (make_power_law,    4),
    "sine_bump":    (make_sine_bump,    2),
    "cosine_arch":  (make_cosine_arch,  2),
    "beta_kernel":  (make_beta_kernel,  2),
    "polynomial":   (make_polynomial,   4),
    "breit_wigner": (make_breit_wigner, 1),
    "log_bump":     (make_log_bump,     2),
    "constant":     (make_constant,     3),
}


def get_filtered_factories(allowed: list[str]) -> tuple[list, list]:
    factories, weights = [], []
    for name in allowed:
        if name not in FACTORY_NAME_MAP:
            raise ValueError(
                f"Unknown factory name '{name}'. "
                f"Valid names: {sorted(FACTORY_NAME_MAP.keys())}"
            )
        fn, w = FACTORY_NAME_MAP[name]
        factories.append(fn)
        weights.append(w)
    return factories, weights


OPERATORS = ["+", "*", "-"]


# ══════════════════════════════════════════════════════════════════════════════
# Composed function
# ══════════════════════════════════════════════════════════════════════════════

class ComposedFunc:
    """List of (operator, BaseFunc, weight) triples."""

    def __init__(self):
        self.terms: list = []

    def add(self, op: str, func: BaseFunc, weight: float = 1.0):
        self.terms.append((op, func, weight))

    def normalise(self, area_under_curve: float):
        self.terms = [
            (op, fn, w / area_under_curve)
            for op, fn, w in self.terms
        ]

    def __call__(self, x):
        xv = np.asarray(x, float)
        op0, f0, w0 = self.terms[0]
        result = f0(xv) * w0
        for op, fn, w in self.terms[1:]:
            b = fn(xv)
            if op == "+":
                result = result + w * b
            elif op == "*":
                result = result * b
            elif op == "-":
                result = result - w * b
        return result

    def sympy_expr(self):
        op0, f0, w0 = self.terms[0]
        expr = f0.sympy_expr * w0 if w0 != 1.0 else f0.sympy_expr
        for op, fn, w in self.terms[1:]:
            b = fn.sympy_expr
            if op == "+":
                expr = expr + w * b
            elif op == "*":
                expr = expr * b
            elif op == "-":
                expr = expr - w * b
        return expr


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — generate_function
# ══════════════════════════════════════════════════════════════════════════════

def generate_function(
    max_components: int = 3,
    max_attempts: int = 100,
    allowed_factories: Optional[list[str]] = None,
) -> Optional[dict]:
    """Build and normalise a random symbolic f(x) over [0,1].

    Returns
    -------
    dict:
        numpy_fn   – ComposedFunc (normalised callable)
        sympy_expr – SymPy expression (used internally for encoding)
    or None if all attempts fail.
    """
    if allowed_factories is not None:
        filt_factories, filt_weights = get_filtered_factories(allowed_factories)
    else:
        filt_factories, filt_weights = BASE_FACTORIES, FACTORY_WEIGHTS

    for _ in range(max_attempts):
        try:
            m = random.randint(1, max_components)

            comp = ComposedFunc()
            comp.add(
                "+",
                random.choices(filt_factories, weights=filt_weights, k=1)[0](),
                weight=1.0,
            )

            for _ in range(m - 1):
                op = random.choice(OPERATORS)
                w = sample_weight() if op == "+" else 1.0
                comp.add(
                    op,
                    random.choices(filt_factories, weights=filt_weights, k=1)[0](),
                    weight=w,
                )

            # Quick pre-check on dense grid
            xs = np.linspace(1e-6, 1 - 1e-6, 500)
            ys = comp(xs)
            if not np.all(np.isfinite(ys)):
                continue
            if np.any(ys < 0):
                continue
            if np.max(ys) < 1e-10:
                continue

            # Integrate
            I, _ = sci_integrate.quad(
                comp, 0.0, 1.0, limit=200, epsabs=1e-7, epsrel=1e-7
            )
            if not (math.isfinite(I) and I > 0):
                continue

            if I > 500 or I < 0.001:
                continue

            comp.normalise(I)

            # Build SymPy expression with 5-second timeout
            raw_expr = comp.sympy_expr() / I

            def _alarm(signum, frame):
                raise TimeoutError

            signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(5)
            try:
                sym_expr = sp.simplify(raw_expr)
            except (TimeoutError, Exception):
                sym_expr = raw_expr
            finally:
                signal.alarm(0)

            if sym_expr is None:
                continue

            return {
                "numpy_fn":   comp,
                "sympy_expr": sym_expr,
            }

        except Exception:
            continue

    return None


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — validate_function
# ══════════════════════════════════════════════════════════════════════════════

def validate_function(result: dict, tol: float = 0.03) -> tuple:
    """Validate a generated function dict.

    Checks: callable, finite, non-negative, normalised to 1, sympy_expr not None.

    Returns (is_valid: bool, reason: str).
    """
    if result is None:
        return False, "result is None"

    fn = result.get("numpy_fn")
    if fn is None or not callable(fn):
        return False, "numpy_fn is not callable"

    xs = np.linspace(1e-6, 1 - 1e-6, 1000)
    try:
        ys = fn(xs)
    except Exception as e:
        return False, f"evaluation error: {e}"

    if not np.all(np.isfinite(ys)):
        n_bad = int(np.sum(~np.isfinite(ys)))
        return False, f"non-finite values at {n_bad} points"

    if np.any(ys < 0):
        n_neg = int(np.sum(ys < 0))
        return False, f"negative values at {n_neg} points"

    try:
        I, _ = sci_integrate.quad(fn, 0.0, 1.0, limit=200, epsabs=1e-7, epsrel=1e-7)
    except Exception as e:
        return False, f"integration error: {e}"

    if not math.isfinite(I):
        return False, f"integral is not finite: {I}"

    if not (1.0 - tol <= I <= 1.0 + tol):
        return False, f"normalisation check failed: ∫f dx = {I:.6f}"

    if result.get("sympy_expr") is None:
        return False, "sympy_expr is None"

    return True, "OK"


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — generate_histogram  (Algorithm 1)
# ══════════════════════════════════════════════════════════════════════════════

def generate_histogram(
    numpy_fn,
    N: Optional[int] = None,
    K: Optional[int] = None,
    n_min: int = 100,
    n_max: int = 10_000,
    k_min: int = 10,
    k_max: int = 100,
) -> dict:
    """Algorithm 1: generate a histogram H from normalised f(x).

    For each bin k:
        n_k = N * integral of f(x) over bin   (expected count)
        N_k ~ Poisson(n_k)                     (sampled count)

    Returns
    -------
    dict:
        bins – ndarray (K,) integer bin counts N_k
        N    – total event count used
        K    – number of bins used
    """
    if N is None:
        N = random.randint(n_min, n_max)
    if K is None:
        K = random.randint(k_min, k_max)

    edges = np.linspace(0.0, 1.0, K + 1)
    means = np.empty(K, dtype=float)

    for k in range(K):
        integral, _ = sci_integrate.quad(
            numpy_fn, edges[k], edges[k + 1], limit=50, epsabs=1e-6, epsrel=1e-6
        )
        means[k] = max(N * integral, 0.0)

    bins = np.random.poisson(means).astype(int)

    return {
        "bins": bins,
        "N":    N,
        "K":    K,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Step 3b — goodness_of_fit  (Section 2.3)
# ══════════════════════════════════════════════════════════════════════════════

def goodness_of_fit(
    numpy_fn, histogram: dict, n_free_params: int = 0, n_pts_per_bin: int = 20
) -> dict:
    """Compute chi-squared goodness-of-fit statistic for a histogram vs f(x).

    X = sum_k (N_k - n_k)^2 / n_k   (bins with N_k >= 5 only)
    ndf = K_valid - n_free_params
    A good fit has X / ndf ≈ 1.

    Uses a single vectorised numpy evaluation over a (K, n_pts_per_bin+1) grid
    and np.trapz for integration — no per-bin quad calls.

    Parameters
    ----------
    numpy_fn      : normalised callable f(x) over [0, 1]
    histogram     : dict {bins, N, K} from generate_histogram
    n_free_params : P, number of free parameters in the fitted function
    n_pts_per_bin : points per bin for trapezoidal integration (default 20)

    Returns
    -------
    dict:
        X              – raw chi-squared sum
        ndf            – degrees of freedom (K_valid - P)
        X_per_ndf      – X / ndf  (≈ 1 for a good fit)
        n_bins_used    – number of bins included (N_k >= 5)
        n_bins_skipped – bins excluded (N_k < 5)
    """
    bins = histogram["bins"]
    N    = histogram["N"]
    K    = histogram["K"]

    # Build a (K, m+1) grid of x values — one row per bin, no shared edges needed
    m       = n_pts_per_bin
    edges   = np.linspace(0.0, 1.0, K + 1)          # (K+1,)
    t       = np.linspace(0.0, 1.0, m + 1)           # (m+1,) normalised positions
    bin_w   = edges[1:] - edges[:-1]                  # (K,)  all equal = 1/K
    x_grid  = edges[:-1, None] + t[None, :] * bin_w[:, None]  # (K, m+1)

    # Single function evaluation over all K*(m+1) points
    y_grid  = numpy_fn(x_grid.ravel()).reshape(K, m + 1)  # (K, m+1)

    # Vectorised trapezoidal integration per bin
    integrals = np.trapz(y_grid, dx=bin_w[0] / m, axis=1)  # (K,)
    n_k = N * integrals                                      # (K,) expected counts

    N_k  = bins.astype(float)
    mask = (N_k >= 5) & (n_k > 0)

    diff     = N_k[mask] - n_k[mask]
    X        = float(np.sum(diff ** 2 / n_k[mask]))
    n_used   = int(mask.sum())
    ndf      = max(n_used - n_free_params, 1)

    return {
        "X":               X,
        "ndf":             ndf,
        "X_per_ndf":       X / ndf,
        "n_bins_used":     n_used,
        "n_bins_skipped":  K - n_used,
    }


def count_free_params(encoding: dict) -> int:
    """Count the floating-point constants in an encoded expression (P for GoF)."""
    return sum(
        1 for tok in encoding["tokens"]
        if tok.startswith("C") and tok != "C0"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — infix_to_prefix  (preorder traversal)
# ══════════════════════════════════════════════════════════════════════════════

def _collect_prefix(expr, tokens: list, mantissas: list) -> None:
    """Recursive preorder walk of a SymPy expression tree."""

    # ── mathematical constants: pi and e (Euler's number) ────────────────────
    if expr is sp.pi:
        tokens.append("pi")
        mantissas.append(0.0)
        return

    if expr is sp.E:
        tokens.append("e")
        mantissas.append(0.0)
        return

    # ── numbers / constants ──────────────────────────────────────────────────
    if expr.is_number or not expr.free_symbols:
        try:
            c = float(expr)
            tokens.append(_num_placeholder(c))
            mantissas.append(c)
            return
        except (TypeError, ValueError):
            pass

    # ── symbolic variable x ──────────────────────────────────────────────────
    if isinstance(expr, sp.Symbol):
        tokens.append("x")
        mantissas.append(0.0)
        return

    # ── addition ─────────────────────────────────────────────────────────────
    if isinstance(expr, sp.Add):
        args = list(expr.args)
        tokens.append("+")
        mantissas.append(0.0)
        _collect_prefix(args[0], tokens, mantissas)
        rest = sp.Add(*args[1:]) if len(args) > 2 else args[1]
        _collect_prefix(rest, tokens, mantissas)
        return

    # ── multiplication ───────────────────────────────────────────────────────
    if isinstance(expr, sp.Mul):
        args = list(expr.args)
        tokens.append("mul")
        mantissas.append(0.0)
        _collect_prefix(args[0], tokens, mantissas)
        rest = sp.Mul(*args[1:]) if len(args) > 2 else args[1]
        _collect_prefix(rest, tokens, mantissas)
        return

    # ── power / sqrt ─────────────────────────────────────────────────────────
    if isinstance(expr, sp.Pow):
        if expr.exp == sp.Rational(1, 2):
            tokens.append("sqrt")
            mantissas.append(0.0)
            _collect_prefix(expr.base, tokens, mantissas)
        else:
            tokens.append("pow")
            mantissas.append(0.0)
            _collect_prefix(expr.base, tokens, mantissas)
            _collect_prefix(expr.exp, tokens, mantissas)
        return

    # ── exp(arg) → pow e arg  (e is the Euler constant, a leaf token) ────
    if isinstance(expr, sp.exp):
        tokens.append("pow")
        mantissas.append(0.0)
        tokens.append("e")
        mantissas.append(0.0)
        _collect_prefix(expr.args[0], tokens, mantissas)
        return

    # ── named unary functions ─────────────────────────────────────────────────
    _FUNC_MAP = {
        sp.log:  "log",
        sp.sin:  "sin",
        sp.cos:  "cos",
    }
    for func_cls, tok in _FUNC_MAP.items():
        if isinstance(expr, func_cls):
            tokens.append(tok)
            mantissas.append(0.0)
            for arg in expr.args:
                _collect_prefix(arg, tokens, mantissas)
            return

    # ── fallback ──────────────────────────────────────────────────────────────
    tokens.append("?")
    mantissas.append(0.0)


def _num_placeholder(c: float) -> str:
    """Return a raw string placeholder for a numeric constant (pre-encoding)."""
    c_int = int(round(c))
    if c == float(c_int) and c_int in SMALL_INTS:
        return str(c_int)
    return f"NUM({c})"


def infix_to_prefix(sympy_expr) -> dict:
    """Convert a SymPy infix expression to a preorder (prefix) token list.

    Returns
    -------
    dict:
    
        tokens        – list[str]
        raw_mantissas – list[float] (0.0 for non-constants)
    """
    tokens: list = []
    mantissas: list = []
    _collect_prefix(sympy_expr, tokens, mantissas)
    return {"tokens": tokens, "raw_mantissas": mantissas}


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — prefix_to_infix
# ══════════════════════════════════════════════════════════════════════════════

_ARITY = {
    "+":    2,
    "mul":  2,
    "pow":  2,
    "sqrt": 1,
    "log":  1,
    "sin":  1,
    "cos":  1,
    "?":    0,
}


def prefix_to_infix(tokens: list, mantissas: Optional[list] = None) -> str:
    """Reconstruct a human-readable infix string from a preorder token list."""
    resolved: list = []
    for i, tok in enumerate(tokens):
        if mantissas and (tok.startswith("C") or tok.startswith("NUM")):
            val = mantissas[i]
            resolved.append(f"{val:.6g}")
        else:
            resolved.append(tok)

    stack: list = []
    for tok in reversed(resolved):
        arity = _ARITY.get(tok, None)

        if arity is None:
            stack.append(tok)
            continue

        if arity == 0:
            stack.append("?")
        elif arity == 1:
            arg = stack.pop() if stack else "?"
            if tok == "sqrt":
                stack.append(f"sqrt({arg})")
            else:
                stack.append(f"{tok}({arg})")
        elif arity == 2:
            a = stack.pop() if stack else "?"
            b = stack.pop() if stack else "?"
            if tok == "+":
                stack.append(f"({a} + {b})")
            elif tok == "mul":
                stack.append(f"({a} * {b})")
            elif tok == "pow":
                stack.append(f"({a} ** {b})")
            else:
                stack.append(f"{tok}({a}, {b})")

    return stack[0] if stack else ""


# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — encode_constants
# ══════════════════════════════════════════════════════════════════════════════

def encode_constant(c: float) -> tuple:
    """Encode a single float constant to (token, mantissa).

    Small integers in [-5, 5] → own vocab token e.g. '3', '-2'; mantissa = 0.0
    Zero                      → 'C0'; mantissa = 0.0
    Other values              → 'C{ce}' where ce = floor(log10|c|) + 1
                                mantissa cm = c / 10^ce, so |cm| ∈ [0.1, 1)
    """
    c_int = int(round(c))
    if c == float(c_int) and c_int in SMALL_INTS:
        return str(c_int), 0.0

    if c == 0.0:
        return "C0", 0.0

    abs_c = abs(c)
    ce = int(math.floor(math.log10(abs_c))) + 1
    ce = max(-4, min(4, ce))
    cm = c / (10.0 ** ce)
    return f"C{ce}", round(cm, 8)


def encode_constants(tokens: list, raw_mantissas: list) -> dict:
    """Replace raw numeric placeholders in a prefix token list with C{ce} tokens.

    Returns
    -------
    dict:
        tokens    – list[str]   final token sequence
        mantissas – list[float] parallel mantissa values (0.0 for non-constants)
    """
    out_tokens: list = []
    out_mantissas: list = []

    for tok, raw in zip(tokens, raw_mantissas):
        if tok.startswith("NUM(") or (
            tok not in _ARITY
            and tok != "x"
            and tok not in ("pi", "e")
            and tok not in [str(i) for i in range(-5, 6)]
        ):
            tok_enc, cm = encode_constant(raw)
            out_tokens.append(tok_enc)
            out_mantissas.append(cm)
        elif tok in [str(i) for i in range(-5, 6)]:
            out_tokens.append(tok)
            out_mantissas.append(0.0)
        else:
            # operator, variable, or math constant (pi, e)
            out_tokens.append(tok)
            out_mantissas.append(0.0)

    return {"tokens": out_tokens, "mantissas": out_mantissas}


# ══════════════════════════════════════════════════════════════════════════════
# Step 7 — encode_expression  (end-to-end)
# ══════════════════════════════════════════════════════════════════════════════

def encode_expression(sympy_expr) -> dict:
    """Convert a SymPy expression to a preorder token sequence with encoded constants.

    Returns
    -------
    dict:
        tokens    – list[str]
        mantissas – list[float]
    """
    prefix = infix_to_prefix(sympy_expr)
    return encode_constants(prefix["tokens"], prefix["raw_mantissas"])


# ══════════════════════════════════════════════════════════════════════════════
# Step 8 — generate_dataset
# ══════════════════════════════════════════════════════════════════════════════

def generate_dataset(
    n: int = 100,
    max_components: int = 3,
    verbose: bool = False,
    seed: Optional[int] = None,
    n_min: int = 100,
    n_max: int = 10_000,
    k_min: int = 10,
    k_max: int = 100,
    allowed_factories: Optional[list[str]] = None,
) -> list:
    """Generate n validated (function + histogram + encoding) records.

    Each record contains only what is needed for the training pipeline:
        expr_str  – human-readable expression string
        histogram – dict {bins, N, K}
        encoding  – dict {tokens, mantissas}

    Parameters
    ----------
    n             : target number of valid records
    max_components: max base functions per sample
    verbose       : print per-step timing
    seed          : random seed for reproducibility
    n_min/max     : range for histogram total count N
    k_min/max     : range for histogram bin count K
    allowed_factories : restrict base function types (None = all)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    dataset: list = []
    attempts: int = 0
    t_dataset_start = time.perf_counter()

    while len(dataset) < n:
        attempts += 1

        # ── Step A: generate ────────────────────────────────────────────────
        t0 = time.perf_counter()
        result = generate_function(
            max_components=max_components,
            allowed_factories=allowed_factories,
        )
        t_gen = time.perf_counter() - t0

        if result is None:
            if verbose:
                print(f"  attempt {attempts:>5} | generate: FAILED ({t_gen*1e3:.1f} ms)")
            continue

        # ── Step B: validate ────────────────────────────────────────────────
        t1 = time.perf_counter()
        is_valid, reason = validate_function(result)
        t_val = time.perf_counter() - t1

        if not is_valid:
            if verbose:
                print(
                    f"  attempt {attempts:>5} | generate: {t_gen*1e3:.1f} ms | "
                    f"validate: FAILED ({reason}) ({t_val*1e3:.1f} ms)"
                )
            continue

        # ── Step C: histogram (Algorithm 1) ─────────────────────────────────
        t2 = time.perf_counter()
        histogram = generate_histogram(
            result["numpy_fn"],
            n_min=n_min, n_max=n_max,
            k_min=k_min, k_max=k_max,
        )
        t_hist = time.perf_counter() - t2

        # ── Step D: encoding + normalised expr_str ───────────────────────────
        t3 = time.perf_counter()
        encoding = (
            encode_expression(result["sympy_expr"])
            if result["sympy_expr"] is not None
            else {"tokens": [], "mantissas": []}
        )

        # Validate prefix structure after encoding; skip invalid records
        from tokenizer import is_valid_prefix as _is_valid_prefix
        if not _is_valid_prefix(encoding["tokens"]):
            if verbose:
                print(
                    f"  attempt {attempts:>5} | encode: invalid prefix "
                    f"(tokens={encoding['tokens'][:10]}...)"
                )
            continue

        expr_str = prefix_to_infix(encoding["tokens"], encoding["mantissas"])
        t_enc = time.perf_counter() - t3

        # ── Step E: goodness-of-fit (Section 2.3) ────────────────────────────
        t4 = time.perf_counter()
        P = count_free_params(encoding)
        gof = goodness_of_fit(result["numpy_fn"], histogram, n_free_params=P)
        t_gof = time.perf_counter() - t4

        dataset.append({
            "expr_str":  expr_str,
            "histogram": histogram,
            "encoding":  encoding,
            "gof":       gof,
        })

        if verbose:
            idx = len(dataset)
            print(
                f"  [{idx:>4}/{n}]  attempt {attempts:>5} | "
                f"generate: {t_gen*1e3:.1f} ms | "
                f"validate: {t_val*1e3:.1f} ms | "
                f"histogram: {t_hist*1e3:.1f} ms | "
                f"encoding: {t_enc*1e3:.1f} ms | "
                f"gof: X/ndf={gof['X_per_ndf']:.3f} ({t_gof*1e3:.1f} ms)"
            )

    t_total = time.perf_counter() - t_dataset_start
    rate = n / attempts * 100
    print(
        f"\nGenerated {n} records | "
        f"success rate {rate:.1f}% ({attempts} total attempts) | "
        f"total time {t_total:.2f}s"
    )
    return dataset


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    POC_FACTORIES = ["sine_bump", "cosine_arch", "exponential", "polynomial"]
    print(f"POC demo — generating 10 samples with factories: {POC_FACTORIES}\n")
    samples = generate_dataset(
        n=10,
        max_components=2,
        verbose=True,
        seed=42,
        n_min=500,
        n_max=5000,
        k_min=20,
        k_max=60,
        allowed_factories=POC_FACTORIES,
    )
    print("\n--- Expressions, token sequences, and GoF ---")
    for i, rec in enumerate(samples, 1):
        toks = rec["encoding"]["tokens"]
        mant = rec["encoding"]["mantissas"]
        hist = rec["histogram"]
        gof  = rec["gof"]
        print(f"  {i:>2}. expr     = {rec['expr_str']}")
        print(f"      tokens   = {toks}")
        print(f"      mantissas= {[round(m, 4) for m in mant]}")
        print(f"      N={hist['N']}, K={hist['K']}, bins[:5]={hist['bins'][:5].tolist()}")
        print(
            f"      GoF: X={gof['X']:.2f}, ndf={gof['ndf']}, "
            f"X/ndf={gof['X_per_ndf']:.3f} "
            f"(bins used={gof['n_bins_used']}, skipped={gof['n_bins_skipped']})\n"
        )
