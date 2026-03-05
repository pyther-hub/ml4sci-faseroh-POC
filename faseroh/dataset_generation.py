"""
FASEROH Function Generator
==========================
Implements Algorithm 1 and Algorithm 2 from the FASEROH paper:
  "Fast Accurate Symbolic Empirical Representation Of Histograms"

Pipeline
--------
1.  generate_function()   – build + normalise a random symbolic f(x) over [0,1]
2.  validate_function()   – check positivity, finiteness, normalisation
3.  generate_histogram()  – Algorithm 1: sample N counts into K bins from f(x)
4.  infix_to_prefix()     – convert SymPy expr → preorder token list
5.  prefix_to_infix()     – reconstruct infix string from preorder tokens
6.  encode_constants()    – replace numeric tokens with C{ce} + mantissa
7.  encode_expression()   – end-to-end: SymPy expr → {tokens, mantissas}
8.  generate_dataset()    – batch generation with verbose timing
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
    # lam is a unary multiplier inside exp(lam*x) — invariant to normalisation.
    # Restricted to [-3, 3] (non-zero) so exp(lam*x) stays well-behaved over
    # [0,1]: exp(3) ≈ 20, small enough to avoid near-zero post-norm coefficients.
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
    # mu ∈ [0.15, 0.85] keeps the peak visible within [0,1].
    # sigma ∈ [0.10, 0.40] — lower bound raised from 0.05 to prevent the exponent
    # amplifying to extreme values (e.g. -88x²) when composed via * or /.
    mu = sample_float_param(0.15, 0.85)
    sigma = sample_float_param(0.10, 0.40)
    return BaseFunc(
        np_fn=lambda x, mu=mu, s=sigma: np.exp(
            -0.5 * ((np.asarray(x, float) - mu) / s) ** 2
        ),
        sympy_expr=sp.exp(-sp.Rational(1, 2) * ((x_sym - mu) / sigma) ** 2),
    )


def make_power_law():
    # alpha ∈ (-0.5, 3.0): negative values give integrable divergence at 0,
    # upper bound lowered from 4.0 to 3.0 to avoid steeply vanishing shapes.
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
    # n is a unary multiplier inside sin(n*pi*x)² — invariant to normalisation.
    # Capped at 3: sin(4πx)² already has 4 bumps over [0,1], higher n causes
    # rapid oscillation that is hard to represent with moderate K bins.
    if random.random() < 0.80:
        n = random.choice([1, 2, 3])
    else:
        n = max(1, round(abs(random.uniform(0.5, 3.0))))
    return BaseFunc(
        np_fn=lambda x, n=n: np.sin(n * math.pi * np.asarray(x, float)) ** 2,
        sympy_expr=sp.sin(n * sp.pi * x_sym) ** 2,
    )


def make_cosine_arch():
    # n is a unary multiplier inside cos(n*pi*x) — invariant to normalisation.
    # Kept at {1, 2}: 1+cos(πx) is one smooth arch, 1+cos(2πx) is two arches.
    if random.random() < 0.80:
        n = random.choice([1, 2])
    else:
        n = max(1, round(abs(random.uniform(0.5, 2.5))))
    return BaseFunc(
        np_fn=lambda x, n=n: 1.0 + np.cos(n * math.pi * np.asarray(x, float)),
        sympy_expr=1 + sp.cos(n * sp.pi * x_sym),
    )


def make_beta_kernel():
    # a, b ∈ [0.5, 2.5]: upper bound tightened from 5.0 — large exponents
    # produce ∫f dx in the hundreds of thousands, giving extreme post-norm coefficients.
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
    coeffs = [max(sample_float_param(0.0, 2.0), 0.0) for _ in range(degree + 1)]
    coeffs[0] = max(coeffs[0], 0.1)

    def np_fn(x, c=coeffs):
        xv = np.asarray(x, float)
        return sum(ci * xv ** i for i, ci in enumerate(c))

    return BaseFunc(
        np_fn=np_fn,
        sympy_expr=sum(c * x_sym ** i for i, c in enumerate(coeffs)),
    )


def make_breit_wigner():
    # x0 ∈ [0.2, 0.8]: keeps peak well within [0,1].
    # gam ∈ [0.15, 0.4]: lower bound raised from 0.05 — narrow widths produce
    # spikes that dominate the shape and are numerically fragile in composition.
    x0 = sample_float_param(0.2, 0.8)
    gam = sample_float_param(0.15, 0.40)
    return BaseFunc(
        np_fn=lambda x, x0=x0, g=gam: g
        / ((np.asarray(x, float) - x0) ** 2 + (g / 2) ** 2),
        sympy_expr=gam / ((x_sym - x0) ** 2 + (gam / 2) ** 2),
    )


def make_log_bump():
    # sigma ∈ [0.3, 0.8]: upper bound lowered slightly for smoother shapes.
    # Offset 0.001 replaces 1e-6 — large enough to display cleanly (not 0.000)
    # while still keeping log(x + 0.001) ≈ log(x) for x >> 0.001.
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
    """Sampler for integer multipliers inside unary functions (e.g. sin(nx), exp(λx)).

    These parameters sit inside f(c·x) and are therefore invariant to the
    normalisation step — dividing f by ∫f dx does not change the multiplier c.
    So integer tokens here are genuinely meaningful.

    80 % → non-zero integer from {-5, …, 5}
    20 % → float sampled uniformly from (-5, 5)
    """
    if random.random() < 0.80:
        candidates = [i for i in range(-5, 6) if (allow_zero or i != 0)]
        return float(random.choice(candidates))
    else:
        val = round(random.uniform(-5.0, 5.0), 3)
        if not allow_zero and abs(val) < 1e-9:
            val = round(random.uniform(0.1, 5.0), 3)
        return val


def sample_float_param(low: float, high: float) -> float:
    """Sampler for shape/position parameters (mu, sigma, alpha, etc.).

    These are always floats — they do not survive normalisation as integers
    and have no reason to be constrained to the integer vocabulary.
    """
    return round(random.uniform(low, high), 3)


def sample_weight() -> float:
    """Sampler for binary operator scale weights ω (used only with '+').

    Floats only, uniformly from (-5, 5) per the constant sampling spec.
    Zero is excluded since a zero weight nullifies a term entirely.
    """
    val = round(random.uniform(-5.0, 5.0), 3)
    while abs(val) < 0.1:
        val = round(random.uniform(-5.0, 5.0), 3)
    return val


def make_constant():
    """Base function that is a non-zero float constant (scalar).

    Deliberately always a float — a bare constant c becomes c/I after
    normalisation, which is not an integer in general.
    """
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
    3,   # constant
]

# ── Name-to-factory mapping for config-driven filtering ──────────────────────
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
    "constant":     (make_constant,     3),
}


def get_filtered_factories(allowed: list[str]) -> tuple[list, list]:
    """Return a filtered (factories, weights) pair for the given names.

    Parameters
    ----------
    allowed : list of factory name strings, e.g. ["sine_bump", "exponential"]
              Valid names: "uniform", "linear", "exponential", "gaussian",
              "power_law", "sine_bump", "cosine_arch", "beta_kernel",
              "polynomial", "breit_wigner", "constant"

    Returns
    -------
    (filtered_factories, filtered_weights)
    """
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

# Division removed from composition operators — dividing by a base function
# (especially power laws, log bumps, or Breit-Wigners) routinely produces
# extreme normalisation constants and near-zero coefficients after normalisation.
# The two remaining operators + and * are sufficient for moderate complexity.
OPERATORS = ["+", "*"]


# ══════════════════════════════════════════════════════════════════════════════
# Composed function
# ══════════════════════════════════════════════════════════════════════════════

class ComposedFunc:
    """List of (operator, BaseFunc, weight) triples.

    Call normalise(area) to bake normalisation into term weights so that
    ∫₀¹ f(x) dx = 1.
    """

    def __init__(self):
        self.terms: list = []          # (operator, BaseFunc, weight)
        self.normalisation_constant: float = 1.0

    def add(self, op: str, func: BaseFunc, weight: float = 1.0):
        self.terms.append((op, func, weight))

    def normalise(self, area_under_curve: float):
        """Divide every term weight by the integral so the function integrates
        to 1 over [0, 1]."""
        self.normalisation_constant = area_under_curve
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
        return expr


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — generate_function
# ══════════════════════════════════════════════════════════════════════════════

def generate_function(
    max_components: int = 3,
    max_attempts: int = 100,
    allowed_factories: Optional[list[str]] = None,
) -> Optional[dict]:
    """Algorithm 2: build and normalise a random symbolic f(x) over [0,1].

    Normalisation is performed here. Validation is NOT done here; call
    validate_function() on the returned dict separately.

    Returns
    -------
    dict:
        numpy_fn      – ComposedFunc (normalised callable)
        sympy_expr    – SymPy expression (simplified where possible)
        expr_str      – str of sympy_expr
        latex         – LaTeX string of sympy_expr
        n_components  – number of base functions combined (m)
        operators     – list of operators used between components
        norm_const    – ∫₀¹ f_raw dx (before normalisation)
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
            ops_used: list = []

            for _ in range(m - 1):
                op = random.choice(OPERATORS)
                w = sample_weight() if op == "+" else 1.0
                comp.add(
                    op,
                    random.choices(filt_factories, weights=filt_weights, k=1)[0](),
                    weight=w,
                )
                ops_used.append(op)

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

            # Reject if normalisation constant is extreme — this means the
            # raw function has very low mass (I << 1) or very high mass (I >> 1),
            # which after dividing by I produces coefficients like 0.000 or 70000.
            # Bound of 500 allows reasonable shapes while blocking pathological ones.
            if I > 500 or I < 0.001:
                continue

            norm_const = I
            comp.normalise(norm_const)

            # Build SymPy expression with 5-second simplification timeout
            raw_expr = comp.sympy_expr() / norm_const

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

            try:
                latex_str = sp.latex(sym_expr)
                expr_str = str(sym_expr)
            except Exception:
                sym_expr = None
                latex_str = "(complex expression)"
                expr_str = "(complex expression)"

            return {
                "numpy_fn": comp,
                "sympy_expr": sym_expr,
                "expr_str": expr_str,
                "latex": latex_str,
                "n_components": m,
                "operators": ops_used,
                "norm_const": norm_const,
            }

        except Exception:
            continue

    return None


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — validate_function
# ══════════════════════════════════════════════════════════════════════════════

def validate_function(result: dict, tol: float = 0.03) -> tuple:
    """Validate a generated function dict returned by generate_function().

    Checks performed
    ----------------
    1. numpy_fn is callable
    2. f(x) is finite for all x in [0, 1]
    3. f(x) >= 0 for all x in [0, 1]
    4. ∫₀¹ f(x) dx ∈ [1 - tol, 1 + tol]  (normalisation check)
    5. sympy_expr is not None

    Parameters
    ----------
    result : dict returned by generate_function()
    tol    : allowed deviation from 1.0 for the integral (default 0.03 = 3%)

    Returns
    -------
    (is_valid: bool, reason: str)
        reason is "OK" on success, or a short description of the failure.
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

    Parameters
    ----------
    numpy_fn  : normalised callable, ∫₀¹ f dx = 1
    N         : total histogram count (random in [n_min, n_max] if None)
    K         : number of bins        (random in [k_min, k_max] if None)
    n_min/max : range for random N
    k_min/max : range for random K

    Returns
    -------
    dict:
        bins   – ndarray (K,)   integer bin counts N_k
        edges  – ndarray (K+1,) bin edges in [0, 1]
        means  – ndarray (K,)   expected mean counts n_k = N * ∫_bin f dx
        N      – total count used
        K      – number of bins used
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

    # Sample bin counts from Poisson distribution (Poisson data per the paper)
    bins = np.random.poisson(means).astype(int)

    return {
        "bins": bins,
        "edges": edges,
        "means": means,
        "N": N,
        "K": K,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — infix_to_prefix  (preorder traversal)
# ══════════════════════════════════════════════════════════════════════════════

def _collect_prefix(expr, tokens: list, mantissas: list) -> None:
    """Recursive preorder walk of a SymPy expression tree."""

    # ── numbers / constants ──────────────────────────────────────────────────
    if expr.is_number or not expr.free_symbols:
        try:
            c = float(expr)
            tokens.append(_num_placeholder(c))   # raw float string, replaced later
            mantissas.append(c)                  # store raw value; encoding in step 6
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

    # ── named unary functions ─────────────────────────────────────────────────
    _FUNC_MAP = {
        sp.exp:  "exp",
        sp.log:  "log",
        sp.sin:  "sin",
        sp.cos:  "cos",
        sp.tan:  "tan",
        sp.sqrt: "sqrt",
        sp.Abs:  "abs",
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

    Constants are stored as raw values in the 'raw_mantissas' field.
    Use encode_constants() afterwards to convert to C{ce} scientific notation.

    Returns
    -------
    dict:
        tokens        – list[str]   preorder operator/variable tokens
        raw_mantissas – list[float] parallel raw constant values (0.0 for non-constants)
    """
    tokens: list = []
    mantissas: list = []
    _collect_prefix(sympy_expr, tokens, mantissas)
    return {"tokens": tokens, "raw_mantissas": mantissas}


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — prefix_to_infix
# ══════════════════════════════════════════════════════════════════════════════

# Arity table for known operator tokens
_ARITY = {
    "+":    2,
    "mul":  2,
    "pow":  2,
    "sqrt": 1,
    "exp":  1,
    "log":  1,
    "sin":  1,
    "cos":  1,
    "tan":  1,
    "abs":  1,
    "?":    0,
}


def prefix_to_infix(tokens: list, mantissas: Optional[list] = None) -> str:
    """Reconstruct a human-readable infix string from a preorder token list.

    Parameters
    ----------
    tokens    : list[str] as returned by infix_to_prefix() or encode_constants()
    mantissas : optional list[float] — if provided, numeric placeholders
                (NUM(...) or C{ce} tokens) are replaced with their float values.

    Returns
    -------
    str  infix expression
    """
    # If mantissas are provided, resolve numeric tokens to floats first
    resolved: list = []
    for i, tok in enumerate(tokens):
        if mantissas and (tok.startswith("C") or tok.startswith("NUM")):
            val = mantissas[i]
            resolved.append(f"{val:.6g}")
        else:
            resolved.append(tok)

    stack: list = []
    # Walk in reverse for a stack-based reconstruction
    for tok in reversed(resolved):
        arity = _ARITY.get(tok, None)

        if arity is None:
            # Leaf: variable, integer token, or resolved numeric
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
# Step 6 — encode_constants  (scientific-notation encoding per the paper)
# ══════════════════════════════════════════════════════════════════════════════

def encode_constant(c: float) -> tuple:
    """Encode a single float constant to (token, mantissa).

    Small integers in [-5, 5] → own vocab token e.g. '3', '-2'; mantissa = 0.0
    Zero                      → 'C0'; mantissa = 0.0
    Other values              → 'C{ce}' where ce = floor(log10|c|) + 1
                                mantissa cm = c / 10^ce, so |cm| ∈ [0.1, 1)

    Examples
    --------
    encode_constant(3)       → ('3',   0.0)
    encode_constant(0.017)   → ('C-1', 0.17)
    encode_constant(1781.5)  → ('C4',  0.17815)
    encode_constant(-0.05)   → ('C-1', -0.5)

    Returns
    -------
    (token: str, mantissa: float)
    """
    c_int = int(round(c))
    if c == float(c_int) and c_int in SMALL_INTS:
        return str(c_int), 0.0

    if c == 0.0:
        return "C0", 0.0

    abs_c = abs(c)
    ce = int(math.floor(math.log10(abs_c))) + 1
    cm = c / (10.0 ** ce)
    return f"C{ce}", round(cm, 8)


def encode_constants(tokens: list, raw_mantissas: list) -> dict:
    """Replace raw numeric placeholders in a prefix token list with C{ce} tokens.

    Parameters
    ----------
    tokens        : list[str] from infix_to_prefix()
    raw_mantissas : list[float] parallel raw values from infix_to_prefix()

    Returns
    -------
    dict:
        tokens    – list[str]   final token sequence (C{ce} / small-int / op / 'x')
        mantissas – list[float] parallel mantissa values (0.0 for non-constants)
    """
    out_tokens: list = []
    out_mantissas: list = []

    for tok, raw in zip(tokens, raw_mantissas):
        if tok.startswith("NUM(") or (tok not in _ARITY and tok != "x" and tok not in [str(i) for i in range(-5, 6)]):
            # Raw numeric placeholder — encode it
            tok_enc, cm = encode_constant(raw)
            out_tokens.append(tok_enc)
            out_mantissas.append(cm)
        elif tok in [str(i) for i in range(-5, 6)]:
            # Small int token — already final
            out_tokens.append(tok)
            out_mantissas.append(0.0)
        else:
            # Operator or variable
            out_tokens.append(tok)
            out_mantissas.append(0.0)

    return {"tokens": out_tokens, "mantissas": out_mantissas}


# ══════════════════════════════════════════════════════════════════════════════
# Step 7 — encode_expression  (end-to-end)
# ══════════════════════════════════════════════════════════════════════════════

def encode_expression(sympy_expr) -> dict:
    """Convert a SymPy expression to a preorder token sequence with encoded constants.

    Combines infix_to_prefix() + encode_constants() in one call.

    Returns
    -------
    dict:
        tokens    – list[str]   final token sequence
        mantissas – list[float] parallel mantissa values
    """
    prefix = infix_to_prefix(sympy_expr)
    return encode_constants(prefix["tokens"], prefix["raw_mantissas"])


# ══════════════════════════════════════════════════════════════════════════════
# Step 8 — generate_dataset  (batch, Algorithm 1 + 2)
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

    Parameters
    ----------
    n             : target number of valid records
    max_components: max base functions per sample  (paper uses 1–3)
    verbose       : print per-step timing for each sample
    seed          : random seed for reproducibility
    n_min/max     : range for histogram total count N
    k_min/max     : range for histogram bin count K

    Returns
    -------
    list of dicts, each containing:
        expr_str      – str representation of f(x)
        latex         – LaTeX string of f(x)
        n_components  – number of base functions combined
        operators     – list of operators used
        norm_const    – ∫₀¹ f_raw dx before normalisation
        histogram     – dict {bins, edges, means, N, K}
        encoding      – dict {tokens, mantissas}
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

        # ── Step C: histogram ────────────────────────────────────────────────
        t2 = time.perf_counter()
        histogram = generate_histogram(
            result["numpy_fn"],
            n_min=n_min, n_max=n_max,
            k_min=k_min, k_max=k_max,
        )
        t_hist = time.perf_counter() - t2

        # ── Step D: encoding ─────────────────────────────────────────────────
        t3 = time.perf_counter()
        encoding = (
            encode_expression(result["sympy_expr"])
            if result["sympy_expr"] is not None
            else {"tokens": [], "mantissas": []}
        )
        t_enc = time.perf_counter() - t3

        # ── Assemble record ───────────────────────────────────────────────────
        record = {
            "expr_str":     result["expr_str"],
            "latex":        result["latex"],
            "n_components": result["n_components"],
            "operators":    result["operators"],
            "norm_const":   result["norm_const"],
            "histogram":    histogram,
            "encoding":     encoding,
        }
        dataset.append(record)

        if verbose:
            idx = len(dataset)
            print(
                f"  [{idx:>4}/{n}]  attempt {attempts:>5} | "
                f"generate: {t_gen*1e3:.1f} ms | "
                f"validate: {t_val*1e3:.1f} ms | "
                f"histogram: {t_hist*1e3:.1f} ms | "
                f"encoding: {t_enc*1e3:.1f} ms"
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
# Gallery display (unchanged logic, updated for new pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def _format_expr(expr_str: str) -> str:
    """Round all floats in an expression string to 3 d.p."""
    import re
    result = re.sub(r"\bpi\b", "3.142", expr_str)
    result = re.sub(r"\bE\b", "2.718", result)
    result = re.sub(
        r"-?\d+\.\d+(?:[eE][+-]?\d+)?",
        lambda m: f"{float(m.group()):.3f}",
        result,
    )
    return result


def display_gallery(max_components: int = 3) -> None:
    """Generate 10 normalised functions and display them in a 2×5 grid."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.widgets import Button

    N_COLS, N_ROWS = 5, 2
    N_SAMPLES = N_COLS * N_ROWS

    def generate_batch():
        batch = []
        while len(batch) < N_SAMPLES:
            r = generate_function(max_components=max_components)
            if r is not None:
                ok, _ = validate_function(r)
                if ok:
                    batch.append(r)
        return batch

    def draw_batch(batch):
        print()
        for i, (ax, s) in enumerate(zip(axes.flat, batch), start=1):
            xs = np.linspace(1e-6, 1 - 1e-6, 600)
            ys = s["numpy_fn"](xs)
            ax.clear()
            ax.plot(xs, ys, lw=1.8, color="steelblue", zorder=3)
            ax.fill_between(xs, ys, alpha=0.18, color="steelblue", zorder=2)
            ax.set_xlim(0, 1)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("x", fontsize=8)
            ax.set_ylabel("f(x)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
            ax.grid(True, linestyle="--", alpha=0.4, zorder=1)
            ax.spines[["top", "right"]].set_visible(False)
            ops_str = ", ".join(s["operators"]) if s["operators"] else "—"
            ax.set_title(
                f"#{i}  m={s['n_components']}  ops=[{ops_str}]  C={s['norm_const']:.4f}",
                fontsize=7, pad=4,
            )
            print(f"  {i:>2}. f(x) = {_format_expr(s['expr_str'])}")
        fig.canvas.draw_idle()

    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(20, 7))
    fig.suptitle("FASEROH — Generated Normalised Functions", fontsize=13, y=0.99)
    plt.subplots_adjust(top=0.90, bottom=0.08, hspace=0.55, wspace=0.35)
    draw_batch(generate_batch())

    bax = fig.add_axes([0.40, 0.02, 0.20, 0.055])
    btn = Button(bax, "Refresh — Generate 10 New", color="#e3f2fd", hovercolor="#90caf9")
    btn.label.set_fontsize(10)
    btn.on_clicked(lambda _: draw_batch(generate_batch()))
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── POC demo: generate 10 samples using only POC-allowed factories ────
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
    print("\n--- Expressions and token sequences ---")
    for i, rec in enumerate(samples, 1):
        toks = rec["encoding"]["tokens"]
        mant = rec["encoding"]["mantissas"]
        print(f"  {i:>2}. expr = {rec['expr_str']}")
        print(f"      tokens    = {toks}")
        print(f"      mantissas = {[round(m, 4) for m in mant]}\n")