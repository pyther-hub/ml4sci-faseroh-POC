"""Token vocabulary and encoding/decoding utilities for the FASEROH POC.

The vocabulary is derived from the POC function space — not hardcoded
arbitrarily.  It covers special tokens, operators, the variable x,
small integers, and the C{ce} scientific-notation exponent tokens.
"""

from __future__ import annotations

# ── POC vocabulary definition ─────────────────────────────────────────────────

_SPECIAL = ["<pad>", "<sos>", "<eos>", "<unk>"]
_OPERATORS = ["+", "mul", "pow", "sqrt", "log", "sin", "cos"]
_VARIABLE = ["x"]
_MATH_CONSTANTS = ["pi", "exp"]  # π and Euler's number as single tokens
_SMALL_INTS = [str(i) for i in range(-5, 6)]
_CONST_TOKENS = [f"C{ce}" for ce in range(-4, 5)]

_ALL_TOKENS: list[str] = _SPECIAL + _OPERATORS + _VARIABLE + _MATH_CONSTANTS + _SMALL_INTS + _CONST_TOKENS

_TOKEN2ID: dict[str, int] = {tok: idx for idx, tok in enumerate(_ALL_TOKENS)}
_ID2TOKEN: dict[int, str] = {idx: tok for tok, idx in _TOKEN2ID.items()}

_OPERATOR_SET = set(_OPERATORS)
_CONST_SET = set(_CONST_TOKENS)
_MATH_CONST_SET = set(_MATH_CONSTANTS)
_LEAF_SET = set(_VARIABLE) | _MATH_CONST_SET | set(_SMALL_INTS) | _CONST_SET


# ── Public API ────────────────────────────────────────────────────────────────

def build_vocabulary() -> dict[str, int]:
    """Return TOKEN2ID dict for the full POC vocabulary.

    Returns
    -------
    dict[str, int]
        Mapping from token string to integer id.
    """
    return dict(_TOKEN2ID)


def get_vocab_size() -> int:
    """Return the total number of tokens in the vocabulary.

    Returns
    -------
    int
    """
    return len(_TOKEN2ID)


def token_to_id(token: str) -> int:
    """Return the integer id for *token*, or the <unk> id if not found.

    Parameters
    ----------
    token : str

    Returns
    -------
    int
    """
    return _TOKEN2ID.get(token, _TOKEN2ID["<unk>"])


def id_to_token(idx: int) -> str:
    """Return the token string for integer id *idx*.

    Parameters
    ----------
    idx : int

    Returns
    -------
    str
    """
    return _ID2TOKEN.get(idx, "<unk>")


def tokens_to_ids(tokens: list[str]) -> list[int]:
    """Convert a list of token strings to integer ids.

    Parameters
    ----------
    tokens : list[str]

    Returns
    -------
    list[int]
    """
    return [token_to_id(t) for t in tokens]


def ids_to_tokens(ids: list[int]) -> list[str]:
    """Convert a list of integer ids to token strings.

    Parameters
    ----------
    ids : list[int]

    Returns
    -------
    list[str]
    """
    return [id_to_token(i) for i in ids]


def is_constant_token(token: str) -> bool:
    """Return True if *token* is a C{ce} constant-exponent token.

    Parameters
    ----------
    token : str

    Returns
    -------
    bool
    """
    return token in _CONST_SET


def is_operator_token(token: str) -> bool:
    """Return True if *token* is an operator.

    Parameters
    ----------
    token : str

    Returns
    -------
    bool
    """
    return token in _OPERATOR_SET


def is_leaf_token(token: str) -> bool:
    """Return True if *token* is a leaf (x, small int, or C{ce}).

    Parameters
    ----------
    token : str

    Returns
    -------
    bool
    """
    return token in _LEAF_SET
