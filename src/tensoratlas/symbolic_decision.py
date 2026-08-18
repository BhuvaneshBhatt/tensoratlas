from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any
import warnings

import sympy as sp

from .basic_algebraic_reduce import algebraically_equal_basic, basic_root_canonicalize, is_nested_algebraic_form
from .simplification_core import light_simplify, canonical_simplify, sympify_expr


class DecisionKind(str, Enum):
    ZERO = "zero"
    NONZERO = "nonzero"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class SymbolicDecision:
    kind: DecisionKind
    method: str
    used_numeric_fallback: bool = False
    cache_hit: bool = False


DEFAULT_COMPLEXITY_LIMIT = 250
WARNING_EXPR_MAXLEN = 120


class PossibleZeroQWarning(UserWarning):
    pass



def _short_expr_text(expr: Any, *, maxlen: int = WARNING_EXPR_MAXLEN) -> str:
    try:
        text = str(sympify_expr(expr))
    except Exception:
        text = str(expr)
    text = " ".join(text.split())
    if len(text) <= maxlen:
        return text
    return text[: maxlen - 3] + "..."



@lru_cache(maxsize=20000)
def _cached_zero_decision(expr: sp.Expr, mode: str) -> SymbolicDecision:
    if expr == 0:
        return SymbolicDecision(DecisionKind.ZERO, "literal")
    iz = getattr(expr, "is_zero", None)
    if iz is True:
        return SymbolicDecision(DecisionKind.ZERO, "is_zero")
    if iz is False:
        return SymbolicDecision(DecisionKind.NONZERO, "is_zero")
    if expr.is_number:
        try:
            return SymbolicDecision(DecisionKind.ZERO if expr.equals(0) else DecisionKind.NONZERO, "number-equals")
        except Exception:
            pass
    try:
        cleaned = light_simplify(expr)
        if cleaned == 0 or cleaned.is_zero is True:
            return SymbolicDecision(DecisionKind.ZERO, "light-simplify")
        if cleaned.is_zero is False:
            return SymbolicDecision(DecisionKind.NONZERO, "light-simplify")
    except Exception:
        cleaned = expr
    try:
        if is_nested_algebraic_form(cleaned):
            reduced = basic_root_canonicalize(cleaned)
            if reduced is not None:
                if reduced == 0 or getattr(reduced, "is_zero", None) is True:
                    return SymbolicDecision(DecisionKind.ZERO, "basic-root-canonicalize")
                if getattr(reduced, "is_zero", None) is False:
                    return SymbolicDecision(DecisionKind.NONZERO, "basic-root-canonicalize")
                if reduced != cleaned:
                    cleaned = reduced
    except Exception:
        pass
    try:
        if sp.count_ops(cleaned, visual=False) <= DEFAULT_COMPLEXITY_LIMIT:
            s = sp.simplify(cleaned)
            if s == 0 or getattr(s, "is_zero", None) is True:
                return SymbolicDecision(DecisionKind.ZERO, "simplify")
            if getattr(s, "is_zero", None) is False:
                return SymbolicDecision(DecisionKind.NONZERO, "simplify")
    except Exception:
        pass
    if mode != "strict":
        try:
            val = sp.N(cleaned, 50)
            if getattr(val, "is_number", False):
                if abs(complex(val)) < 1e-30:
                    return SymbolicDecision(DecisionKind.UNCERTAIN, "numeric-near-zero", True)
                if mode not in {"safe", "conservative"}:
                    return SymbolicDecision(DecisionKind.NONZERO, "numeric", True)
                return SymbolicDecision(DecisionKind.UNCERTAIN, "numeric-nonzero-conservative", True)
        except Exception:
            pass
    return SymbolicDecision(DecisionKind.UNCERTAIN, "fallback")


def zero_decision(expr: Any, *, mode: str = "safe") -> SymbolicDecision:
    decision = _cached_zero_decision(sympify_expr(expr), mode)
    return SymbolicDecision(decision.kind, decision.method, decision.used_numeric_fallback, True)


def explain_zero_decision(expr: Any, *, mode: str = "safe") -> dict[str, Any]:
    decision = zero_decision(expr, mode=mode)
    return {
        "expression": _short_expr_text(expr),
        "kind": decision.kind.value,
        "method": decision.method,
        "used_numeric_fallback": decision.used_numeric_fallback,
        "cache_hit": decision.cache_hit,
        "mode": mode,
    }


def is_zero(expr: Any, *, mode: str = "safe") -> bool:
    return zero_decision(expr, mode=mode).kind is DecisionKind.ZERO


def equal_decision(left: Any, right: Any, *, mode: str = "safe") -> SymbolicDecision:
    l = sympify_expr(left)
    r = sympify_expr(right)
    if isinstance(l, sp.logic.boolalg.Boolean) or isinstance(r, sp.logic.boolalg.Boolean):
        try:
            if l == r:
                return SymbolicDecision(DecisionKind.ZERO, "boolean-equality")
            return SymbolicDecision(DecisionKind.NONZERO, "boolean-equality")
        except Exception:
            return SymbolicDecision(DecisionKind.UNCERTAIN, "boolean-fallback")
    algebraic_eq = algebraically_equal_basic(l, r)
    if algebraic_eq is True:
        return SymbolicDecision(DecisionKind.ZERO, "equal:basic-algebraic")
    if algebraic_eq is False:
        return SymbolicDecision(DecisionKind.NONZERO, "equal:basic-algebraic")
    diff = l - r
    decision = zero_decision(diff, mode=mode)
    return SymbolicDecision(decision.kind, f"equal:{decision.method}", decision.used_numeric_fallback, decision.cache_hit)


def is_equal(left: Any, right: Any, *, mode: str = "safe") -> bool:
    return equal_decision(left, right, mode=mode).kind is DecisionKind.ZERO


def clear_symbolic_decision_cache() -> None:
    _cached_zero_decision.cache_clear()


def symbolic_decision_cache_info() -> dict[str, int]:
    info = _cached_zero_decision.cache_info()
    return {
        "hits": info.hits,
        "misses": info.misses,
        "maxsize": info.maxsize or 0,
        "currsize": info.currsize,
    }


def possibly_zero(expr: Any, *, emit_warning: bool = True, mode: str = "safe") -> bool:
    decision = zero_decision(expr, mode=mode)
    if decision.kind is DecisionKind.NONZERO:
        return False
    if decision.kind is DecisionKind.UNCERTAIN and emit_warning:
        warnings.warn(
            f"Could not decide whether or not {_short_expr_text(expr)} is zero.",
            PossibleZeroQWarning,
            stacklevel=2,
        )
    return True
