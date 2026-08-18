"""Bounded symbolic simplification policies.

This module deliberately contains no import-time modification of SymPy.  Code that
needs stronger or domain-aware simplification must call these helpers
explicitly, keeping ``import tensoratlas`` side-effect light.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import sympy as sp

from .simplification_core import light_simplify, canonical_simplify, sympify_expr


@dataclass(frozen=True)
class CoordinateDomainPolicy:
    """Minimal sign information derived from a chart's coordinate domains."""

    positive: frozenset[sp.Expr] = frozenset()
    nonnegative: frozenset[sp.Expr] = frozenset()
    negative: frozenset[sp.Expr] = frozenset()
    nonpositive: frozenset[sp.Expr] = frozenset()

    def abs_replacement(self, expr: sp.Expr) -> sp.Expr | None:
        if expr in self.positive or expr in self.nonnegative:
            return expr
        if expr in self.negative or expr in self.nonpositive:
            return -expr
        if getattr(expr, "is_positive", None) is True or getattr(expr, "is_nonnegative", None) is True:
            return expr
        if getattr(expr, "is_negative", None) is True or getattr(expr, "is_nonpositive", None) is True:
            return -expr
        return None


def domain_policy_from_specs(
    coordinate_names: Sequence[str],
    coords: Sequence[sp.Expr],
    domains: Mapping[str, Mapping[str, Any]],
) -> CoordinateDomainPolicy:
    positive: set[sp.Expr] = set()
    nonnegative: set[sp.Expr] = set()
    negative: set[sp.Expr] = set()
    nonpositive: set[sp.Expr] = set()
    for name, coord in zip(coordinate_names, coords):
        spec = domains.get(name, {})
        kind = spec.get("kind")
        lower = spec.get("min")
        upper = spec.get("max")
        if kind == "half_line" and lower == 0:
            nonnegative.add(coord)
        elif kind == "open_interval" and lower == 0:
            positive.add(coord)
        elif kind == "closed_interval" and lower == 0:
            nonnegative.add(coord)
        elif kind == "open_interval" and upper == 0:
            negative.add(coord)
        elif kind == "closed_interval" and upper == 0:
            nonpositive.add(coord)
    return CoordinateDomainPolicy(
        positive=frozenset(positive),
        nonnegative=frozenset(nonnegative),
        negative=frozenset(negative),
        nonpositive=frozenset(nonpositive),
    )


def replace_abs_with_domain_policy(expr: Any, policy: CoordinateDomainPolicy) -> sp.Expr:
    out = sympify_expr(expr)
    for abs_term in list(out.atoms(sp.Abs)):
        if len(abs_term.args) != 1:
            continue
        replacement = policy.abs_replacement(abs_term.args[0])
        if replacement is not None:
            out = out.xreplace({abs_term: replacement})
    return out


def bounded_algebraic_simplify_expr(expr: Any) -> sp.Expr:
    """Cheap rational/algebraic cleanup only; never calls ``sympy.simplify``."""
    return light_simplify(expr)


def coordinate_simplify_expr(expr: Any, policy: CoordinateDomainPolicy | None = None) -> sp.Expr:
    """Cleanup suitable for coordinate formulas.

    The pass is intentionally bounded: cancellation/together/factor_terms plus
    sign cleanup for explicit ``Abs`` nodes.  This avoids the nontermination
    risk of repeatedly applying broad simplification to metric expressions.
    """
    out = bounded_algebraic_simplify_expr(expr)
    if policy is not None:
        out = replace_abs_with_domain_policy(out, policy)
        out = bounded_algebraic_simplify_expr(out)
    return out


def explicit_strong_simplify(expr: Any) -> sp.Expr:
    """Opt-in wrapper for callers that deliberately want SymPy's simplify."""
    return canonical_simplify(expr, final=True)
