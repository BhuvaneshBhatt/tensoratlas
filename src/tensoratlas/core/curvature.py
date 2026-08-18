"""Curvature-head helpers and algebraic curvature identities."""
from __future__ import annotations

from .indices import AbstractIndex, IndexType
from .manifolds import TensorKernelError
from .tensor_expr import TensorExpr, canonicalize
from .tensor_heads import TensorHead


def curvature_heads(index_type: IndexType, *, prefix: str = "") -> dict[str, TensorHead]:
    """Return standard curvature-related tensor heads for one index family."""
    p = prefix
    return {
        "riemann_mixed": TensorHead.curvature(f"{p}R" if p else "R", index_type),
        "riemann_lower": TensorHead.riemann(f"{p}Riem" if p else "Riem", index_type, variance=("down",) * 4),
        "weyl_lower": TensorHead.weyl(f"{p}C" if p else "C", index_type, variance=("down",) * 4),
        "ricci": TensorHead.ricci(f"{p}Ric" if p else "Ric", index_type),
        "einstein": TensorHead.einstein(f"{p}G" if p else "G", index_type),
        "scalar": TensorHead.scalar_curvature(f"{p}Scal" if p else "Scal"),
    }


def first_bianchi_identity(
    curvature: TensorHead,
    raised_index: AbstractIndex,
    first_lower: AbstractIndex,
    second_lower: AbstractIndex,
    third_lower: AbstractIndex,
) -> TensorExpr:
    """Build the torsion-free algebraic Bianchi cyclic sum for mixed curvature.

    The expression is returned in canonical TensorAtlas core form.  A later
    multiterm reduction engine can reduce arbitrary expressions modulo this
    identity; this helper gives a precise, testable identity object.
    """
    _validate_mixed_curvature(curvature)
    for idx in (first_lower, second_lower, third_lower):
        if not idx.is_down:
            raise TensorKernelError("Bianchi lower indices must be covariant/down.")
    if not raised_index.is_up:
        raise TensorKernelError("Bianchi raised index must be contravariant/up.")
    return canonicalize(
        curvature(raised_index, first_lower, second_lower, third_lower)
        + curvature(raised_index, second_lower, third_lower, first_lower)
        + curvature(raised_index, third_lower, first_lower, second_lower)
    )


def is_first_bianchi_sum(expr: TensorExpr, curvature: TensorHead | None = None) -> bool:
    """Recognize the exact three-term mixed-curvature cyclic sum.

    This is intentionally narrow.  General linear reduction modulo Bianchi
    identities belongs to the general multiterm reduction engine.
    """
    expr = canonicalize(expr)
    if len(expr.terms) != 3:
        return False
    factors = []
    for term in expr.terms:
        if len(term.factors) != 1:
            return False
        factor = term.factors[0]
        if curvature is not None and factor.head != curvature:
            return False
        try:
            _validate_mixed_curvature(factor.head)
        except TensorKernelError:
            return False
        factors.append(factor)
    head = factors[0].head
    if any(factor.head != head for factor in factors):
        return False
    raised = factors[0].indices[0]
    if any(factor.indices[0] != raised for factor in factors):
        return False
    lowers = [factor.indices[1:] for factor in factors]
    pool = sorted({repr(idx) for triple in lowers for idx in triple})
    if len(pool) != 3:
        return False
    # Compare against the canonical identity generated from one representative
    # and allow an overall nonzero scalar multiple.
    i, j, k = lowers[0]
    candidate = first_bianchi_identity(head, raised, i, j, k)
    return _proportional(expr, candidate)


def apply_first_bianchi(expr: TensorExpr, curvature: TensorHead | None = None) -> TensorExpr:
    """Reduce an exact recognized first-Bianchi sum to zero."""
    return TensorExpr.zero() if is_first_bianchi_sum(expr, curvature) else canonicalize(expr)


def _validate_mixed_curvature(head: TensorHead) -> None:
    if head.rank != 4 or head.variance != ("up", "down", "down", "down"):
        raise TensorKernelError("Expected mixed curvature head with variance (^a, _b, _c, _d).")


def _proportional(left: TensorExpr, right: TensorExpr) -> bool:
    if left.is_zero or right.is_zero or len(left.terms) != len(right.terms):
        return False
    left_terms = {term.key(): term.coefficient for term in left.terms}
    right_terms = {term.key(): term.coefficient for term in right.terms}
    if set(left_terms) != set(right_terms):
        return False
    ratio = None
    for key in left_terms:
        if right_terms[key] == 0:
            return False
        current = left_terms[key] / right_terms[key]
        ratio = current if ratio is None else ratio
        if current != ratio:
            return False
    return ratio != 0
