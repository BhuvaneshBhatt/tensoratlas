"""Epsilon, generalized-delta, and determinant-style tensor identities."""
from __future__ import annotations

from itertools import permutations

from .indices import AbstractIndex, IndexType
from .manifolds import TensorKernelError
from .tensor_expr import TensorExpr, TensorFactor, TensorTerm, canonicalize
from .tensor_heads import TensorHead


def levi_civita_head(name: str, index_type: IndexType, *, variance: str = "down") -> TensorHead:
    """Return a fully antisymmetric Levi-Civita tensor/symbol head."""

    dim = index_type.dimension
    if not isinstance(dim, int):
        raise TensorKernelError("Levi-Civita rank requires an integer index-type dimension.")
    if variance not in {"up", "down"}:
        raise TensorKernelError("Levi-Civita variance must be 'up' or 'down'.")
    return TensorHead(name, (index_type,) * dim, symmetry="antisymmetric", variance=(variance,) * dim, role="epsilon")


def generalized_delta_head(name: str, index_type: IndexType, order: int) -> TensorHead:
    """Return generalized Kronecker delta ``delta^{a...}_{b...}``."""

    if order <= 0:
        raise TensorKernelError("Generalized-delta order must be positive.")
    return TensorHead(
        name,
        (index_type,) * (2 * order),
        symmetry="none",
        variance=("up",) * order + ("down",) * order,
        role="generalized_delta",
    )


def generalized_delta_expansion(head: TensorHead, indices: tuple[AbstractIndex, ...]) -> TensorExpr:
    """Expand a generalized delta into ordinary delta products."""

    if head.role != "generalized_delta" or len(indices) != head.rank or head.rank % 2:
        raise TensorKernelError("Expected a generalized-delta head with matching indices.")
    order = head.rank // 2
    uppers = indices[:order]
    lowers = indices[order:]
    delta = TensorHead.delta(f"delta_{head.index_types[0].name}", head.index_types[0])
    terms = []
    for perm in permutations(range(order)):
        sign = _permutation_sign(perm)
        expr = TensorExpr.scalar(sign)
        for up_pos, low_pos in enumerate(perm):
            expr = expr * delta(uppers[up_pos], lowers[low_pos])
        terms.extend(expr.terms)
    return TensorExpr(tuple(terms)).canonicalized()


def epsilon_product_to_generalized_delta(expr: TensorExpr, *, name: str = "Delta") -> TensorExpr:
    """Replace full contraction of one upper and one lower epsilon by generalized delta.

    For matching rank ``n`` this applies the structural identity
    ``epsilon^{a1...an} epsilon_{b1...bn} -> Delta^{a1...an}_{b1...bn}``.
    Signature-dependent global signs belong to the convention layer; this helper
    records the orientation-neutral tensor identity.
    """

    new_terms = []
    changed = False
    for term in expr.terms:
        eps_positions = [pos for pos, factor in enumerate(term.factors) if factor.head.role == "epsilon"]
        if len(eps_positions) < 2:
            new_terms.append(term)
            continue
        used = set()
        factors = list(term.factors)
        replacement_terms = [term]
        for left_pos in eps_positions:
            if left_pos in used:
                continue
            left = factors[left_pos]
            for right_pos in eps_positions:
                if right_pos <= left_pos or right_pos in used:
                    continue
                right = factors[right_pos]
                if left.head.rank != right.head.rank or left.head.index_types != right.head.index_types:
                    continue
                left_var = {idx.variance for idx in left.indices}
                right_var = {idx.variance for idx in right.indices}
                if left_var == right_var:
                    continue
                if left_var == {"up"} and right_var == {"down"}:
                    uppers, lowers = left.indices, right.indices
                elif left_var == {"down"} and right_var == {"up"}:
                    uppers, lowers = right.indices, left.indices
                else:
                    continue
                gd = generalized_delta_head(name, left.head.index_types[0], left.head.rank)
                gd_factor = TensorFactor(gd, tuple(uppers + lowers))
                remaining = [factor for pos, factor in enumerate(factors) if pos not in {left_pos, right_pos}]
                remaining.append(gd_factor)
                new_terms.append(TensorTerm(term.coefficient, tuple(remaining)).canonicalized())
                used.update({left_pos, right_pos})
                changed = True
                break
            if used:
                break
        if not used:
            new_terms.append(term)
    return canonicalize(TensorExpr(tuple(new_terms))) if changed else expr.canonicalized()


def _permutation_sign(perm: tuple[int, ...]) -> int:
    inversions = 0
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            if perm[i] > perm[j]:
                inversions += 1
    return -1 if inversions % 2 else 1
