"""Permutation-group style monoterm canonicalization for semantic tensors.

This is a pure-Python, bounded Butler-Portugal/xPerm-inspired canonicalizer for
small-to-medium semantic expressions.  It enumerates declared slot-symmetry
images, canonicalizes commuting factor order, alpha-renames dummy indices, and
selects the lexicographically minimal representative.  Large production xPerm
engines use stabilizer-chain algorithms; this module provides the same semantic
contract for ordinary package-sized expressions while remaining simple and
backend-independent.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product, permutations

from .indices import AbstractIndex
from .manifolds import TensorKernelError
from .tensor_expr import TensorExpr, TensorFactor, TensorTerm


@dataclass(frozen=True, slots=True)
class CanonicalizationReport:
    """Result and diagnostics for strong monoterm canonicalization."""

    expression: TensorExpr
    candidates_considered: int
    budget_exhausted: bool = False


def canonicalize_monoterm(term: TensorTerm, *, budget: int = 20000) -> TensorTerm:
    """Return a strong canonical representative for one tensor monomial."""

    report = canonicalize_expression(TensorExpr((term,)), budget=budget)
    if not report.expression.terms:
        return TensorTerm.zero()
    if len(report.expression.terms) != 1:
        raise TensorKernelError("Internal monoterm canonicalization produced multiple terms.")
    return report.expression.terms[0]


def canonicalize_expression(expr: TensorExpr, *, budget: int = 20000) -> CanonicalizationReport:
    """Canonicalize all monomials using bounded permutation search."""

    if budget <= 0:
        raise TensorKernelError("Canonicalization budget must be positive.")
    candidates = 0
    exhausted = False
    buckets: dict[tuple, Fraction] = {}
    reps: dict[tuple, tuple[TensorFactor, ...]] = {}
    for term in expr.terms:
        best_term, used, term_exhausted = _best_term(term, budget=max(1, budget - candidates))
        candidates += used
        exhausted = exhausted or term_exhausted
        if best_term.coefficient == 0:
            continue
        best_term = best_term.canonicalized()
        key = _term_key(best_term)
        buckets[key] = buckets.get(key, Fraction(0)) + best_term.coefficient
        reps[key] = best_term.factors
        if candidates >= budget:
            exhausted = True
            break
    terms = [TensorTerm(coeff, reps[key]) for key, coeff in buckets.items() if coeff]
    terms.sort(key=_term_key)
    return CanonicalizationReport(TensorExpr(tuple(terms)).canonicalized(), candidates, exhausted)


def _best_term(term: TensorTerm, *, budget: int) -> tuple[TensorTerm, int, bool]:
    if term.coefficient == 0:
        return TensorTerm.zero(), 0, False
    factor_options = [_factor_images(factor) for factor in term.factors]
    best_key = None
    best_term = None
    seen = 0
    exhausted = False
    for choices in product(*factor_options):
        sign = 1
        factors = []
        for local_sign, factor in choices:
            sign *= local_sign
            factors.append(factor)
        if sign == 0:
            candidate = TensorTerm.zero()
        else:
            factors = _sort_commuting_runs(factors)
            candidate = TensorTerm(term.coefficient * sign, tuple(factors)).rename_dummies()
            try:
                candidate.validate_indices()
            except TensorKernelError:
                seen += 1
                continue
        key = _term_key(candidate)
        if best_key is None or key < best_key:
            best_key = key
            best_term = candidate
        seen += 1
        if seen >= budget:
            exhausted = True
            break
    if best_term is None:
        best_term = term.canonicalized()
    return best_term, seen, exhausted


def _factor_images(factor: TensorFactor) -> list[tuple[int, TensorFactor]]:
    kind = factor.head.symmetry.kind
    indices = factor.indices
    images: list[tuple[int, tuple[AbstractIndex, ...]]] = []
    if kind == "none" or len(indices) < 2:
        images = [(1, tuple(indices))]
    elif kind == "symmetric":
        images = [(1, tuple(p)) for p in sorted(set(permutations(indices)), key=lambda item: tuple(map(repr, item)))]
    elif kind == "antisymmetric":
        if len({(idx.name, idx.index_type, idx.variance) for idx in indices}) < len(indices):
            images = [(0, tuple(indices))]
        else:
            images = [(_permutation_parity(indices, p), tuple(p)) for p in permutations(indices)]
    elif kind == "antisym_last2":
        images = [(1, tuple(indices)), (-1, tuple(indices[:-2]) + (indices[-1], indices[-2]))]
    elif kind in {"riemann", "weyl"} and len(indices) == 4:
        a, b, c, d = indices
        raw = [
            (1, (a, b, c, d)),
            (-1, (b, a, c, d)),
            (-1, (a, b, d, c)),
            (1, (b, a, d, c)),
            (1, (c, d, a, b)),
            (-1, (d, c, a, b)),
            (-1, (c, d, b, a)),
            (1, (d, c, b, a)),
        ]
        images = raw
    else:
        images = [(1, tuple(indices))]
    out = []
    seen = set()
    for sign, idxs in images:
        key = (sign, tuple((idx.name, id(idx.index_type), idx.variance) for idx in idxs))
        if key in seen:
            continue
        seen.add(key)
        if sign == 0:
            out.append((0, factor))
        else:
            try:
                out.append((sign, TensorFactor(factor.head, tuple(idxs))))
            except TensorKernelError:
                continue
    return out or [(1, factor)]


def _permutation_parity(original: tuple[AbstractIndex, ...], candidate: tuple[AbstractIndex, ...]) -> int:
    positions = {id(value): pos for pos, value in enumerate(original)}
    order = [positions[id(value)] for value in candidate]
    inversions = 0
    for left in range(len(order)):
        for right in range(left + 1, len(order)):
            if order[left] > order[right]:
                inversions += 1
    return -1 if inversions % 2 else 1


def _sort_commuting_runs(factors: list[TensorFactor]) -> list[TensorFactor]:
    if all(factor.head.commutative for factor in factors):
        return sorted(factors, key=_factor_sort_key)
    result: list[TensorFactor] = []
    run: list[TensorFactor] = []
    for factor in factors:
        if factor.head.commutative:
            run.append(factor)
            continue
        result.extend(sorted(run, key=_factor_sort_key))
        run = []
        result.append(factor)
    result.extend(sorted(run, key=_factor_sort_key))
    return result


def _factor_sort_key(factor: TensorFactor):
    return (factor.head.name, factor.head.role, tuple(repr(idx) for idx in factor.indices))


def _term_key(term: TensorTerm):
    if term.coefficient == 0:
        return (0, ())
    return (1, tuple(_factor_sort_key(factor) for factor in term.factors), str(term.coefficient))
