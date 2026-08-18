"""Semantic tensor expression tree and monoterm canonicalization.

The core layer is intentionally independent of SymPy.  It performs only tensor
structural normalization: slot symmetries, dummy-index alpha-renaming,
commutative-factor ordering, Kronecker-delta contraction, and metric/inverse
metric contractions.  Scalar algebra is deliberately limited to rational
coefficients so this layer remains import-light and predictable.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

from .indices import AbstractIndex
from .manifolds import TensorKernelError
from .tensor_heads import TensorHead

IndexKey = tuple[str, object, str]
FamilyKey = tuple[str, object]
FreeIndexSignature = tuple[tuple[str, str, object], ...]


@dataclass(frozen=True, slots=True)
class TensorFactor:
    """One indexed tensor-head occurrence."""

    head: TensorHead
    indices: tuple[AbstractIndex, ...]

    def __post_init__(self) -> None:
        if len(self.indices) != self.head.rank:
            raise TensorKernelError(
                f"Tensor factor {self.head.name!r} expects {self.head.rank} indices, got {len(self.indices)}."
            )
        for slot, (idx, expected) in enumerate(zip(self.indices, self.head.index_types)):
            if idx.index_type != expected:
                raise TensorKernelError(
                    f"Index {idx.name!r} in slot {slot} has type {idx.index_type.name!r}; expected {expected.name!r}."
                )
        self.head._validate_role_variance(self.indices)

    def canonicalized(self) -> tuple[int, "TensorFactor"]:
        sign, new_indices = self.head.symmetry.canonicalize_indices(self.indices)
        return sign, TensorFactor(self.head, tuple(new_indices))

    def key(self):
        return (
            0 if self.head.commutative else 1,
            self.head.role,
            self.head.name,
            tuple((idx.name, idx.index_type.name, id(idx.index_type), idx.variance) for idx in self.indices),
        )

    def with_indices(self, indices: Iterable[AbstractIndex]) -> "TensorFactor":
        return TensorFactor(self.head, tuple(indices))

    def __repr__(self) -> str:
        body = ",".join(("^" if idx.is_up else "_") + idx.name for idx in self.indices)
        return f"{self.head.name}({body})"


@dataclass(frozen=True, slots=True)
class TensorTerm:
    """A product of tensor factors multiplied by a rational coefficient."""

    coefficient: Fraction
    factors: tuple[TensorFactor, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "coefficient", Fraction(self.coefficient))

    @classmethod
    def one(cls) -> "TensorTerm":
        return cls(Fraction(1), ())

    @classmethod
    def zero(cls) -> "TensorTerm":
        return cls(Fraction(0), ())

    def canonicalized(self) -> "TensorTerm":
        if self.coefficient == 0:
            return TensorTerm.zero()
        coeff = self.coefficient
        factors: list[TensorFactor] = []
        for factor in self.factors:
            sign, canonical = factor.canonicalized()
            if sign == 0:
                return TensorTerm.zero()
            coeff *= sign
            factors.append(canonical)

        factors, coeff = _contract_structural_factors(factors, coeff)
        factors = _sort_factors(factors)
        term = TensorTerm(coeff, tuple(factors)).rename_dummies()
        term.validate_indices()
        return term

    def rename_dummies(self) -> "TensorTerm":
        """Rename dummy pairs deterministically after factor ordering.

        Free indices keep their user-facing names.  Each contracted up/down pair
        of the same index type receives a stable name ``d1``, ``d2``, ... per
        index type according to first occurrence in the canonical factor order.
        """
        occurrences = _index_occurrences(self.factors)
        dummy_keys: set[FamilyKey] = set()
        for key, items in occurrences.items():
            if len(items) == 2 and {item.variance for item in items} == {"up", "down"}:
                dummy_keys.add(key)
        if not dummy_keys:
            return self

        counters: dict[object, int] = defaultdict(int)
        mapping: dict[FamilyKey, str] = {}
        for factor in self.factors:
            for idx in factor.indices:
                key = (idx.name, idx.index_type)
                if key not in dummy_keys or key in mapping:
                    continue
                counters[idx.index_type] += 1
                mapping[key] = f"d{counters[idx.index_type]}"

        new_factors: list[TensorFactor] = []
        for factor in self.factors:
            new_indices = []
            for idx in factor.indices:
                name = mapping.get((idx.name, idx.index_type), idx.name)
                new_indices.append(AbstractIndex(name, idx.index_type, idx.variance))
            new_factors.append(TensorFactor(factor.head, tuple(new_indices)))
        return TensorTerm(self.coefficient, tuple(new_factors))

    def free_indices(self) -> tuple[AbstractIndex, ...]:
        occurrences = _index_occurrences(self.factors)
        free: list[AbstractIndex] = []
        for (name, _itype), items in occurrences.items():
            if len(items) == 1:
                free.append(items[0])
            elif len(items) == 2 and {item.variance for item in items} == {"up", "down"}:
                continue
            else:
                raise TensorKernelError(f"Invalid repeated index {name!r} in tensor term.")
        return tuple(sorted(free, key=lambda idx: (idx.name, idx.index_type.name, id(idx.index_type), idx.variance)))

    def free_index_signature(self) -> FreeIndexSignature:
        return tuple((idx.name, idx.variance, idx.index_type) for idx in self.free_indices())

    def validate_indices(self) -> None:
        self.free_index_signature()

    def key(self):
        return tuple(factor.key() for factor in self.factors)

    def __repr__(self) -> str:
        if self.coefficient == 0:
            return "0"
        if not self.factors:
            return str(self.coefficient)
        prefix = "" if self.coefficient == 1 else f"{self.coefficient}*"
        return prefix + "*".join(map(repr, self.factors))


@dataclass(frozen=True, slots=True)
class TensorExpr:
    """A finite sum of semantic tensor terms."""

    terms: tuple[TensorTerm, ...]

    @classmethod
    def zero(cls) -> "TensorExpr":
        return cls(())

    @classmethod
    def from_factor(cls, factor: TensorFactor) -> "TensorExpr":
        return cls((TensorTerm(Fraction(1), (factor,)),)).canonicalized()

    @classmethod
    def scalar(cls, value: int | Fraction) -> "TensorExpr":
        coeff = Fraction(value)
        return cls(()) if coeff == 0 else cls((TensorTerm(coeff, ()),))

    @property
    def is_zero(self) -> bool:
        return not self.terms

    @property
    def free_index_signature(self) -> FreeIndexSignature:
        if not self.terms:
            return ()
        signatures = {term.free_index_signature() for term in self.terms}
        if len(signatures) > 1:
            raise TensorKernelError("Expression contains incompatible free-index signatures.")
        return next(iter(signatures))

    def canonicalized(self) -> "TensorExpr":
        buckets: dict[tuple, Fraction] = {}
        representatives: dict[tuple, tuple[TensorFactor, ...]] = {}
        free_signature: FreeIndexSignature | None = None
        for term in self.terms:
            cterm = term.canonicalized()
            if cterm.coefficient == 0:
                continue
            current_free = cterm.free_index_signature()
            if free_signature is None:
                free_signature = current_free
            elif current_free != free_signature:
                raise TensorKernelError("Cannot add tensor terms with different free-index signatures.")
            key = cterm.key()
            buckets[key] = buckets.get(key, Fraction(0)) + cterm.coefficient
            representatives[key] = cterm.factors
        new_terms = [TensorTerm(coeff, representatives[key]) for key, coeff in buckets.items() if coeff]
        new_terms.sort(key=lambda term: term.key())
        return TensorExpr(tuple(new_terms))

    def validate_indices(self) -> None:
        _ = self.free_index_signature

    def __add__(self, other: "TensorExpr") -> "TensorExpr":
        return TensorExpr(self.terms + _coerce_expr(other).terms).canonicalized()

    def __radd__(self, other: "TensorExpr") -> "TensorExpr":
        return _coerce_expr(other) + self

    def __sub__(self, other: "TensorExpr") -> "TensorExpr":
        return self + (-_coerce_expr(other))

    def __rsub__(self, other: "TensorExpr") -> "TensorExpr":
        return _coerce_expr(other) + (-self)

    def __neg__(self) -> "TensorExpr":
        return TensorExpr(tuple(TensorTerm(-term.coefficient, term.factors) for term in self.terms)).canonicalized()

    def __mul__(self, other: "TensorExpr") -> "TensorExpr":
        other_expr = _coerce_expr(other)
        if not self.terms or not other_expr.terms:
            return TensorExpr.zero()
        terms: list[TensorTerm] = []
        for left in self.terms:
            for right in other_expr.terms:
                terms.append(TensorTerm(left.coefficient * right.coefficient, left.factors + right.factors))
        return TensorExpr(tuple(terms)).canonicalized()

    def __rmul__(self, other):
        if isinstance(other, (int, Fraction)):
            return TensorExpr(
                tuple(TensorTerm(Fraction(other) * term.coefficient, term.factors) for term in self.terms)
            ).canonicalized()
        return NotImplemented

    def __repr__(self) -> str:
        if not self.terms:
            return "0"
        chunks: list[str] = []
        for term in self.terms:
            text = repr(term)
            if chunks and not text.startswith("-"):
                chunks.append("+ " + text)
            elif chunks:
                chunks.append("- " + text[1:])
            else:
                chunks.append(text)
        return " ".join(chunks)


def _idx_key(idx: AbstractIndex) -> IndexKey:
    return (idx.name, idx.index_type, idx.variance)


def _same_family(left: AbstractIndex, right: AbstractIndex) -> bool:
    return left.name == right.name and left.index_type == right.index_type


def _index_occurrences(factors: tuple[TensorFactor, ...] | list[TensorFactor]) -> dict[FamilyKey, list[AbstractIndex]]:
    seen: dict[FamilyKey, list[AbstractIndex]] = defaultdict(list)
    for factor in factors:
        for idx in factor.indices:
            seen[(idx.name, idx.index_type)].append(idx)
    return seen


def _replace_index_in_factor(factor: TensorFactor, target: AbstractIndex, replacement: AbstractIndex) -> TensorFactor:
    new_indices = tuple(replacement if _idx_key(idx) == _idx_key(target) else idx for idx in factor.indices)
    try:
        return factor.with_indices(new_indices)
    except TensorKernelError:
        # Raising/lowering an ordinary tensor changes the variance of the
        # affected slot. Semantic roles with fixed variance should still fail
        # normally; only plain tensor heads receive an adjusted declaration.
        if factor.head.role != "tensor":
            raise
        adjusted_head = TensorHead(
            factor.head.name,
            factor.head.index_types,
            symmetry=factor.head.symmetry.kind,
            variance=tuple(idx.variance for idx in new_indices),
            commutative=factor.head.commutative,
            role=factor.head.role,
        )
        return TensorFactor(adjusted_head, new_indices)


def _replace_first_matching_index(
    factors: list[TensorFactor],
    skip_pos: int,
    target: AbstractIndex,
    replacement: AbstractIndex,
    *,
    replace_roles: set[str] | None = None,
) -> bool:
    for pos, factor in enumerate(factors):
        if pos == skip_pos:
            continue
        if replace_roles is not None and factor.head.role not in replace_roles:
            continue
        for idx in factor.indices:
            if _idx_key(idx) == _idx_key(target):
                factors[pos] = _replace_index_in_factor(factor, target, replacement)
                return True
    return False


def _remove_positions(factors: list[TensorFactor], positions: Iterable[int]) -> None:
    for pos in sorted(set(positions), reverse=True):
        del factors[pos]


def _delta_trace_once(factors: list[TensorFactor], coeff: Fraction) -> tuple[bool, Fraction]:
    for pos, factor in enumerate(factors):
        if factor.head.role != "delta":
            continue
        first, second = factor.indices
        if _same_family(first, second) and first.variance != second.variance:
            dimension = first.index_type.dimension
            if not isinstance(dimension, int):
                continue
            del factors[pos]
            return True, coeff * dimension
    return False, coeff


def _contract_delta_once(factors: list[TensorFactor]) -> bool:
    for pos, factor in enumerate(factors):
        if factor.head.role != "delta":
            continue
        first, second = factor.indices
        for contracted, free in ((first, second), (second, first)):
            target = contracted.flipped()
            if _replace_first_matching_index(factors, pos, target, free):
                del factors[pos]
                return True
    return False


def _contract_metric_inverse_once(factors: list[TensorFactor]) -> bool:
    for left_pos, left in enumerate(factors):
        if left.head.role not in {"metric", "inverse_metric"}:
            continue
        opposite_role = "inverse_metric" if left.head.role == "metric" else "metric"
        for right_pos, right in enumerate(factors):
            if right_pos == left_pos or right.head.role != opposite_role:
                continue
            for left_slot, left_idx in enumerate(left.indices):
                for right_slot, right_idx in enumerate(right.indices):
                    if not _same_family(left_idx, right_idx):
                        continue
                    left_remaining = left.indices[1 - left_slot]
                    right_remaining = right.indices[1 - right_slot]
                    delta_type = left_idx.index_type
                    delta = TensorHead.delta(f"delta_{delta_type.name}", delta_type)
                    if right_remaining.is_up and left_remaining.is_down:
                        new_factor = TensorFactor(delta, (right_remaining, left_remaining))
                    elif left_remaining.is_up and right_remaining.is_down:
                        new_factor = TensorFactor(delta, (left_remaining, right_remaining))
                    else:
                        continue
                    _remove_positions(factors, (left_pos, right_pos))
                    factors.append(new_factor)
                    return True
    return False


def _contract_metric_with_tensor_once(factors: list[TensorFactor]) -> bool:
    for pos, factor in enumerate(factors):
        if factor.head.role not in {"metric", "inverse_metric"}:
            continue
        replacement_roles = {"tensor", "delta"}
        for contracted, free in ((factor.indices[0], factor.indices[1]), (factor.indices[1], factor.indices[0])):
            target = contracted.flipped()
            if _replace_first_matching_index(factors, pos, target, free, replace_roles=replacement_roles):
                del factors[pos]
                return True
    return False


def _contract_structural_factors(factors: list[TensorFactor], coeff: Fraction) -> tuple[list[TensorFactor], Fraction]:
    factors = list(factors)
    for _step in range(128):
        changed, coeff = _delta_trace_once(factors, coeff)
        if changed:
            continue
        if _contract_delta_once(factors):
            continue
        if _contract_metric_inverse_once(factors):
            continue
        if _contract_metric_with_tensor_once(factors):
            continue
        break
    else:
        raise TensorKernelError("Structural tensor contraction exceeded its operation budget.")
    return factors, coeff


def _sort_factors(factors: list[TensorFactor]) -> list[TensorFactor]:
    if all(factor.head.commutative for factor in factors):
        return sorted(factors, key=lambda item: item.key())
    result: list[TensorFactor] = []
    run: list[TensorFactor] = []
    for factor in factors:
        if factor.head.commutative:
            run.append(factor)
            continue
        result.extend(sorted(run, key=lambda item: item.key()))
        run = []
        result.append(factor)
    result.extend(sorted(run, key=lambda item: item.key()))
    return result


def _coerce_expr(value) -> TensorExpr:
    if isinstance(value, TensorExpr):
        return value
    if isinstance(value, (int, Fraction)):
        return TensorExpr.scalar(value)
    raise TensorKernelError(f"Expected TensorExpr, got {type(value)!r}.")


def canonicalize(expr: TensorExpr) -> TensorExpr:
    return _coerce_expr(expr).canonicalized()
