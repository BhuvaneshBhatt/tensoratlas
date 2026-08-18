"""Sparse multiterm identity reduction for semantic tensor expressions."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations
from typing import Iterable, Mapping, Sequence

from .indices import AbstractIndex
from .manifolds import TensorKernelError
from .tensor_expr import TensorExpr, TensorTerm, canonicalize
from .tensor_heads import TensorHead

TermKey = tuple


def _permutation_sign(order: Sequence[int]) -> int:
    inversions = 0
    for left in range(len(order)):
        for right in range(left + 1, len(order)):
            if order[left] > order[right]:
                inversions += 1
    return -1 if inversions % 2 else 1


@dataclass(frozen=True, slots=True)
class LinearIdentity:
    """A homogeneous linear identity among canonical tensor monomials."""

    expression: TensorExpr
    name: str = "identity"
    preferred_pivot_key: TermKey | None = None

    def __post_init__(self) -> None:
        expr = canonicalize(self.expression)
        if expr.is_zero:
            raise TensorKernelError("Linear identities must contain at least one nonzero term.")
        keys = {term.key() for term in expr.terms}
        if self.preferred_pivot_key is not None and self.preferred_pivot_key not in keys:
            raise TensorKernelError("Preferred identity pivot is not present in the identity expression.")
        object.__setattr__(self, "expression", expr)

    @property
    def pivot_key(self) -> TermKey:
        if self.preferred_pivot_key is not None:
            return self.preferred_pivot_key
        return max(term.key() for term in self.expression.terms)

    @property
    def pivot_coefficient(self) -> Fraction:
        key = self.pivot_key
        for term in self.expression.terms:
            if term.key() == key:
                return term.coefficient
        raise TensorKernelError("Internal identity pivot lookup failed.")

    def replacement_terms(self) -> tuple[TensorTerm, ...]:
        """Return terms replacing the pivot monomial."""
        pivot = self.pivot_key
        coeff = self.pivot_coefficient
        return tuple(
            TensorTerm(-term.coefficient / coeff, term.factors)
            for term in self.expression.terms
            if term.key() != pivot
        )


@dataclass(frozen=True, slots=True)
class ReductionTraceStep:
    """One oriented identity application in a multiterm reduction."""

    identity_name: str
    pivot_key: TermKey
    coefficient: Fraction


@dataclass(frozen=True, slots=True)
class ReductionResult:
    """Reduced expression plus a compact proof trace."""

    expression: TensorExpr
    steps: tuple[ReductionTraceStep, ...]


@dataclass(frozen=True, slots=True, init=False)
class MultitermReductionSystem:
    """A terminating reducer based on oriented sparse linear identities."""

    identities: tuple[LinearIdentity, ...]
    max_steps: int

    def __init__(self, identities: Iterable[LinearIdentity], *, max_steps: int = 256):
        ordered = sorted(tuple(identities), key=lambda item: (item.pivot_key, item.name), reverse=True)
        object.__setattr__(self, "identities", ordered)
        object.__setattr__(self, "max_steps", max_steps)

    def reduce(self, expression: TensorExpr) -> TensorExpr:
        return self.reduce_with_trace(expression).expression

    def reduce_with_trace(self, expression: TensorExpr) -> ReductionResult:
        expr = canonicalize(expression)
        if expr.is_zero or not self.identities:
            return ReductionResult(expr, ())
        trace: list[ReductionTraceStep] = []
        for _ in range(self.max_steps):
            changed, expr, step = self._reduce_once(expr)
            if not changed:
                return ReductionResult(expr, tuple(trace))
            if step is not None:
                trace.append(step)
        raise TensorKernelError("Multiterm reduction exceeded its operation budget.")

    def _reduce_once(self, expression: TensorExpr) -> tuple[bool, TensorExpr, ReductionTraceStep | None]:
        terms = list(expression.terms)
        term_by_key: dict[TermKey, TensorTerm] = {term.key(): term for term in terms}
        for identity in self.identities:
            pivot = identity.pivot_key
            target = term_by_key.get(pivot)
            if target is None:
                continue
            replacements = [
                TensorTerm(target.coefficient * item.coefficient, item.factors)
                for item in identity.replacement_terms()
            ]
            new_terms = [term for term in terms if term.key() != pivot]
            new_terms.extend(replacements)
            step = ReductionTraceStep(identity.name, pivot, target.coefficient)
            return True, canonicalize(TensorExpr(tuple(new_terms))), step
        return False, expression, None


def reduce_multiterm(expression: TensorExpr, identities: Iterable[LinearIdentity], *, max_steps: int = 256) -> TensorExpr:
    """Reduce an expression with a temporary multiterm-reduction system."""
    return MultitermReductionSystem(tuple(identities), max_steps=max_steps).reduce(expression)


def reduce_multiterm_with_trace(expression: TensorExpr, identities: Iterable[LinearIdentity], *, max_steps: int = 256) -> ReductionResult:
    """Reduce an expression and return a compact proof trace."""
    return MultitermReductionSystem(tuple(identities), max_steps=max_steps).reduce_with_trace(expression)


def identity_from_expression(expression: TensorExpr, *, name: str = "identity") -> LinearIdentity:
    """Build a homogeneous identity from an expression expected to vanish."""
    expr = canonicalize(expression)
    if expr.is_zero:
        raise TensorKernelError("Linear identities must contain at least one nonzero term.")
    return LinearIdentity(expr, name=name, preferred_pivot_key=expr.terms[0].key())


def first_bianchi_identity(curvature: TensorHead, raised: AbstractIndex, first: AbstractIndex, second: AbstractIndex, third: AbstractIndex) -> LinearIdentity:
    """Return the mixed-curvature first Bianchi relation as a linear identity."""
    if curvature.rank != 4 or curvature.variance != ("up", "down", "down", "down"):
        raise TensorKernelError("Expected mixed curvature head with variance (^a, _b, _c, _d).")
    expr = canonicalize(
        curvature(raised, first, second, third)
        + curvature(raised, second, third, first)
        + curvature(raised, third, first, second)
    )
    reduced = canonicalize(expr)
    return LinearIdentity(reduced, name="first_bianchi", preferred_pivot_key=max(term.key() for term in reduced.terms))


def first_bianchi_reduction_system(curvature: TensorHead, raised: AbstractIndex, first: AbstractIndex, second: AbstractIndex, third: AbstractIndex) -> MultitermReductionSystem:
    """Create a reducer for one concrete first-Bianchi index orbit."""
    return MultitermReductionSystem((first_bianchi_identity(curvature, raised, first, second, third),))


def reduce_first_bianchi(expression: TensorExpr, curvature: TensorHead, raised: AbstractIndex, first: AbstractIndex, second: AbstractIndex, third: AbstractIndex) -> TensorExpr:
    """Reduce an expression modulo one concrete first-Bianchi cyclic relation."""
    return first_bianchi_reduction_system(curvature, raised, first, second, third).reduce(expression)


def cyclic_identity(head: TensorHead, indices: Sequence[AbstractIndex], cycle_slots: Sequence[int], *, name: str = "cyclic_identity") -> LinearIdentity:
    """Create a linear identity from a cyclic sum over selected slots.

    The supplied ``indices`` sequence is the full indexed tensor argument list.
    ``cycle_slots`` selects the positions to rotate.  This represents identities
    such as the first Bianchi cyclic relation without invoking generic pattern
    rewriting.
    """
    if len(indices) != head.rank:
        raise TensorKernelError("Identity indices must match tensor rank.")
    slots = tuple(cycle_slots)
    if len(slots) < 2 or len(set(slots)) != len(slots):
        raise TensorKernelError("Cycle slots must be distinct and contain at least two entries.")
    base = list(indices)
    expr = TensorExpr.zero()
    current = list(slots)
    for _ in range(len(slots)):
        rotated = list(base)
        values = [indices[pos] for pos in current]
        for pos, value in zip(slots, values):
            rotated[pos] = value
        expr = expr + head(*rotated)
        current = current[1:] + current[:1]
    reduced = canonicalize(expr)
    return LinearIdentity(reduced, name=name, preferred_pivot_key=max(term.key() for term in reduced.terms))


def antisymmetrized_identity(head: TensorHead, indices: Sequence[AbstractIndex], antisym_slots: Sequence[int], *, name: str = "antisymmetrized_identity") -> LinearIdentity:
    """Create a total antisymmetrization identity over selected slots.

    This is useful for Schouten- and dimension-dependent identities: when more
    indices are antisymmetrized than the dimension allows, the resulting sum is
    structurally zero and can be oriented as a rewrite relation.
    """
    if len(indices) != head.rank:
        raise TensorKernelError("Identity indices must match tensor rank.")
    slots = tuple(antisym_slots)
    if len(slots) < 2 or len(set(slots)) != len(slots):
        raise TensorKernelError("Antisymmetrization slots must be distinct and contain at least two entries.")
    expr = TensorExpr.zero()
    slot_values = [indices[pos] for pos in slots]
    for order in permutations(range(len(slots))):
        permuted = list(indices)
        sign = _permutation_sign(order)
        for pos, value_index in zip(slots, order):
            permuted[pos] = slot_values[value_index]
        expr = expr + sign * head(*permuted)
    reduced = canonicalize(expr)
    return LinearIdentity(reduced, name=name, preferred_pivot_key=max(term.key() for term in reduced.terms))


def dimension_dependent_antisymmetry_identity(
    head: TensorHead,
    indices: Sequence[AbstractIndex],
    antisym_slots: Sequence[int],
    *,
    dimension: int | None = None,
    name: str = "dimension_dependent_antisymmetry",
) -> LinearIdentity:
    """Create an identity for antisymmetrizing over too many slots.

    The identity is valid when ``len(antisym_slots)`` exceeds the supplied
    dimension.  If ``dimension`` is omitted, it is inferred from the first index
    type in the selected slots when that dimension is an integer.
    """
    slots = tuple(antisym_slots)
    if not slots:
        raise TensorKernelError("At least one slot is required.")
    inferred = dimension
    if inferred is None:
        inferred = indices[slots[0]].index_type.dimension
    if not isinstance(inferred, int):
        raise TensorKernelError("A concrete integer dimension is required for this identity.")
    if len(slots) <= inferred:
        raise TensorKernelError("Antisymmetrization does not exceed the supplied dimension.")
    return antisymmetrized_identity(head, indices, slots, name=name)


def schouten_identity(head: TensorHead, indices: Sequence[AbstractIndex], antisym_slots: Sequence[int], *, dimension: int | None = None) -> LinearIdentity:
    """Alias for a dimension-dependent total-antisymmetry relation."""
    return dimension_dependent_antisymmetry_identity(
        head,
        indices,
        antisym_slots,
        dimension=dimension,
        name="schouten_identity",
    )


def identity_orbit(identity: LinearIdentity, index_renamings: Iterable[dict[AbstractIndex, AbstractIndex]], *, name_prefix: str | None = None) -> tuple[LinearIdentity, ...]:
    """Generate renamed copies of a linear identity.

    This small helper covers many practical curvature-identity use cases without
    a full pattern matcher.  Each mapping replaces exact index objects appearing
    in the identity expression.
    """
    out: list[LinearIdentity] = []
    for number, mapping in enumerate(index_renamings, start=1):
        terms: list[TensorTerm] = []
        for term in identity.expression.terms:
            new_factors = []
            for factor in term.factors:
                new_indices = tuple(mapping.get(idx, idx) for idx in factor.indices)
                new_factors.append(factor.with_indices(new_indices))
            terms.append(TensorTerm(term.coefficient, tuple(new_factors)))
        label = f"{name_prefix or identity.name}_{number}"
        preferred = None
        for term in terms:
            if term.key() == identity.pivot_key:
                preferred = term.key()
                break
        if preferred is None:
            for old_term, new_term in zip(identity.expression.terms, terms):
                if old_term.key() == identity.pivot_key:
                    preferred = new_term.key()
                    break
        out.append(LinearIdentity(TensorExpr(tuple(terms)), name=label, preferred_pivot_key=preferred))
    return tuple(out)

@dataclass(frozen=True, slots=True)
class IdentityBasis:
    """Sparse row-echelon basis for homogeneous tensor identities.

    The basis orients each relation by its largest canonical monomial key and
    stores normalized replacement terms.  It is intentionally sparse and uses
    exact rational arithmetic, making it suitable for bounded tensor-polynomial
    reductions without invoking scalar CAS simplification.
    """

    identities: tuple[LinearIdentity, ...]

    @classmethod
    def from_expressions(cls, expressions: Iterable[TensorExpr], *, name: str = "identity") -> "IdentityBasis":
        return cls(tuple(identity_from_expression(expr, name=f"{name}_{number}") for number, expr in enumerate(expressions, start=1)))

    @property
    def pivot_keys(self) -> tuple[TermKey, ...]:
        return tuple(identity.pivot_key for identity in self.identities)

    def reducer(self, *, max_steps: int = 512) -> MultitermReductionSystem:
        return MultitermReductionSystem(self.identities, max_steps=max_steps)

    def reduce(self, expression: TensorExpr, *, max_steps: int = 512) -> TensorExpr:
        return self.reducer(max_steps=max_steps).reduce(expression)

    def reduce_with_trace(self, expression: TensorExpr, *, max_steps: int = 512) -> ReductionResult:
        return self.reducer(max_steps=max_steps).reduce_with_trace(expression)


def term_coefficient_map(expression: TensorExpr) -> dict[TermKey, Fraction]:
    """Return a sparse coefficient map for a canonical expression."""
    expr = canonicalize(expression)
    return {term.key(): term.coefficient for term in expr.terms}


def identity_basis(identities: Iterable[LinearIdentity]) -> IdentityBasis:
    """Build a sparse identity basis from already oriented identities."""
    return IdentityBasis(tuple(identities))


def expression_from_coefficient_map(coefficients: Mapping[TermKey, Fraction], representatives: Mapping[TermKey, tuple]) -> TensorExpr:
    """Reconstruct an expression from sparse coefficients and factor representatives."""
    terms = [TensorTerm(coeff, representatives[key]) for key, coeff in coefficients.items() if coeff]
    return canonicalize(TensorExpr(tuple(terms)))


def generate_renaming_orbit(identity: LinearIdentity, replacements: Sequence[tuple[AbstractIndex, ...]], *, name_prefix: str | None = None) -> tuple[LinearIdentity, ...]:
    """Generate identity copies by replacing the identity's sorted free-index set.

    This is a convenience wrapper for curvature and Schouten calculations where
    one identity template must be applied over many concrete index labels.
    """
    free = sorted(
        {idx for term in identity.expression.terms for factor in term.factors for idx in factor.indices},
        key=lambda idx: (idx.name, idx.variance, idx.index_type.name, id(idx.index_type)),
    )
    mappings = []
    for repl in replacements:
        if len(repl) != len(free):
            raise TensorKernelError("Renaming orbit replacement tuple has the wrong length.")
        mappings.append(dict(zip(free, repl)))
    return identity_orbit(identity, mappings, name_prefix=name_prefix)


def young_antisymmetry_identity(head: TensorHead, indices: Sequence[AbstractIndex], column_slots: Sequence[int], *, name: str = "young_column_antisymmetry") -> LinearIdentity:
    """Return the antisymmetry relation associated with one Young-tableau column."""
    return antisymmetrized_identity(head, indices, column_slots, name=name)


def young_row_symmetry_identity(head: TensorHead, indices: Sequence[AbstractIndex], row_slots: Sequence[int], *, name: str = "young_row_symmetry") -> LinearIdentity:
    """Return adjacent pair-symmetry relations for a Young-tableau row."""
    if len(indices) != head.rank:
        raise TensorKernelError("Identity indices must match tensor rank.")
    slots = tuple(row_slots)
    if len(slots) < 2:
        raise TensorKernelError("At least two row slots are required.")
    expr = TensorExpr.zero()
    for left, right in zip(slots, slots[1:]):
        swapped = list(indices)
        swapped[left], swapped[right] = swapped[right], swapped[left]
        expr = expr + head(*indices) - head(*swapped)
    reduced = canonicalize(expr)
    return LinearIdentity(reduced, name=name, preferred_pivot_key=max(term.key() for term in reduced.terms))


def identity_closure(identities: Iterable[LinearIdentity], renamings: Iterable[dict[AbstractIndex, AbstractIndex]] = ()) -> IdentityBasis:
    """Return a sparse identity basis including optional renamed identity copies."""
    base = tuple(identities)
    extra = []
    for identity in base:
        extra.extend(identity_orbit(identity, renamings, name_prefix=identity.name))
    return IdentityBasis(base + tuple(extra))


def _coefficient_map_and_representatives(expression: TensorExpr) -> tuple[dict[TermKey, Fraction], dict[TermKey, tuple]]:
    expr = canonicalize(expression)
    coeffs: dict[TermKey, Fraction] = {}
    reps: dict[TermKey, tuple] = {}
    for term in expr.terms:
        key = term.key()
        coeffs[key] = coeffs.get(key, Fraction(0)) + term.coefficient
        reps[key] = term.factors
    return {key: value for key, value in coeffs.items() if value}, reps


def echelon_identity_basis(identities: Iterable[LinearIdentity], *, name_prefix: str = "identity") -> IdentityBasis:
    """Build a sparse row-echelon basis for a finite identity set."""
    rows: dict[TermKey, tuple[dict[TermKey, Fraction], dict[TermKey, tuple]]] = {}
    for identity in identities:
        coeffs, reps = _coefficient_map_and_representatives(identity.expression)
        preferred = identity.preferred_pivot_key if identity.preferred_pivot_key in coeffs else None
        while coeffs:
            pivot = preferred if preferred in coeffs else max(coeffs)
            existing = rows.get(pivot)
            if existing is None:
                pivot_coeff = coeffs[pivot]
                rows[pivot] = ({key: value / pivot_coeff for key, value in coeffs.items() if value}, reps)
                break
            preferred = None
            pivot_row, pivot_reps = existing
            scale = coeffs[pivot] / pivot_row[pivot]
            merged_reps = dict(pivot_reps)
            merged_reps.update(reps)
            coeffs = {
                key: coeffs.get(key, Fraction(0)) - scale * pivot_row.get(key, Fraction(0))
                for key in set(coeffs) | set(pivot_row)
            }
            coeffs = {key: value for key, value in coeffs.items() if value}
            reps = merged_reps
    out: list[LinearIdentity] = []
    for number, pivot in enumerate(sorted(rows, reverse=True), start=1):
        coeffs, reps = rows[pivot]
        expr = expression_from_coefficient_map(coeffs, reps)
        out.append(LinearIdentity(expr, name=f"{name_prefix}_{number}", preferred_pivot_key=pivot))
    return IdentityBasis(tuple(out))


def normal_form(expression: TensorExpr, identities: Iterable[LinearIdentity], *, max_steps: int = 1024) -> TensorExpr:
    """Reduce an expression with an echelonized identity basis."""
    return echelon_identity_basis(tuple(identities)).reduce(expression, max_steps=max_steps)


def normal_form_with_trace(expression: TensorExpr, identities: Iterable[LinearIdentity], *, max_steps: int = 1024) -> ReductionResult:
    """Reduce an expression with an echelonized basis and return the trace."""
    return echelon_identity_basis(tuple(identities)).reduce_with_trace(expression, max_steps=max_steps)


def weyl_ricci_scalar_decomposition_identity(
    riemann: TensorHead,
    weyl: TensorHead,
    ricci: TensorHead,
    scalar: TensorHead,
    metric: TensorHead,
    indices: Sequence[AbstractIndex],
    *,
    dimension: int,
    name: str = "weyl_ricci_scalar_decomposition",
) -> LinearIdentity:
    """Return a structural all-lower Riemann decomposition identity."""
    if dimension <= 2:
        raise TensorKernelError("Weyl/Ricci/scalar decomposition requires dimension greater than two.")
    if len(indices) != 4 or not all(idx.is_down for idx in indices):
        raise TensorKernelError("Riemann decomposition helper expects four covariant indices.")
    a, b, c, d = indices
    coeff1 = Fraction(1, dimension - 2)
    coeff2 = Fraction(1, (dimension - 1) * (dimension - 2))
    expr = (
        riemann(a, b, c, d)
        - weyl(a, b, c, d)
        - coeff1 * (metric(a, c) * ricci(b, d) - metric(a, d) * ricci(b, c) - metric(b, c) * ricci(a, d) + metric(b, d) * ricci(a, c))
        + coeff2 * scalar() * (metric(a, c) * metric(b, d) - metric(a, d) * metric(b, c))
    )
    reduced = canonicalize(expr)
    return LinearIdentity(reduced, name=name, preferred_pivot_key=max(term.key() for term in reduced.terms))
