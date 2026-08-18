"""Pure-Python permutation-group backend for tensor-index canonicalization.

The backend is correctness-first and intentionally mirrors the public shape of
an optimized canonicalization backend.  It supports signed generators, explicit
small-group closure for oracle tests, stabilizer-chain operations, Schreier
transversal data, and a backend protocol that a later Rust + pyo3 extension can
implement without changing the semantic tensor layer.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence, runtime_checkable

from .manifolds import TensorKernelError

DEFAULT_CLOSURE_LIMIT = 100_000


@dataclass(frozen=True, slots=True, order=True)
class Permutation:
    """A zero-based permutation represented by its image tuple.

    ``mapping[i]`` is the image of point ``i``.  Composition uses the standard
    function convention: ``p.compose(q)`` means ``p ∘ q`` and maps
    ``i -> p[q[i]]``.
    """

    mapping: tuple[int, ...]

    def __init__(self, mapping: Iterable[int]):
        data = tuple(int(item) for item in mapping)
        if sorted(data) != list(range(len(data))):
            raise TensorKernelError(f"Invalid permutation mapping: {data!r}.")
        object.__setattr__(self, "mapping", data)

    @classmethod
    def identity(cls, degree: int) -> "Permutation":
        if degree < 0:
            raise TensorKernelError("Permutation degree must be nonnegative.")
        return cls(range(degree))

    @classmethod
    def transposition(cls, degree: int, first: int, second: int) -> "Permutation":
        if not (0 <= first < degree and 0 <= second < degree):
            raise TensorKernelError("Transposition points must lie inside the permutation degree.")
        data = list(range(degree))
        data[first], data[second] = data[second], data[first]
        return cls(data)

    @classmethod
    def cycle(cls, degree: int, *points: int) -> "Permutation":
        if any(point < 0 or point >= degree for point in points):
            raise TensorKernelError("Cycle points must lie inside the permutation degree.")
        data = list(range(degree))
        if points:
            for source, target in zip(points, points[1:] + points[:1]):
                data[source] = target
        return cls(data)

    @property
    def degree(self) -> int:
        return len(self.mapping)

    def apply(self, point: int) -> int:
        return self.mapping[point]

    def apply_to_sequence(self, values: Sequence) -> tuple:
        """Return the active image of a sequence under this permutation.

        The result has position ``self[i]`` filled by ``values[i]``.  This is the
        convention used for slot-label canonicalization.
        """
        if len(values) != self.degree:
            raise TensorKernelError("Sequence length must match permutation degree.")
        out = [None] * self.degree
        for source, target in enumerate(self.mapping):
            out[target] = values[source]
        return tuple(out)

    def compose(self, other: "Permutation") -> "Permutation":
        self._check_same_degree(other)
        return Permutation(self.mapping[other.mapping[i]] for i in range(self.degree))

    def inverse(self) -> "Permutation":
        out = [0] * self.degree
        for source, target in enumerate(self.mapping):
            out[target] = source
        return Permutation(out)

    def parity(self) -> int:
        inversions = 0
        for left in range(self.degree):
            for right in range(left + 1, self.degree):
                if self.mapping[left] > self.mapping[right]:
                    inversions += 1
        return -1 if inversions % 2 else 1

    def extend(self, degree: int) -> "Permutation":
        if degree < self.degree:
            raise TensorKernelError("Cannot extend a permutation to a smaller degree.")
        return Permutation(self.mapping + tuple(range(self.degree, degree)))

    def _check_same_degree(self, other: "Permutation") -> None:
        if self.degree != other.degree:
            raise TensorKernelError("Permutation degrees must match.")

    def __len__(self) -> int:
        return self.degree


@dataclass(frozen=True, slots=True, order=True)
class SignedPermutation:
    """A permutation with an independent multiplicative sign."""

    permutation: Permutation
    sign: int = 1

    def __post_init__(self) -> None:
        if self.sign not in {-1, 1}:
            raise TensorKernelError("Signed permutations require sign +1 or -1.")

    @classmethod
    def identity(cls, degree: int) -> "SignedPermutation":
        return cls(Permutation.identity(degree), 1)

    @property
    def degree(self) -> int:
        return self.permutation.degree

    def compose(self, other: "SignedPermutation") -> "SignedPermutation":
        if self.degree != other.degree:
            raise TensorKernelError("Signed permutation degrees must match.")
        return SignedPermutation(self.permutation.compose(other.permutation), self.sign * other.sign)

    def inverse(self) -> "SignedPermutation":
        return SignedPermutation(self.permutation.inverse(), self.sign)

    def apply_to_sequence(self, values: Sequence) -> tuple:
        return self.permutation.apply_to_sequence(values)

    def unsigned(self) -> Permutation:
        return self.permutation


@dataclass(frozen=True, slots=True)
class PermutationGroup:
    """A finite signed permutation group generated by signed permutations.

    The public operations prefer generator/transversal data where feasible and
    compute explicit closures only for correctness checks, small groups, and the
    reference double-coset backend.
    """

    degree: int
    generators: tuple[SignedPermutation, ...]

    def __init__(self, degree: int, generators: Iterable[Permutation | SignedPermutation] = ()):  # noqa: D107
        if degree < 0:
            raise TensorKernelError("Permutation-group degree must be nonnegative.")
        converted: list[SignedPermutation] = []
        for generator in generators:
            signed = _as_signed(generator, degree)
            if signed.degree != degree:
                raise TensorKernelError("Generator degree must match group degree.")
            converted.append(signed)
        object.__setattr__(self, "degree", degree)
        object.__setattr__(self, "generators", tuple(converted))

    @classmethod
    def trivial(cls, degree: int) -> "PermutationGroup":
        return cls(degree, ())

    @classmethod
    def symmetric(cls, degree: int, points: Iterable[int] | None = None) -> "PermutationGroup":
        block = tuple(range(degree) if points is None else points)
        generators = [Permutation.transposition(degree, left, right) for left, right in zip(block, block[1:])]
        return cls(degree, generators)

    @classmethod
    def antisymmetric(cls, degree: int, points: Iterable[int] | None = None) -> "PermutationGroup":
        block = tuple(range(degree) if points is None else points)
        generators = [SignedPermutation(Permutation.transposition(degree, left, right), -1) for left, right in zip(block, block[1:])]
        return cls(degree, generators)

    @property
    def identity(self) -> SignedPermutation:
        return SignedPermutation.identity(self.degree)

    @property
    def inverse_generators(self) -> tuple[SignedPermutation, ...]:
        return tuple(generator.inverse() for generator in self.generators)

    @property
    def signed_generators(self) -> tuple[SignedPermutation, ...]:
        """Return the signed generators explicitly stored by this group."""
        return self.generators

    @property
    def unsigned_generators(self) -> tuple[Permutation, ...]:
        return tuple(generator.permutation for generator in self.generators)

    def closure(self, *, max_size: int | None = DEFAULT_CLOSURE_LIMIT) -> tuple[SignedPermutation, ...]:
        """Return the explicit closure generated by this group's generators.

        Explicit closure is a correctness/reference path.  The guard prevents
        accidental factorial blowups in tensor canonicalization.
        """
        generators = tuple(self.generators) + self.inverse_generators
        seen = {self.identity}
        queue: deque[SignedPermutation] = deque([self.identity])
        while queue:
            current = queue.popleft()
            for generator in generators:
                candidate = generator.compose(current)
                if candidate in seen:
                    continue
                seen.add(candidate)
                if max_size is not None and len(seen) > max_size:
                    raise TensorKernelError(
                        f"Permutation-group closure exceeded limit {max_size}; "
                        "use a stabilizer/native backend for this problem."
                    )
                queue.append(candidate)
        return tuple(sorted(seen, key=lambda item: (item.permutation.mapping, item.sign)))

    @property
    def elements(self) -> tuple[SignedPermutation, ...]:
        """Return explicit closure with the default safety guard."""
        return self.closure()

    def order_bounded(self, *, max_size: int | None = DEFAULT_CLOSURE_LIMIT) -> int:
        return len(self.closure(max_size=max_size))

    @property
    def order(self) -> int:
        return self.order_bounded()

    def contains(self, item: Permutation | SignedPermutation) -> bool:
        signed = _as_signed(item, self.degree)
        chain = schreier_sims(self)
        return chain.contains(signed)

    def orbit(self, point: int) -> tuple[int, ...]:
        return tuple(sorted(self.orbit_transversal(point).keys()))

    def orbit_transversal(self, point: int) -> dict[int, SignedPermutation]:
        """Return a generator-BFS transversal for the orbit of ``point``.

        The dictionary maps each reached image to a signed group element sending
        ``point`` to that image.  This avoids full closure when the orbit is
        much smaller than the group.
        """
        if point < 0 or point >= self.degree:
            raise TensorKernelError("Orbit point must lie inside the group degree.")
        moves = tuple(self.generators) + self.inverse_generators
        start = self.identity
        transversals: dict[int, SignedPermutation] = {point: start}
        queue: deque[SignedPermutation] = deque([start])
        while queue:
            current = queue.popleft()
            for generator in moves:
                candidate = generator.compose(current)
                image = candidate.permutation.apply(point)
                if image in transversals:
                    continue
                transversals[image] = candidate
                queue.append(candidate)
        return transversals

    def stabilizer(self, point: int) -> "PermutationGroup":
        if point < 0 or point >= self.degree:
            raise TensorKernelError("Stabilizer point must lie inside the group degree.")
        generators = schreier_generators(self, point)
        if not generators:
            fixed = [element for element in self.closure() if element.permutation.apply(point) == point]
            generators = tuple(fixed)
        return PermutationGroup(self.degree, generators)


@dataclass(frozen=True, slots=True)
class SiftResult:
    """Result of sifting a signed permutation through a stabilizer chain."""

    residue: SignedPermutation
    representative: SignedPermutation
    level: int
    success: bool


@dataclass(frozen=True, slots=True)
class StabilizerChain:
    """Stabilizer-chain data and operations for a signed permutation group."""

    degree: int
    base: tuple[int, ...]
    strong_generators: tuple[SignedPermutation, ...]
    orders: tuple[int, ...]
    orbits: tuple[tuple[int, ...], ...]
    transversals: tuple[dict[int, SignedPermutation], ...]
    signed_identity_available: bool = False

    @property
    def group_order(self) -> int:
        return self.orders[0] if self.orders else 1

    def coset_representative(self, level: int, image: int) -> SignedPermutation | None:
        """Return a transversal element sending ``base[level]`` to ``image``."""
        if level < 0 or level >= len(self.transversals):
            raise TensorKernelError("Stabilizer-chain level out of range.")
        return self.transversals[level].get(image)

    def sift(self, item: Permutation | SignedPermutation) -> SiftResult:
        """Sift ``item`` through the chain using stored transversals.

        A successful sift leaves the identity residue.  The returned
        representative is the product of transversals that reproduces the input
        when the residue is identity.  For this signed backend, signs are part of
        membership: an unsigned permutation reached with the wrong sign fails.
        """
        target = _as_signed(item, self.degree)
        residue = target
        representative = SignedPermutation.identity(self.degree)
        for level, base_point in enumerate(self.base):
            image = residue.permutation.apply(base_point)
            transversal = self.coset_representative(level, image)
            if transversal is None:
                return SiftResult(residue=residue, representative=representative, level=level, success=False)
            # Remove this orbit motion from the residue and accumulate it.
            residue = transversal.inverse().compose(residue)
            representative = representative.compose(transversal)
        identity = SignedPermutation.identity(self.degree)
        success = residue == identity
        if not success and self.signed_identity_available:
            signed_identity = SignedPermutation(Permutation.identity(self.degree), -1)
            if residue == signed_identity:
                representative = representative.compose(signed_identity)
                residue = identity
                success = True
        return SiftResult(
            residue=residue,
            representative=representative,
            level=len(self.base),
            success=success,
        )

    def contains(self, item: Permutation | SignedPermutation) -> bool:
        return self.sift(item).success

    def membership_test(self, item: Permutation | SignedPermutation) -> bool:
        return self.contains(item)


@dataclass(frozen=True, slots=True)
class CanonicalDoubleCosetResult:
    """Canonical representative found for a signed double-coset problem."""

    canonical: SignedPermutation | None
    image: tuple
    sign: int
    zero: bool
    candidates_considered: int


@runtime_checkable
class CanonicalizationBackend(Protocol):
    """Backend protocol for permutation double-coset canonicalization.

    A native Rust + pyo3 backend should implement this protocol so tensor-level
    code can swap backend implementations without depending on native details.
    """

    def schreier_sims(self, group: PermutationGroup, base: Iterable[int] | None = None) -> StabilizerChain:
        ...

    def canonicalize_double_coset(
        self,
        left_group: PermutationGroup,
        representative: Permutation | SignedPermutation,
        right_group: PermutationGroup,
        *,
        labels: Sequence | None = None,
        base: Iterable[int] | None = None,
    ) -> CanonicalDoubleCosetResult:
        ...


@dataclass(frozen=True, slots=True)
class PythonPermutationBackend:
    """Reference backend implemented in pure Python."""

    def schreier_sims(self, group: PermutationGroup, base: Iterable[int] | None = None) -> StabilizerChain:
        return schreier_sims(group, base)

    def canonicalize_double_coset(
        self,
        left_group: PermutationGroup,
        representative: Permutation | SignedPermutation,
        right_group: PermutationGroup,
        *,
        labels: Sequence | None = None,
        base: Iterable[int] | None = None,
    ) -> CanonicalDoubleCosetResult:
        return canonical_double_coset_reference(left_group, representative, right_group, labels=labels, base=base)


def schreier_sims(group: PermutationGroup, base: Iterable[int] | None = None) -> StabilizerChain:
    """Build a stabilizer chain for a finite signed permutation group.

    This correctness-first implementation uses generator-BFS transversals and
    Schreier generators at each level, while still using explicit closure to
    derive exact signed strong generators for small groups.  The shape mirrors a
    native stabilizer-chain implementation and supports sifting, membership,
    coset representative lookup, and Schreier-generator inspection.
    """
    if base is None:
        base_tuple = tuple(point for point in range(group.degree) if len(group.orbit(point)) > 1)
    else:
        base_tuple = tuple(base)
    for point in base_tuple:
        if point < 0 or point >= group.degree:
            raise TensorKernelError("Base points must lie inside the group degree.")

    current = group
    orders: list[int] = [current.order]
    orbits: list[tuple[int, ...]] = []
    transversals: list[dict[int, SignedPermutation]] = []
    for point in base_tuple:
        transversal = current.orbit_transversal(point)
        orbit = tuple(sorted(transversal))
        orbits.append(orbit)
        transversals.append(transversal)
        current = current.stabilizer(point)
        orders.append(current.order)
    strong = _derive_strong_generators(group, base_tuple)
    return StabilizerChain(
        degree=group.degree,
        base=base_tuple,
        strong_generators=strong,
        orders=tuple(orders),
        orbits=tuple(orbits),
        transversals=tuple(transversals),
        signed_identity_available=SignedPermutation(Permutation.identity(group.degree), -1) in group.closure(),
    )


def schreier_generators(group: PermutationGroup, base_point: int) -> tuple[SignedPermutation, ...]:
    """Construct Schreier generators for the stabilizer of ``base_point``.

    For each orbit transversal ``u_x`` and each group generator ``s``, the
    Schreier generator is ``u_{x^s}^{-1} s u_x``.  Identity generators are
    omitted.  Signs are preserved throughout.
    """
    if base_point < 0 or base_point >= group.degree:
        raise TensorKernelError("Base point must lie inside the group degree.")
    transversals = group.orbit_transversal(base_point)
    moves = tuple(group.generators) + group.inverse_generators
    identity = group.identity
    out: list[SignedPermutation] = []
    seen: set[SignedPermutation] = set()
    for image, transversal in transversals.items():
        for generator in moves:
            moved = generator.permutation.apply(image)
            target_transversal = transversals[moved]
            candidate = target_transversal.inverse().compose(generator).compose(transversal)
            if candidate == identity or candidate in seen:
                continue
            seen.add(candidate)
            out.append(candidate)
    return tuple(out)


def reference_double_coset(
    left_group: PermutationGroup,
    representative: Permutation | SignedPermutation,
    right_group: PermutationGroup,
    *,
    labels: Sequence | None = None,
    base: Iterable[int] | None = None,
) -> CanonicalDoubleCosetResult:
    """Return a canonical representative of ``left * representative * right``.

    This reference implementation deliberately enumerates the two explicitly
    generated small groups and is used as an oracle for optional stabilizer-chain
    and native backends.  If the same minimal image is attainable with both
    signs, ``zero`` is set to ``True``.
    """
    degree = left_group.degree
    if right_group.degree != degree:
        raise TensorKernelError("Left and right groups must have the same degree.")
    middle = _as_signed(representative, degree)
    if labels is None:
        labels_tuple = tuple(range(degree))
    else:
        labels_tuple = tuple(labels)
        if len(labels_tuple) != degree:
            raise TensorKernelError("Label length must match the double-coset degree.")

    base_tuple = tuple(base or ())
    for point in base_tuple:
        if point < 0 or point >= degree:
            raise TensorKernelError("Base points must lie inside the double-coset degree.")

    def _base_key(item: SignedPermutation) -> tuple:
        if not base_tuple:
            return item.permutation.mapping
        image = item.apply_to_sequence(labels_tuple)
        return tuple(image[pos] for pos in base_tuple) + item.permutation.mapping

    left_elements = tuple(sorted(left_group.elements, key=_base_key))
    right_elements = tuple(sorted(right_group.elements, key=_base_key))

    best_image: tuple | None = None
    best_candidate: SignedPermutation | None = None
    best_signs: set[int] = set()
    count = 0
    for left in left_elements:
        left_middle = left.compose(middle)
        for right in right_elements:
            candidate = left_middle.compose(right)
            image = candidate.apply_to_sequence(labels_tuple)
            count += 1
            if best_image is None or image < best_image:
                best_image = image
                best_candidate = candidate
                best_signs = {candidate.sign}
            elif image == best_image:
                best_signs.add(candidate.sign)
                if best_candidate is None or candidate.permutation.mapping < best_candidate.permutation.mapping:
                    best_candidate = candidate
    assert best_image is not None
    zero = len(best_signs) > 1
    sign = 0 if zero else next(iter(best_signs))
    canonical = None if zero else best_candidate
    return CanonicalDoubleCosetResult(canonical, best_image, sign, zero, count)


def canonical_double_coset_reference(
    left_group: PermutationGroup,
    representative: Permutation | SignedPermutation,
    right_group: PermutationGroup,
    *,
    labels: Sequence | None = None,
    base: Iterable[int] | None = None,
) -> CanonicalDoubleCosetResult:
    """Explicit closure-enumerating double-coset oracle.

    This is the correctness/reference implementation used by tests and by the
    current pure-Python backend.  It is intentionally not advertised as a fast
    production backend; optional native code should implement the backend
    protocol and leave this function as the oracle.
    """
    return reference_double_coset(left_group, representative, right_group, labels=labels, base=base)



def brute_force_double_coset(
    left_group: PermutationGroup,
    representative: Permutation | SignedPermutation,
    right_group: PermutationGroup,
    *,
    labels: Sequence | None = None,
) -> CanonicalDoubleCosetResult:
    """Alias for the explicit oracle used by tests and native-backend checks."""
    return canonical_double_coset_reference(left_group, representative, right_group, labels=labels)


def default_permutation_backend() -> CanonicalizationBackend:
    """Return the default reference backend.

    Future native loading should happen behind this function, with the pure
    Python implementation retained as the correctness oracle and fallback.
    """
    return PythonPermutationBackend()


def _as_signed(item: Permutation | SignedPermutation, degree: int) -> SignedPermutation:
    if isinstance(item, SignedPermutation):
        signed = item
    elif isinstance(item, Permutation):
        signed = SignedPermutation(item, 1)
    else:
        raise TensorKernelError(f"Expected Permutation or SignedPermutation, got {type(item)!r}.")
    if signed.degree != degree:
        raise TensorKernelError("Permutation degree mismatch.")
    return signed


def _derive_strong_generators(group: PermutationGroup, base: tuple[int, ...]) -> tuple[SignedPermutation, ...]:
    if not base:
        return group.closure()
    strong: list[SignedPermutation] = []
    seen: set[SignedPermutation] = set()
    for level in range(len(base) + 1):
        fixed_prefix = base[:level]
        for element in group.closure():
            if all(element.permutation.apply(point) == point for point in fixed_prefix):
                if element not in seen:
                    seen.add(element)
                    strong.append(element)
    return tuple(strong)
