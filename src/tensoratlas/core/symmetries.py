"""Slot-symmetry descriptors for the semantic tensor kernel."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Iterable

from .manifolds import TensorKernelError


@dataclass(frozen=True, slots=True)
class SlotSymmetry:
    """A monoterm slot-symmetry descriptor.

    ``kind`` covers common built-in symmetry families.  ``signed_generators``
    optionally stores arbitrary signed local-slot generators as
    ``((image_0, image_1, ...), sign)`` pairs.  This representation is kept
    independent of the permutation backend so tensor-head metadata remains
    lightweight and import-friendly.
    """

    kind: str = "none"
    signed_generators: tuple[tuple[tuple[int, ...], int], ...] = ()

    def __init__(
        self,
        kind: str = "none",
        signed_generators: Iterable[tuple[Iterable[int], int]] = (),
    ):
        object.__setattr__(self, "kind", kind)
        normalized = tuple((tuple(int(point) for point in mapping), int(sign)) for mapping, sign in signed_generators)
        object.__setattr__(self, "signed_generators", normalized)
        self.__post_init__()

    @classmethod
    def from_generators(cls, rank: int, generators: Iterable[tuple[Iterable[int], int]]) -> "SlotSymmetry":
        """Create an arbitrary monoterm symmetry from local slot generators."""
        normalized = []
        for mapping, sign in generators:
            data = tuple(int(point) for point in mapping)
            if len(data) != rank or sorted(data) != list(range(rank)):
                raise TensorKernelError("Slot-symmetry generators must be local permutations of the tensor rank.")
            normalized.append((data, int(sign)))
        return cls("custom", tuple(normalized))

    def __post_init__(self) -> None:
        allowed = {"none", "symmetric", "antisymmetric", "antisym_last2", "riemann", "weyl", "custom"}
        if self.kind not in allowed:
            raise TensorKernelError(f"Unsupported slot symmetry: {self.kind!r}.")
        for mapping, sign in self.signed_generators:
            if sign not in {-1, 1}:
                raise TensorKernelError("Slot-symmetry generator signs must be +1 or -1.")
            if sorted(mapping) != list(range(len(mapping))):
                raise TensorKernelError("Slot-symmetry generator mappings must be permutations.")

    def canonicalize_indices(self, indices: tuple) -> tuple[int, tuple]:
        if self.kind == "none" or len(indices) < 2:
            return 1, indices
        if self.kind == "symmetric":
            return 1, tuple(sorted(indices, key=repr))
        if self.kind == "antisymmetric":
            return _canonical_antisymmetric(indices)
        if self.kind == "antisym_last2":
            return _canonical_antisym_last2(indices)
        if self.kind in {"riemann", "weyl"}:
            return _canonical_riemann_like(indices)
        if self.kind == "custom":
            # Custom symmetry canonicalization is delegated to the permutation
            # backend.  The lightweight local method leaves the factor alone.
            return 1, indices
        return 1, indices


def _permutation_parity(original: tuple, candidate: tuple) -> int:
    positions = {id(value): pos for pos, value in enumerate(original)}
    order = [positions[id(value)] for value in candidate]
    inversions = 0
    for left in range(len(order)):
        for right in range(left + 1, len(order)):
            if order[left] > order[right]:
                inversions += 1
    return -1 if inversions % 2 else 1


def _canonical_antisymmetric(indices: tuple) -> tuple[int, tuple]:
    if len({(idx.name, idx.index_type, idx.variance) for idx in indices}) < len(indices):
        return 0, indices
    best = None
    best_sign = 1
    for perm in permutations(indices):
        key = tuple(repr(item) for item in perm)
        sign = _permutation_parity(indices, perm)
        if best is None or key < best[0]:
            best = (key, perm)
            best_sign = sign
    return best_sign, tuple(best[1])


def _canonical_pair_antisym(pair: tuple) -> tuple[int, tuple]:
    left, right = pair
    if (left.name, left.index_type, left.variance) == (right.name, right.index_type, right.variance):
        return 0, pair
    if repr(right) < repr(left):
        return -1, (right, left)
    return 1, pair


def _canonical_riemann_like(indices: tuple) -> tuple[int, tuple]:
    if len(indices) != 4:
        return 1, indices
    s1, p1 = _canonical_pair_antisym(indices[:2])
    s2, p2 = _canonical_pair_antisym(indices[2:])
    if s1 == 0 or s2 == 0:
        return 0, indices
    sign = s1 * s2
    first = tuple(p1)
    second = tuple(p2)
    if tuple(map(repr, second)) < tuple(map(repr, first)):
        first, second = second, first
    return sign, first + second


def _canonical_antisym_last2(indices: tuple) -> tuple[int, tuple]:
    """Canonicalize tensors antisymmetric in their last two slots."""
    if len(indices) < 2:
        return 1, indices
    prefix = tuple(indices[:-2])
    sign, pair = _canonical_pair_antisym(tuple(indices[-2:]))
    if sign == 0:
        return 0, indices
    return sign, prefix + tuple(pair)
