"""Geometric-algebra definitions for orthogonal metric signatures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import sympy as sp

from tensoratlas.errors import MetricError, UnsupportedGeometryError
from tensoratlas.validation import ValidationReport, invalid_report, valid_report

Blade = tuple[int, ...]


def canonical_exterior_blade(indices: Iterable[int]) -> tuple[int, Blade]:
    """Return the sign and sorted exterior blade for distinct indices."""
    items = tuple(int(i) for i in indices)
    if len(set(items)) != len(items):
        return 0, ()
    inversions = 0
    for pos, left in enumerate(items):
        for right in items[pos + 1 :]:
            if left > right:
                inversions += 1
    return (-1 if inversions % 2 else 1), tuple(sorted(items))


@dataclass(frozen=True)
class GeometricAlgebra:
    """Symbolic geometric algebra over an orthogonal basis.

    Non-diagonal bilinear forms are intentionally rejected until the package has
    a complete Clifford product for arbitrary symmetric forms.  This prevents
    the implementation from silently returning exterior-algebra results for a
    genuinely Clifford-algebra computation.
    """

    basis_names: tuple[str, ...]
    metric: tuple[Any, ...] | sp.Matrix | None = None
    simplify: bool | Any = False

    def __init__(
        self,
        basis_names: Iterable[str] | int,
        metric: Iterable[Any] | sp.Matrix | None = None,
        simplify: bool | Any = False,
    ):
        if isinstance(basis_names, int):
            names = tuple(f"e{i + 1}" for i in range(basis_names))
        else:
            names = tuple(str(name) for name in basis_names)
        object.__setattr__(self, "basis_names", names)
        object.__setattr__(self, "metric", self._normalize_metric(metric, len(names)))
        object.__setattr__(self, "simplify", simplify)
        object.__setattr__(self, "_blade_product_cache", {})

    @staticmethod
    def _normalize_metric(metric: Iterable[Any] | sp.Matrix | None, dimension: int) -> tuple[Any, ...]:
        if metric is None:
            return tuple(sp.Integer(1) for _ in range(dimension))
        if isinstance(metric, sp.MatrixBase):
            if metric.shape != (dimension, dimension):
                raise MetricError("metric matrix shape must match basis dimension")
            for row in range(dimension):
                for col in range(dimension):
                    if row == col:
                        continue
                    entry = sp.sympify(metric[row, col])
                    if entry != 0 and sp.simplify(entry) != 0:
                        raise UnsupportedGeometryError(
                            "GeometricAlgebra currently supports diagonal/orthogonal metrics only; "
                            "non-diagonal bilinear forms are intentionally rejected."
                        )
            return tuple(sp.sympify(metric[i, i]) for i in range(dimension))
        values = tuple(sp.sympify(value) for value in metric)
        if len(values) != dimension:
            raise MetricError("metric diagonal length must match basis dimension")
        return values

    @classmethod
    def euclidean(cls, dimension: int, prefix: str = "e") -> "GeometricAlgebra":
        """Return a Euclidean algebra with basis ``prefix1``, ``prefix2``, ... ."""
        return cls(tuple(f"{prefix}{i + 1}" for i in range(dimension)), metric=(1,) * dimension)

    @classmethod
    def spacetime(cls) -> "GeometricAlgebra":
        """Return a mostly-plus spacetime algebra with basis gamma0..gamma3."""
        return cls(("gamma0", "gamma1", "gamma2", "gamma3"), metric=(-1, 1, 1, 1))

    @property
    def dimension(self) -> int:
        return len(self.basis_names)

    def summary(self) -> dict[str, Any]:
        """Return dimension, basis names, and diagonal metric entries."""
        return {"dimension": self.dimension, "basis_names": self.basis_names, "metric": self.metric}

    def validation_report(self) -> ValidationReport:
        """Return structured validation diagnostics without raising."""
        if len(self.metric) != self.dimension:
            return invalid_report("metric diagonal length must match basis dimension")
        return valid_report(warnings=("only orthogonal/diagonal metrics are supported",))

    def validate(self) -> bool:
        """Validate basis and metric dimensions."""
        self.validation_report().raise_as(MetricError)
        return True

    def scalar(self, value: Any):
        from .multivector import Multivector

        return Multivector(self, {(): sp.sympify(value)})

    def zero(self):
        from .multivector import Multivector

        return Multivector(self, {})

    def _basis_index(self, item: Any) -> int:
        if isinstance(item, str):
            try:
                return self.basis_names.index(item)
            except ValueError as exc:
                raise IndexError(f"unknown basis vector name {item!r}") from exc
        return int(item)

    def _coerce_indices(self, args: tuple[Any, ...]) -> tuple[int, ...]:
        if len(args) == 1 and not isinstance(args[0], (str, bytes, int)):
            try:
                return tuple(self._basis_index(index) for index in args[0])
            except TypeError:
                return (self._basis_index(args[0]),)
        return tuple(self._basis_index(index) for index in args)

    def blade(self, *indices: Any, coeff: Any = 1):
        """Return an exterior basis blade.

        Accepted forms include ``blade(0, 1)``, ``blade([0, 1])``, and
        ``blade("e1", "e2")`` when those basis names exist.  Repeated indices
        are exterior blades and therefore produce zero.  Use ``basis_product``
        for geometric products such as ``basis_product(0, 0)``.
        """
        from .multivector import Multivector

        sign, blade = canonical_exterior_blade(self._coerce_indices(indices))
        if sign == 0:
            return self.zero()
        return Multivector(self, {blade: sp.sympify(sign) * sp.sympify(coeff)})

    def vector(self, name_or_index: str | int, coeff: Any = 1):
        from .multivector import Multivector

        index = self._basis_index(name_or_index)
        if index < 0 or index >= self.dimension:
            raise IndexError("basis vector index is out of range")
        return Multivector(self, {(index,): sp.sympify(coeff)})

    def basis_vectors(self):
        """Return all basis vectors as a tuple suitable for unpacking."""
        return tuple(self.vector(index) for index in range(self.dimension))

    def basis_product(self, *indices: Any):
        """Return the geometric product of basis vectors.

        Accepted forms include ``basis_product(0, 0)``, ``basis_product([0, 0])``,
        and basis-name inputs.  Repeated indices are evaluated using the metric.
        """
        result = self.scalar(1)
        for index in self._coerce_indices(indices):
            result = result * self.vector(index)
        return result

    def blade_product(self, left: Blade, right: Blade) -> tuple[tuple[Blade, Any], ...]:
        """Return the geometric product of two canonical basis blades."""
        key = (tuple(left), tuple(right))
        cache = self._blade_product_cache
        if key in cache:
            return cache[key]
        coeff = sp.Integer(1)
        blade = tuple(int(i) for i in left)
        for item in right:
            index = int(item)
            if index in blade:
                position = blade.index(index)
                swaps = len(blade) - position - 1
                coeff *= -1 if swaps % 2 else 1
                coeff *= self.metric[index]
                blade = blade[:position] + blade[position + 1 :]
            else:
                swaps = sum(1 for existing in blade if existing > index)
                coeff *= -1 if swaps % 2 else 1
                insert_at = sum(1 for existing in blade if existing < index)
                blade = blade[:insert_at] + (index,) + blade[insert_at:]
        result = ((blade, coeff),)
        if len(cache) >= 8192:
            cache.pop(next(iter(cache)))
        cache[key] = result
        return result
