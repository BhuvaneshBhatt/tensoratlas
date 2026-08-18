"""Sparse multivectors for orthogonal-metric geometric algebra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, TYPE_CHECKING

import sympy as sp

from tensoratlas.errors import UnsupportedGeometryError
from tensoratlas.simplification_policy import normalize_simplifier
from tensoratlas.validation import ValidationReport, invalid_report, valid_report

from .algebra import Blade, canonical_exterior_blade

if TYPE_CHECKING:  # pragma: no cover
    from .algebra import GeometricAlgebra


@dataclass(frozen=True)
class Multivector:
    """Sparse symbolic multivector attached to a :class:`GeometricAlgebra`."""

    algebra: "GeometricAlgebra"
    coeffs: Mapping[Blade, Any]

    def __post_init__(self) -> None:
        clean = normalize_simplifier(self.algebra.simplify)
        raw: dict[Blade, Any] = {}
        for blade, coeff in self.coeffs.items():
            blade = tuple(int(index) for index in blade)
            if any(index < 0 or index >= self.algebra.dimension for index in blade):
                raise IndexError("blade index is out of range")
            sign, canonical = canonical_exterior_blade(blade)
            if sign == 0:
                continue
            raw[canonical] = raw.get(canonical, 0) + sp.sympify(sign) * sp.sympify(coeff)
        normalized = {blade: value for blade, value in ((blade, clean(value)) for blade, value in raw.items()) if value != 0}
        object.__setattr__(self, "coeffs", normalized)

    def summary(self) -> dict[str, Any]:
        """Return nonzero blades, grades, and coefficients."""
        return {"grades": sorted(self.grades()), "nonzero_blades": len(self.coeffs), "coeffs": dict(self.coeffs)}

    def validation_report(self) -> ValidationReport:
        """Return structured validation diagnostics without raising."""
        for blade in self.coeffs:
            bad = tuple(index for index in blade if index < 0 or index >= self.algebra.dimension)
            if bad:
                return invalid_report(f"blade index out of range: {bad}")
        return valid_report()

    def validate(self) -> bool:
        """Validate blade indices against the parent algebra dimension."""
        self.validation_report().raise_as(IndexError)
        return True

    def to_sympy(self) -> sp.Expr:
        """Return a symbolic additive expression using noncommutative basis names."""
        total = sp.Integer(0)
        for blade, coeff in self.coeffs.items():
            if blade == ():
                basis = sp.Integer(1)
            else:
                basis = sp.Symbol("^".join(self.algebra.basis_names[i] for i in blade), commutative=False)
            total += coeff * basis
        return total

    def _check_same_algebra(self, other: "Multivector") -> None:
        if self.algebra != other.algebra:
            raise UnsupportedGeometryError("multivectors belong to different geometric algebras")

    def __add__(self, other: "Multivector") -> "Multivector":
        if not isinstance(other, Multivector):
            return NotImplemented
        self._check_same_algebra(other)
        coeffs = dict(self.coeffs)
        for blade, coeff in other.coeffs.items():
            coeffs[blade] = coeffs.get(blade, 0) + coeff
        return Multivector(self.algebra, coeffs)

    def __neg__(self) -> "Multivector":
        return Multivector(self.algebra, {blade: -coeff for blade, coeff in self.coeffs.items()})

    def __sub__(self, other: "Multivector") -> "Multivector":
        return self + (-other)

    def __mul__(self, other: Any) -> "Multivector":
        clean = normalize_simplifier(self.algebra.simplify)
        if not isinstance(other, Multivector):
            return Multivector(self.algebra, {blade: coeff * sp.sympify(other) for blade, coeff in self.coeffs.items()})
        self._check_same_algebra(other)
        coeffs: dict[Blade, Any] = {}
        for left_blade, left_coeff in self.coeffs.items():
            for right_blade, right_coeff in other.coeffs.items():
                for blade, metric_coeff in self.algebra.blade_product(left_blade, right_blade):
                    coeffs[blade] = coeffs.get(blade, 0) + left_coeff * right_coeff * metric_coeff
        return Multivector(self.algebra, {blade: clean(coeff) for blade, coeff in coeffs.items()})

    def __rmul__(self, other: Any) -> "Multivector":
        scalar = sp.sympify(other)
        if isinstance(scalar, sp.Basic) and scalar.is_commutative is False:
            return NotImplemented
        return Multivector(self.algebra, {blade: scalar * coeff for blade, coeff in self.coeffs.items()})

    def grades(self) -> set[int]:
        return {len(blade) for blade in self.coeffs}

    def grade(self, grade: int) -> "Multivector":
        return Multivector(self.algebra, {blade: coeff for blade, coeff in self.coeffs.items() if len(blade) == grade})

    grade_part = grade

    def scalar_part(self) -> Any:
        return self.coeffs.get((), sp.Integer(0))

    def _grade_selected_product(self, other: "Multivector", grade_rule) -> "Multivector":
        self._check_same_algebra(other)
        clean = normalize_simplifier(self.algebra.simplify)
        coeffs: dict[Blade, Any] = {}
        for left_blade, left_coeff in self.coeffs.items():
            for right_blade, right_coeff in other.coeffs.items():
                target_grade = grade_rule(len(left_blade), len(right_blade))
                if target_grade is None:
                    continue
                for blade, metric_coeff in self.algebra.blade_product(left_blade, right_blade):
                    if len(blade) == target_grade:
                        coeffs[blade] = coeffs.get(blade, 0) + left_coeff * right_coeff * metric_coeff
        return Multivector(self.algebra, {blade: clean(coeff) for blade, coeff in coeffs.items()})

    def exterior(self, other: "Multivector") -> "Multivector":
        return self._grade_selected_product(other, lambda left_grade, right_grade: left_grade + right_grade)

    wedge = exterior

    def inner(self, other: "Multivector") -> "Multivector":
        return self._grade_selected_product(other, lambda left_grade, right_grade: abs(right_grade - left_grade))

    dot = inner

    def left_contraction(self, other: "Multivector") -> "Multivector":
        return self._grade_selected_product(
            other,
            lambda left_grade, right_grade: None if left_grade > right_grade else right_grade - left_grade,
        )

    lcontract = left_contraction

    def reverse(self) -> "Multivector":
        return Multivector(self.algebra, {blade: ((-1) ** (len(blade) * (len(blade) - 1) // 2)) * coeff for blade, coeff in self.coeffs.items()})

    rev = reverse

    def grade_involution(self) -> "Multivector":
        return Multivector(self.algebra, {blade: ((-1) ** len(blade)) * coeff for blade, coeff in self.coeffs.items()})

    def clifford_conjugate(self) -> "Multivector":
        return self.grade_involution().reverse()

    def reverse_product(self) -> "Multivector":
        """Return ``A * reverse(A)`` without discarding nonscalar grades."""
        return self * self.reverse()

    def norm_squared(self, *, require_scalar: bool = False) -> Any:
        """Return the scalar part of ``A * reverse(A)``.

        Set ``require_scalar=True`` to reject multivectors whose reverse product
        has nonscalar grades.  This is the condition needed by the simple
        reverse-over-norm inverse path.
        """
        product = self.reverse_product()
        if require_scalar:
            nonscalar = {blade: coeff for blade, coeff in product.coeffs.items() if blade != () and coeff != 0}
            if nonscalar:
                raise UnsupportedGeometryError("A * reverse(A) is not scalar for this multivector")
        return product.scalar_part()

    def inverse(self) -> "Multivector":
        """Return the reverse-over-norm inverse when that formula applies."""
        norm = sp.simplify(self.norm_squared(require_scalar=True))
        if norm == 0:
            raise ZeroDivisionError("multivector has zero norm-squared in this inverse path")
        return self.reverse() * (1 / norm)

    versor_inverse = inverse

    def commutator(self, other: "Multivector") -> "Multivector":
        return (self * other - other * self) * sp.Rational(1, 2)

    def anticommutator(self, other: "Multivector") -> "Multivector":
        return (self * other + other * self) * sp.Rational(1, 2)

    def dual(self) -> "Multivector":
        pseudoscalar = self.algebra.blade(range(self.algebra.dimension))
        return self * pseudoscalar.inverse()

    def is_zero(self) -> bool:
        return not self.coeffs

    def __repr__(self) -> str:
        if not self.coeffs:
            return "0"
        pieces = []
        for blade, coeff in sorted(self.coeffs.items(), key=lambda item: (len(item[0]), item[0])):
            name = "1" if blade == () else "^".join(self.algebra.basis_names[i] for i in blade)
            pieces.append(f"{coeff}*{name}")
        return " + ".join(pieces)
