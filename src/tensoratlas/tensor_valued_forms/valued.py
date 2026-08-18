"""Formal tensor-valued differential forms.

A tensor-valued form has an exterior degree and a finite tensor-value
variance.  Component keys index only the tensor-value slots; the exterior
basis element is represented by the component expression itself.  This keeps
Cartan and gauge identities inspectable without committing to a coordinate
basis for the exterior algebra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import sympy as sp
from tensoratlas.errors import FormDegreeError
from tensoratlas.simplification_policy import normalize_simplifier
from tensoratlas.validation import ValidationReport, invalid_report, valid_report


def form_simplifier(simplify: bool | Any):
    return normalize_simplifier(simplify)


@dataclass(frozen=True)
class TensorValuedForm:
    """Formal tensor-valued differential form.

    Parameters
    ----------
    degree:
        Exterior degree of the form.
    variance:
        Tensor-value slot variance, with ``+1`` contravariant and ``-1``
        covariant.  Component keys must have the same length as ``variance``.
    components:
        Mapping from tensor-value index keys to formal form coefficients.
    """

    degree: int
    variance: tuple[int, ...]
    components: Mapping[tuple[Any, ...], Any]
    label: str | None = None
    simplify: bool | Any = True

    def __post_init__(self) -> None:
        if self.degree < 0:
            raise FormDegreeError("form degree must be nonnegative")
        if any(slot not in (-1, 1) for slot in self.variance):
            raise FormDegreeError("variance entries must be +1 or -1")
        clean = form_simplifier(self.simplify)
        normalized = {}
        for key, value in self.components.items():
            key = tuple(key)
            if len(key) != len(self.variance):
                raise FormDegreeError("component key rank must equal variance rank")
            expr = clean(sp.sympify(value))
            if expr != 0:
                normalized[key] = expr
        object.__setattr__(self, "components", normalized)

    @property
    def rank(self) -> int:
        return len(self.variance)

    def validation_report(self) -> ValidationReport:
        """Return structured validation diagnostics without raising."""
        errors = []
        if self.degree < 0:
            errors.append("form degree must be nonnegative")
        if any(slot not in (-1, 1) for slot in self.variance):
            errors.append("variance entries must be +1 or -1")
        for key in self.components:
            if len(key) != len(self.variance):
                errors.append("component key rank must equal variance rank")
                break
        return invalid_report(*errors) if errors else valid_report()

    def validate(self) -> bool:
        """Validate degree, variance, and component key ranks."""
        report = self.validation_report()
        if not report.ok:
            raise FormDegreeError("; ".join(report.errors))
        return True

    def summary(self) -> dict[str, Any]:
        """Return degree, tensor-value rank, variance, label, and nonzero count."""
        return {
            "degree": self.degree,
            "rank": self.rank,
            "variance": self.variance,
            "label": self.label,
            "nonzero_components": len(self.components),
            "component_keys": tuple(self.components),
        }

    def wedge(self, other: "TensorValuedForm") -> "TensorValuedForm":
        """Method form of :func:`wedge_tensor_valued_forms`."""
        return wedge_tensor_valued_forms(self, other)

    def exterior_derivative(self, *, mode: str = "formal") -> "TensorValuedForm":
        """Method form of :func:`exterior_derivative_tvform`."""
        return exterior_derivative_tvform(self, mode=mode)

    def _check_compatible(self, other: "TensorValuedForm") -> None:
        if self.degree != other.degree or self.variance != other.variance:
            raise FormDegreeError("tensor-valued forms must have same degree and variance")

    def __add__(self, other: "TensorValuedForm") -> "TensorValuedForm":
        if not isinstance(other, TensorValuedForm):
            return NotImplemented
        self._check_compatible(other)
        clean = form_simplifier(self.simplify)
        keys = set(self.components) | set(other.components)
        return TensorValuedForm(
            self.degree,
            self.variance,
            {key: clean(self.components.get(key, 0) + other.components.get(key, 0)) for key in keys},
            simplify=self.simplify,
        )

    def __neg__(self) -> "TensorValuedForm":
        return TensorValuedForm(
            self.degree,
            self.variance,
            {key: -value for key, value in self.components.items()},
            label=self.label,
            simplify=self.simplify,
        )

    def __sub__(self, other: "TensorValuedForm") -> "TensorValuedForm":
        if not isinstance(other, TensorValuedForm):
            return NotImplemented
        return self + (-other)

    def scale(self, scalar: Any) -> "TensorValuedForm":
        clean = form_simplifier(self.simplify)
        factor = sp.sympify(scalar)
        return TensorValuedForm(
            self.degree,
            self.variance,
            {key: clean(factor * value) for key, value in self.components.items()},
            label=self.label,
            simplify=self.simplify,
        )


def exterior_derivative_tvform(form: TensorValuedForm, *, mode: str = "formal") -> TensorValuedForm:
    """Return a formal exterior derivative of a tensor-valued form.

    Coordinate exterior differentiation is intentionally separate from this
    formal constructor; passing any mode other than ``"formal"`` raises a clear
    error rather than silently implying a coordinate computation.
    """
    if mode != "formal":
        raise NotImplementedError("coordinate exterior derivatives for tensor-valued forms are not implemented yet")
    return TensorValuedForm(
        form.degree + 1,
        form.variance,
        {key: sp.Function("d")(value) for key, value in form.components.items()},
        label=f"d{form.label or ''}",
        simplify=form.simplify,
    )


def wedge_tensor_valued_forms(left: TensorValuedForm, right: TensorValuedForm) -> TensorValuedForm:
    clean = form_simplifier(left.simplify)
    variance = left.variance + right.variance
    degree = left.degree + right.degree
    components: dict[tuple[Any, ...], Any] = {}
    for left_key, left_value in left.components.items():
        for right_key, right_value in right.components.items():
            key = tuple(left_key) + tuple(right_key)
            components[key] = clean(components.get(key, 0) + left_value * right_value)
    return TensorValuedForm(degree, variance, components, simplify=left.simplify)


def all_component_labels(*forms: TensorValuedForm) -> tuple[Any, ...]:
    labels = set()
    for form in forms:
        for key in form.components:
            labels.update(key)
    return tuple(sorted(labels, key=repr))
