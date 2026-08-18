"""Cartan, connection, curvature, and gauge tensor-valued form helpers.

The Cartan convention used here is T^a = d theta^a + omega^a{}_b theta^b
and Omega^a{}_b = d omega^a{}_b + omega^a{}_c omega^c{}_b.  Products are
formal exterior products represented symbolically in component expressions.
"""

from __future__ import annotations

from typing import Any

import sympy as sp

from .valued import TensorValuedForm, all_component_labels, exterior_derivative_tvform, form_simplifier


def solder_form(labels: tuple[Any, ...], *, simplify: bool | Any = True) -> TensorValuedForm:
    return TensorValuedForm(1, (1,), {(label,): sp.Symbol(f"theta^{label}") for label in labels}, label="theta", simplify=simplify)


def connection_form(labels: tuple[Any, ...], *, name: str = "omega", simplify: bool | Any = True) -> TensorValuedForm:
    return TensorValuedForm(1, (1, -1), {(a, b): sp.Symbol(f"{name}^{a}_{b}") for a in labels for b in labels}, label=name, simplify=simplify)


def compose_endomorphism_forms(left: TensorValuedForm, right: TensorValuedForm) -> TensorValuedForm:
    if left.variance != (1, -1) or right.variance != (1, -1):
        raise ValueError("endomorphism composition expects variance (+1, -1)")
    clean = form_simplifier(left.simplify)
    labels = all_component_labels(left, right)
    components: dict[tuple[Any, ...], Any] = {}
    for target in labels:
        for source in labels:
            value = sum(left.components.get((target, middle), 0) * right.components.get((middle, source), 0) for middle in labels)
            if value != 0:
                components[(target, source)] = clean(value)
    return TensorValuedForm(left.degree + right.degree, (1, -1), components, simplify=left.simplify)


def curvature_form(omega: TensorValuedForm) -> TensorValuedForm:
    if omega.variance != (1, -1) or omega.degree != 1:
        raise ValueError("curvature_form expects a connection 1-form with variance (+1, -1)")
    return exterior_derivative_tvform(omega) + compose_endomorphism_forms(omega, omega)


def torsion_form(theta: TensorValuedForm, omega: TensorValuedForm) -> TensorValuedForm:
    if theta.variance != (1,) or theta.degree != 1:
        raise ValueError("theta must be a vector-valued 1-form")
    if omega.variance != (1, -1) or omega.degree != 1:
        raise ValueError("omega must be an endomorphism-valued 1-form")
    clean = form_simplifier(theta.simplify)
    labels = all_component_labels(theta, omega)
    dtheta = exterior_derivative_tvform(theta)
    components = dict(dtheta.components)
    for target in labels:
        value = components.get((target,), 0)
        for source in labels:
            value += omega.components.get((target, source), 0) * theta.components.get((source,), 0)
        if value != 0:
            components[(target,)] = clean(value)
    return TensorValuedForm(2, (1,), components, label="T", simplify=theta.simplify)


def cartan_first_equation(theta: TensorValuedForm, omega: TensorValuedForm) -> TensorValuedForm:
    return torsion_form(theta, omega)


def cartan_second_equation(omega: TensorValuedForm) -> TensorValuedForm:
    return curvature_form(omega)


def gauge_curvature(gauge_potential: TensorValuedForm) -> TensorValuedForm:
    return curvature_form(gauge_potential)
