from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import sympy as sp

from .abstract_tensor import (
    invariant_basis_catalog,
    reduce_to_invariant_basis,
    classify_differential_invariants,
)
from .basis import (
    TensorBasis,
    frame_basis,
    frame_metric,
    frame_structure_coefficients,
    frame_connection_coefficients,
    curvature_two_forms,
    torsion_two_forms,
    frame_to_chart_matrix,
    chart_to_frame_matrix,
)
from .charts import CoordinateChart


@dataclass(frozen=True)
class VariationResult:
    lagrangian: sp.Expr
    field: sp.Expr
    coordinates: tuple[sp.Symbol, ...]
    euler_lagrange: sp.Expr
    jet_orders: tuple[tuple[sp.Symbol, ...], ...] = tuple()
    integration_by_parts_steps: tuple[str, ...] = tuple()
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PerturbationExpansionResult:
    original_expr: sp.Expr
    expanded_expr: sp.Expr
    parameter: sp.Symbol
    order: int
    substitutions: tuple[tuple[sp.Expr, sp.Expr, sp.Expr], ...] = tuple()
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OperatorBasisClassification:
    original_exprs: tuple[object, ...]
    reduced_exprs: tuple[object, ...]
    basis_signatures: tuple[str, ...]
    bucket_keys: tuple[tuple[int, int], ...]
    term_signatures: tuple[tuple[str, ...], ...]
    coefficient_maps: tuple[Mapping[str, object], ...] = tuple()
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ComponentFrameWorkflowReport:
    chart_name: str
    coordinates: tuple[sp.Symbol, ...]
    frame_name: str
    orthonormal: bool
    chart_metric: sp.Matrix
    frame_to_chart: sp.Matrix
    chart_to_frame: sp.Matrix
    frame_metric: sp.Matrix
    structure_coefficients: Any
    connection_coefficients: Any
    curvature_forms: Any
    torsion_forms: Any
    provenance: Mapping[str, Any] = field(default_factory=dict)


def _as_expr(expr: object) -> sp.Expr:
    return sp.sympify(expr)


def _field_jet_specs(lagrangian: sp.Expr, field: sp.Expr) -> tuple[tuple[sp.Symbol, ...], ...]:
    specs: set[tuple[sp.Symbol, ...]] = {tuple()}
    for deriv in lagrangian.atoms(sp.Derivative):
        if deriv.expr == field:
            specs.add(tuple(deriv.variables))
    return tuple(sorted(specs, key=lambda item: (len(item), tuple(str(v) for v in item))))


def total_derivative(expr: object, variables: Sequence[sp.Symbol]) -> sp.Expr:
    out = _as_expr(expr)
    for var in variables:
        out = sp.diff(out, var)
    return sp.simplify(sp.expand(out))


def variational_derivative(lagrangian: object, field: sp.Expr, coordinates: Sequence[sp.Symbol]) -> sp.Expr:
    L = _as_expr(lagrangian)
    coords = tuple(coordinates)
    total = sp.Integer(0)
    for spec in _field_jet_specs(L, field):
        jet = field if not spec else sp.Derivative(field, *spec)
        partial = sp.diff(L, jet)
        if partial == 0:
            continue
        term = partial
        for var in spec:
            term = -sp.diff(term, var)
        total += term
    return sp.simplify(sp.expand(total))


def integration_by_parts_reduce_scalar(expr: object, field: sp.Expr, coordinates: Sequence[sp.Symbol] | None = None) -> tuple[sp.Expr, tuple[str, ...]]:
    expanded = sp.expand(_as_expr(expr))
    terms = []
    steps: list[str] = []
    coords = tuple(coordinates or ())
    for term in sp.Add.make_args(expanded):
        coeff, rest = term.as_coeff_Mul()
        factors = list(sp.Mul.make_args(rest))
        replaced = False
        for i, fac in enumerate(factors):
            if isinstance(fac, sp.Derivative) and fac.expr == field:
                other = sp.Mul(*factors[:i], *factors[i+1:]) if len(factors) > 1 else sp.Integer(1)
                if coords and not set(fac.variables).issubset(set(coords)):
                    continue
                moved = (-1) ** len(fac.variables) * field * total_derivative(other, fac.variables)
                terms.append(coeff * sp.expand(moved))
                steps.append(f"ibp:{','.join(str(v) for v in fac.variables)}")
                replaced = True
                break
        if not replaced:
            terms.append(term)
    return sp.simplify(sp.expand(sum(terms, sp.Integer(0)))), tuple(steps)


def variational_problem(lagrangian: object, field: sp.Expr, coordinates: Sequence[sp.Symbol]) -> VariationResult:
    L = _as_expr(lagrangian)
    coords = tuple(coordinates)
    euler = variational_derivative(L, field, coords)
    ibp_reduced, ibp_steps = integration_by_parts_reduce_scalar(L, field, coords)
    return VariationResult(
        lagrangian=L,
        field=field,
        coordinates=coords,
        euler_lagrange=euler,
        jet_orders=_field_jet_specs(L, field),
        integration_by_parts_steps=ibp_steps,
        provenance={
            "ibp_reduced_lagrangian": ibp_reduced,
            "jet_order_count": len(_field_jet_specs(L, field)),
        },
    )


def perturbation_expand(
    expr: object,
    background_map: Mapping[sp.Expr, object],
    perturbation_map: Mapping[sp.Expr, object],
    *,
    parameter: sp.Symbol | None = None,
    order: int = 1,
) -> PerturbationExpansionResult:
    parameter = sp.Symbol('eps') if parameter is None else parameter
    current = _as_expr(expr)
    substitutions = []
    for key, bg in background_map.items():
        if key not in perturbation_map:
            raise KeyError(f"Missing perturbation entry for {key!r}")
        bg_expr = _as_expr(bg)
        pert_expr = _as_expr(perturbation_map[key])
        substitutions.append((sp.sympify(key), bg_expr, pert_expr))
        current = current.subs(key, bg_expr + parameter * pert_expr)
    expanded = sp.expand(current)
    try:
        expanded = sp.series(expanded, parameter, 0, order + 1).removeO()
    except Exception:
        expanded = sp.expand(expanded)
        expanded = sp.Poly(expanded, parameter).truncate(order + 1).as_expr() if expanded != 0 else expanded
    expanded = sp.expand(expanded)
    return PerturbationExpansionResult(
        original_expr=_as_expr(expr),
        expanded_expr=expanded,
        parameter=parameter,
        order=order,
        substitutions=tuple(substitutions),
        provenance={
            "background_keys": tuple(str(k) for k in background_map),
            "term_count": len(sp.Add.make_args(expanded)),
        },
    )


def classify_operator_basis(
    exprs: object | Iterable[object],
    *,
    dimension: int | sp.Expr | None = None,
    integration_by_parts: bool = False,
) -> OperatorBasisClassification:
    items = tuple(exprs if isinstance(exprs, Iterable) and not isinstance(exprs, (sp.Basic, str, bytes)) else (exprs,))
    reduced_exprs = []
    term_signatures = []
    coefficient_maps = []
    all_sigs: set[str] = set()
    bucket_keys: set[tuple[int, int]] = set()
    for expr in items:
        catalog = invariant_basis_catalog(expr, dimension=dimension)
        bucket_keys.update(catalog.by_order_and_derivative.keys())
        reduced, report = reduce_to_invariant_basis(expr, catalog, dimension=dimension, integration_by_parts=integration_by_parts, with_report=True)
        reduced_exprs.append(reduced.expr if hasattr(reduced, 'expr') else reduced)
        term_desc = classify_differential_invariants(expr, dimension=dimension)
        sigs = tuple(d.signature for d in term_desc)
        term_signatures.append(sigs)
        all_sigs.update(sigs)
        coefficient_maps.append(dict(report.coefficient_map))
        all_sigs.update(report.coefficient_map.keys())
    return OperatorBasisClassification(
        original_exprs=tuple(items),
        reduced_exprs=tuple(reduced_exprs),
        basis_signatures=tuple(sorted((s for s in all_sigs if s is not None), key=repr)),
        bucket_keys=tuple(sorted(bucket_keys)),
        term_signatures=tuple(term_signatures),
        coefficient_maps=tuple(coefficient_maps),
        provenance={
            "dimension": dimension,
            "integration_by_parts": integration_by_parts,
            "expression_count": len(items),
        },
    )


def component_frame_workflow(chart: CoordinateChart, frame: TensorBasis, coords: Sequence[sp.Symbol] | None = None) -> ComponentFrameWorkflowReport:
    if frame.chart != chart:
        raise ValueError("Frame must be attached to the supplied chart.")
    actual_coords = tuple(coords or chart.symbols())
    working_frame = frame
    if frame.metadata.get('transform_to_chart') is None:
        working_frame = frame_basis(frame.name, chart, lambda c: sp.eye(chart.dimension), orthonormal=frame.kind.startswith('orthonormal'))
    return ComponentFrameWorkflowReport(
        chart_name=chart.chart_name,
        coordinates=actual_coords,
        frame_name=working_frame.name,
        orthonormal=working_frame.kind.startswith('orthonormal'),
        chart_metric=chart.metric(actual_coords),
        frame_to_chart=frame_to_chart_matrix(working_frame, actual_coords),
        chart_to_frame=chart_to_frame_matrix(working_frame, actual_coords),
        frame_metric=frame_metric(working_frame, actual_coords),
        structure_coefficients=frame_structure_coefficients(working_frame, actual_coords),
        connection_coefficients=frame_connection_coefficients(working_frame, actual_coords),
        curvature_forms=curvature_two_forms(working_frame, None, actual_coords),
        torsion_forms=torsion_two_forms(working_frame, None, actual_coords),
        provenance={
            "chart_dimension": chart.dimension,
            "metric_name": chart.metric_name,
            "used_identity_frame": working_frame is not frame,
        },
    )


__all__ = [
    'VariationResult',
    'PerturbationExpansionResult',
    'OperatorBasisClassification',
    'ComponentFrameWorkflowReport',
    'total_derivative',
    'variational_derivative',
    'integration_by_parts_reduce_scalar',
    'variational_problem',
    'perturbation_expand',
    'classify_operator_basis',
    'component_frame_workflow',
]
