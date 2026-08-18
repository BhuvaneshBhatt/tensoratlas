from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import sympy as sp

from .charts import CoordinateChart
from .variational_workflows import variational_derivative
from .exterior_geometry import ExteriorFormNF


@dataclass(frozen=True)
class DensityDef:
    name: str
    chart: CoordinateChart | None
    weight: sp.Expr
    expression: sp.Expr
    coordinates: tuple[sp.Symbol, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CovariantVariationResult:
    lagrangian: sp.Expr
    density: DensityDef
    weighted_lagrangian: sp.Expr
    field: sp.Expr
    coordinates: tuple[sp.Symbol, ...]
    density_weighted_euler: sp.Expr
    covariant_euler: sp.Expr
    perturbation_parameter: sp.Symbol | None = None
    perturbation_order: int | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PerturbedMetricGeometry:
    coordinates: tuple[sp.Symbol, ...]
    parameter: sp.Symbol
    order: int
    background_metric: sp.Matrix
    perturbation_metric: sp.Matrix
    expanded_metric: sp.Matrix
    inverse_metric: sp.Matrix
    determinant: sp.Expr
    volume_density: sp.Expr
    christoffel_symbols: tuple[tuple[tuple[sp.Expr, ...], ...], ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HypersurfaceGeometryReport:
    chart_name: str
    fixed_coordinate: sp.Symbol
    fixed_index: int
    level: sp.Expr
    induced_coordinates: tuple[sp.Symbol, ...]
    induced_metric: sp.Matrix
    normal_covector: tuple[sp.Expr, ...]
    normal_vector: tuple[sp.Expr, ...]
    extrinsic_curvature: sp.Matrix
    mean_curvature: sp.Expr
    volume_density: sp.Expr
    volume_form: ExteriorFormNF
    provenance: Mapping[str, Any] = field(default_factory=dict)



def _coords(chart: CoordinateChart, coordinates: Sequence[sp.Symbol] | None = None) -> tuple[sp.Symbol, ...]:
    return tuple(coordinates) if coordinates is not None else tuple(chart.symbols())



def _refine_with_chart_assumptions(chart: CoordinateChart, expr: Any, coords: tuple[sp.Symbol, ...]) -> sp.Expr:
    out = sp.sympify(expr)
    assumptions = chart.domain_assumptions(coords)
    if assumptions is not None:
        try:
            out = sp.refine(out, assumptions)
        except Exception:
            pass
        for rel in assumptions.atoms(sp.StrictGreaterThan, sp.GreaterThan, sp.StrictLessThan, sp.LessThan):
            lhs = getattr(rel, 'lhs', None)
            rhs = getattr(rel, 'rhs', None)
            if lhs not in coords or rhs != 0:
                continue
            if rel.rel_op in {'>', '>='}:
                out = out.xreplace({sp.Abs(lhs): lhs})
            elif rel.rel_op in {'<', '<='}:
                out = out.xreplace({sp.Abs(lhs): -lhs})
    return sp.simplify(out)


def metric_density(
    chart: CoordinateChart,
    *,
    coordinates: Sequence[sp.Symbol] | None = None,
    weight: Any = 1,
    expression: Any | None = None,
    name: str | None = None,
) -> DensityDef:
    coords = _coords(chart, coordinates)
    base = _refine_with_chart_assumptions(chart, chart.volume_density(coords), coords)
    expr = sp.sympify(expression) if expression is not None else base ** sp.sympify(weight)
    expr = _refine_with_chart_assumptions(chart, expr, coords)
    return DensityDef(
        name=name or f"density({chart.chart_name})",
        chart=chart,
        weight=sp.sympify(weight),
        expression=sp.simplify(expr),
        coordinates=coords,
        metadata={"base_density": base},
    )



def metric_volume_form(chart: CoordinateChart, coordinates: Sequence[sp.Symbol] | None = None) -> ExteriorFormNF:
    coords = _coords(chart, coordinates)
    coeff = _refine_with_chart_assumptions(chart, chart.volume_density(coords), coords)
    labels = tuple(str(c) for c in coords)
    blade = tuple(range(chart.dimension))
    return ExteriorFormNF(chart.dimension, {blade: sp.simplify(coeff)}, basis_labels=labels, metadata={"chart": chart.chart_name, "kind": "volume_form"})



def covariant_variational_problem(
    lagrangian: Any,
    field: sp.Expr,
    chart: CoordinateChart,
    *,
    coordinates: Sequence[sp.Symbol] | None = None,
    density: DensityDef | None = None,
    perturbation_parameter: sp.Symbol | None = None,
    perturbation_order: int | None = None,
) -> CovariantVariationResult:
    coords = _coords(chart, coordinates)
    L = sp.sympify(lagrangian)
    dens = density or metric_density(chart, coordinates=coords)
    weighted = sp.expand(sp.sympify(dens.expression) * L)
    weighted_euler = sp.simplify(sp.expand(variational_derivative(weighted, field, coords)))
    if sp.simplify(dens.expression) == 0:
        covariant = weighted_euler
    else:
        covariant = sp.simplify(sp.expand(weighted_euler / dens.expression))
    return CovariantVariationResult(
        lagrangian=L,
        density=dens,
        weighted_lagrangian=weighted,
        field=field,
        coordinates=coords,
        density_weighted_euler=weighted_euler,
        covariant_euler=covariant,
        perturbation_parameter=perturbation_parameter,
        perturbation_order=perturbation_order,
        provenance={
            "chart": chart.chart_name,
            "metric_name": chart.metric_name,
            "used_default_density": density is None,
        },
    )



def _truncate_series(expr: sp.Expr, parameter: sp.Symbol, order: int) -> sp.Expr:
    expr = sp.expand(sp.sympify(expr))
    try:
        return sp.expand(sp.series(expr, parameter, 0, order + 1).removeO())
    except Exception:
        try:
            poly = sp.Poly(expr, parameter)
            total = sp.Integer(0)
            for power, coeff in poly.terms():
                p = power[0] if isinstance(power, tuple) else power
                if p <= order:
                    total += coeff * parameter ** p
            return sp.expand(total)
        except Exception:
            return expr



def perturb_metric_geometry(
    chart: CoordinateChart,
    perturbation_metric: Any,
    *,
    coordinates: Sequence[sp.Symbol] | None = None,
    parameter: sp.Symbol | None = None,
    order: int = 1,
) -> PerturbedMetricGeometry:
    coords = _coords(chart, coordinates)
    eps = sp.Symbol("eps") if parameter is None else parameter
    g0 = sp.Matrix(chart.metric(coords))
    h = sp.Matrix(perturbation_metric)
    if h.shape != g0.shape:
        raise ValueError("perturbation_metric must have the same shape as the chart metric")
    g = g0 + eps * h
    ginv0 = sp.Matrix(chart.inverse_metric(coords))
    if order < 1:
        ginv = ginv0
    else:
        # Neumann expansion to first order in eps.
        ginv = ginv0 - eps * (ginv0 * h * ginv0)
    ginv = ginv.applyfunc(lambda e: _truncate_series(e, eps, order))
    detg = _truncate_series(sp.expand(g.det()), eps, order)
    vol = _truncate_series(sp.expand(sp.sqrt(sp.Abs(detg))), eps, order)
    dim = chart.dimension
    gamma = []
    for a in range(dim):
        row = []
        for b in range(dim):
            col = []
            for c in range(dim):
                val = sp.Integer(0)
                for m in range(dim):
                    val += sp.Rational(1, 2) * ginv[a, m] * (
                        sp.diff(g[m, c], coords[b])
                        + sp.diff(g[m, b], coords[c])
                        - sp.diff(g[b, c], coords[m])
                    )
                col.append(_truncate_series(sp.simplify(sp.expand(val)), eps, order))
            row.append(tuple(col))
        gamma.append(tuple(row))
    return PerturbedMetricGeometry(
        coordinates=coords,
        parameter=eps,
        order=order,
        background_metric=g0,
        perturbation_metric=h,
        expanded_metric=g.applyfunc(lambda e: _truncate_series(e, eps, order)),
        inverse_metric=ginv,
        determinant=detg,
        volume_density=vol,
        christoffel_symbols=tuple(gamma),
        provenance={"chart": chart.chart_name, "metric_name": chart.metric_name},
    )



def coordinate_hypersurface_geometry(
    chart: CoordinateChart,
    fixed_coordinate: int | sp.Symbol,
    *,
    level: Any = 0,
    coordinates: Sequence[sp.Symbol] | None = None,
) -> HypersurfaceGeometryReport:
    coords = _coords(chart, coordinates)
    if isinstance(fixed_coordinate, int):
        fixed_index = int(fixed_coordinate)
    else:
        try:
            fixed_index = coords.index(sp.sympify(fixed_coordinate))
        except ValueError as exc:
            raise ValueError("fixed_coordinate must be a valid coordinate or index") from exc
    g = sp.Matrix(chart.metric(coords))
    dim = chart.dimension
    # Conservative current scope: coordinate-orthogonal hypersurfaces.
    for i in range(dim):
        if i == fixed_index:
            continue
        if sp.simplify(g[fixed_index, i]) != 0 or sp.simplify(g[i, fixed_index]) != 0:
            raise NotImplementedError("coordinate_hypersurface_geometry currently requires a coordinate-orthogonal hypersurface")
    induced_indices = tuple(i for i in range(dim) if i != fixed_index)
    induced_coords = tuple(coords[i] for i in induced_indices)
    h = g.extract(induced_indices, induced_indices)
    gnn = sp.simplify(g[fixed_index, fixed_index])
    normal_covector = tuple(sp.Integer(0) if i != fixed_index else sp.simplify(sp.sqrt(sp.Abs(gnn))) for i in range(dim))
    normal_vector = tuple(sp.Integer(0) if i != fixed_index else sp.simplify(1 / sp.sqrt(sp.Abs(gnn))) for i in range(dim))
    n_contra = normal_vector[fixed_index]
    K = sp.Matrix(
        len(induced_indices),
        len(induced_indices),
        lambda a, b: sp.simplify(sp.Rational(1, 2) * n_contra * sp.diff(h[a, b], coords[fixed_index])),
    )
    h_inv = sp.simplify(h.inv())
    mean = sp.simplify(sum(h_inv[i, j] * K[i, j] for i in range(h.rows) for j in range(h.cols))) if h.rows else sp.Integer(0)
    level_subs = {coords[fixed_index]: sp.sympify(level)}
    h_level = h.applyfunc(lambda e: sp.simplify(e.subs(level_subs)))
    K_level = K.applyfunc(lambda e: sp.simplify(e.subs(level_subs)))
    det_h = sp.simplify(h_level.det()) if h_level.rows else sp.Integer(1)
    vol = sp.simplify(sp.sqrt(sp.Abs(det_h)))
    blade = tuple(range(len(induced_indices)))
    vol_form = ExteriorFormNF(len(induced_indices), {blade: vol}, basis_labels=tuple(str(c) for c in induced_coords), metadata={"chart": chart.chart_name, "kind": "hypersurface_volume_form"})
    return HypersurfaceGeometryReport(
        chart_name=chart.chart_name,
        fixed_coordinate=coords[fixed_index],
        fixed_index=fixed_index,
        level=sp.sympify(level),
        induced_coordinates=induced_coords,
        induced_metric=h_level,
        normal_covector=tuple(sp.simplify(v.subs(level_subs)) for v in normal_covector),
        normal_vector=tuple(sp.simplify(v.subs(level_subs)) for v in normal_vector),
        extrinsic_curvature=K_level,
        mean_curvature=sp.simplify(mean.subs(level_subs)),
        volume_density=vol,
        volume_form=vol_form,
        provenance={"metric_name": chart.metric_name, "orthogonal_coordinate_hypersurface": True},
    )


__all__ = [
    "DensityDef",
    "CovariantVariationResult",
    "PerturbedMetricGeometry",
    "HypersurfaceGeometryReport",
    "metric_density",
    "metric_volume_form",
    "covariant_variational_problem",
    "perturb_metric_geometry",
    "coordinate_hypersurface_geometry",
]
