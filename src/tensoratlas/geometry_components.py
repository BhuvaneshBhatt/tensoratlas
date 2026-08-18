
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional
import sympy as sp
from .basis import basis_transformation_matrix
from .charts import CoordinateChart
from .exterior_geometry import ExteriorFormNF, canonicalize_exterior_form, exterior_derivative_nf

@dataclass(frozen=True)
class ComponentTensorField:
    name: str
    chart: CoordinateChart
    variance_spec: str
    components: sp.MutableDenseNDimArray
    basis_kind: str = "coordinate"
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class BasisFrameTransformReport:
    source_basis: str
    target_basis: str
    transform_matrix: sp.Matrix
    inverse_matrix: sp.Matrix
    determinant: sp.Expr
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ComponentGeometryReport:
    chart_name: str
    metric_matrix: sp.Matrix
    inverse_metric: sp.Matrix
    determinant: sp.Expr
    volume_density: sp.Expr
    christoffel_symbols: tuple
    riemann_tensor: tuple | None = None
    ricci_tensor: sp.Matrix | None = None
    scalar_curvature: sp.Expr | None = None

@dataclass(frozen=True)
class MetricHodgeReport:
    input_degree: int
    output_degree: int
    metric_determinant: sp.Expr
    signature_parity: int
    result: ExteriorFormNF

def component_tensor_field(name: str, chart: CoordinateChart, variance_spec: str, components: Any, *, basis_kind: str = "coordinate", metadata: Optional[Mapping[str, Any]] = None) -> ComponentTensorField:
    return ComponentTensorField(name=name, chart=chart, variance_spec=variance_spec, components=sp.MutableDenseNDimArray(components), basis_kind=basis_kind, metadata=dict(metadata or {}))

def basis_frame_transform_report(source_basis, target_basis, coords=None) -> BasisFrameTransformReport:
    mat = basis_transformation_matrix(source_basis, target_basis, coords)
    inv = basis_transformation_matrix(target_basis, source_basis, coords)
    return BasisFrameTransformReport(
        source_basis=getattr(source_basis, "name", "source"),
        target_basis=getattr(target_basis, "name", "target"),
        transform_matrix=mat,
        inverse_matrix=inv,
        determinant=sp.simplify(mat.det()),
        metadata={"coords": tuple(coords) if coords is not None else None},
    )

def _metric_matrix(chart: CoordinateChart) -> sp.Matrix:
    metric = getattr(chart, "metric", None)
    if callable(metric):
        metric = metric()
    if metric is None:
        return sp.eye(chart.dimension)
    return sp.Matrix(metric)

def _christoffel_from_metric(chart: CoordinateChart, metric: sp.Matrix):
    coords = tuple(chart.symbols())
    dim = chart.dimension
    ginv = sp.simplify(metric.inv())
    Gamma = [[[sp.Integer(0) for _ in range(dim)] for _ in range(dim)] for _ in range(dim)]
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                expr = sp.Integer(0)
                for l in range(dim):
                    expr += ginv[i, l] * (sp.diff(metric[l, j], coords[k]) + sp.diff(metric[l, k], coords[j]) - sp.diff(metric[j, k], coords[l]))
                Gamma[i][j][k] = sp.simplify(sp.Rational(1, 2) * expr)
    return tuple(tuple(tuple(Gamma[i][j][k] for k in range(dim)) for j in range(dim)) for i in range(dim))

def component_geometry_report(chart: CoordinateChart, *, include_curvature: bool = True) -> ComponentGeometryReport:
    dim = chart.dimension
    g = _metric_matrix(chart)
    ginv = sp.simplify(g.inv())
    detg = sp.simplify(g.det())
    vol = sp.simplify(sp.sqrt(sp.Abs(detg)))
    Gamma = _christoffel_from_metric(chart, g)
    riemann = None
    ricci = None
    scalar = None
    if include_curvature:
        coords = tuple(chart.symbols())
        R = [[[[sp.Integer(0) for _ in range(dim)] for _ in range(dim)] for _ in range(dim)] for _ in range(dim)]
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    for l in range(dim):
                        expr = sp.diff(Gamma[i][j][l], coords[k]) - sp.diff(Gamma[i][j][k], coords[l])
                        for m in range(dim):
                            expr += Gamma[i][m][k] * Gamma[m][j][l] - Gamma[i][m][l] * Gamma[m][j][k]
                        R[i][j][k][l] = sp.simplify(expr)
        riemann = tuple(tuple(tuple(tuple(R[i][j][k][l] for l in range(dim)) for k in range(dim)) for j in range(dim)) for i in range(dim))
        ric = sp.MutableDenseMatrix(dim, dim, lambda i, j: sp.Integer(0))
        for j in range(dim):
            for l in range(dim):
                ric[j, l] = sp.simplify(sum(R[i][j][i][l] for i in range(dim)))
        ricci = sp.Matrix(ric)
        scalar = sp.simplify(sum(ginv[i, j] * ricci[i, j] for i in range(dim) for j in range(dim)))
    return ComponentGeometryReport(
        chart_name=getattr(chart, "chart_name", "chart"),
        metric_matrix=g,
        inverse_metric=ginv,
        determinant=detg,
        volume_density=vol,
        christoffel_symbols=Gamma,
        riemann_tensor=riemann,
        ricci_tensor=ricci,
        scalar_curvature=scalar,
    )

def _labels(n: int) -> tuple[str, ...]:
    return tuple(f"e{i}" for i in range(n))

def general_metric_hodge(form: ExteriorFormNF, metric_matrix: Any, *, coords: tuple[Any, ...] | None = None, orientation_sign: int = 1) -> ExteriorFormNF:
    g = sp.Matrix(metric_matrix)
    n = g.rows
    detg = sp.simplify(g.det())
    ginv = sp.simplify(g.inv())
    labels = form.basis_labels or _labels(n)
    coeffs = {}
    for basis_term, coeff in form.terms.items():
        idxs = tuple(int(i) for i in basis_term)
        complement = tuple(i for i in range(n) if i not in idxs)
        metric_factor = sp.Integer(1)
        for i in idxs:
            metric_factor *= ginv[i, i]
        perm = list(idxs) + list(complement)
        inv_count = sum(1 for a in range(len(perm)) for b in range(a + 1, len(perm)) if perm[a] > perm[b])
        sign = -1 if inv_count % 2 else 1
        coeffs[complement] = sp.simplify(coeffs.get(complement, 0) + coeff * sign * orientation_sign * sp.sqrt(sp.Abs(detg)) * metric_factor)
    return canonicalize_exterior_form(ExteriorFormNF(n, coeffs, basis_labels=labels, metadata=dict(form.metadata)))

def general_metric_codifferential(form: ExteriorFormNF, metric_matrix: Any, *, coords: tuple[Any, ...] | None = None, orientation_sign: int = 1) -> ExteriorFormNF:
    g = sp.Matrix(metric_matrix)
    n = g.rows
    coord_tuple = coords or tuple(sp.Symbol(f"x{i}") for i in range(n))
    star1 = general_metric_hodge(form, g, coords=coord_tuple, orientation_sign=orientation_sign)
    d_star = exterior_derivative_nf(star1, coord_tuple)
    star2 = general_metric_hodge(d_star, g, coords=coord_tuple, orientation_sign=orientation_sign)
    sign = (-1) ** (n * form.degree + n + 1)
    coeffs = {k: sp.simplify(sign * v) for k, v in star2.terms.items()}
    return canonicalize_exterior_form(ExteriorFormNF(n, coeffs, basis_labels=star2.basis_labels, metadata=dict(form.metadata)))

def general_metric_hodge_report(form: ExteriorFormNF, metric_matrix: Any, *, coords: tuple[Any, ...] | None = None, orientation_sign: int = 1) -> MetricHodgeReport:
    g = sp.Matrix(metric_matrix)
    result = general_metric_hodge(form, g, coords=coords, orientation_sign=orientation_sign)
    return MetricHodgeReport(input_degree=form.degree, output_degree=result.degree, metric_determinant=sp.simplify(g.det()), signature_parity=0, result=result)
