from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

import sympy as sp

from .symbolic_decision import light_simplify
from .comparison_utils import scalar_is_zero

from .charts import CoordinateChart
from .mappings import CoordinateMap, get_map
from .normal_forms import TNFMatrix, TNFTensorArray, as_tnf_matrix, tnf_build_array, tnf_build_matrix, tnf_matrix_to_sympy


def _basis_simplify_expr(expr):
    expr = sp.sympify(expr)
    return light_simplify(sp.expand(expr))


@dataclass(frozen=True)
class IndexBundle:
    name: str
    dimension: Optional[int] = None


@dataclass(frozen=True)
class TensorFrame:
    """User-defined frame or coframe attached to a chart."""

    name: str
    kind: str
    chart: CoordinateChart
    dimension: int
    transform_to_chart: Callable[[Tuple[sp.Symbol, ...]], sp.Matrix]
    dual_name: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    bundle: Optional[IndexBundle] = None

    def as_basis(self) -> "TensorBasis":
        return TensorBasis(
            self.name,
            self.kind,
            self.chart,
            self.dimension,
            self.dual_name,
            metadata={"transform_to_chart": self.transform_to_chart, "bundle": self.bundle or IndexBundle(self.name, self.dimension)},
        )



@dataclass(frozen=True)
class BasisTransformationReport:
    source: str
    target: str
    matrix: sp.Matrix
    inverse_matrix: sp.Matrix
    roundtrip_error: sp.Matrix
    metadata: dict = field(default_factory=dict)

@dataclass(frozen=True)
class TensorBasis:
    name: str
    kind: str
    chart: Optional[CoordinateChart] = None
    dimension: Optional[int] = None
    dual_name: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def dual(self) -> "TensorBasis":
        if self.kind == "tangent":
            return TensorBasis(self.dual_name or f"d{self.name}", "cotangent", self.chart, self.dimension, self.name, dict(self.metadata))
        if self.kind == "cotangent":
            return TensorBasis(self.dual_name or self.name.removeprefix("d"), "tangent", self.chart, self.dimension, self.name, dict(self.metadata))
        if self.kind == "orthonormal_tangent":
            return TensorBasis(self.dual_name or f"theta({self.name})", "orthonormal_cotangent", self.chart, self.dimension, self.name, dict(self.metadata))
        if self.kind == "orthonormal_cotangent":
            return TensorBasis(self.dual_name or self.name.removeprefix("theta(").removesuffix(")"), "orthonormal_tangent", self.chart, self.dimension, self.name, dict(self.metadata))
        raise ValueError(f"No dual basis rule for kind={self.kind!r}")


def frame_basis(name: str, chart: CoordinateChart, transform_to_chart: Callable[[Tuple[sp.Symbol, ...]], sp.Matrix], *, orthonormal: bool = False, bundle: Optional[IndexBundle] = None, dual_name: Optional[str] = None) -> TensorBasis:
    kind = "orthonormal_tangent" if orthonormal else "tangent"
    return TensorFrame(name, kind, chart, chart.dimension, transform_to_chart, dual_name or f"d{name}", bundle).as_basis()


def coframe_basis(name: str, chart: CoordinateChart, transform_to_chart: Callable[[Tuple[sp.Symbol, ...]], sp.Matrix], *, orthonormal: bool = False, bundle: Optional[IndexBundle] = None, dual_name: Optional[str] = None) -> TensorBasis:
    kind = "orthonormal_cotangent" if orthonormal else "cotangent"
    return TensorFrame(name, kind, chart, chart.dimension, transform_to_chart, dual_name or name.removeprefix("d"), bundle).as_basis()


def tangent_basis(chart: CoordinateChart) -> TensorBasis:
    return TensorBasis(f"T({chart.chart_name})", "tangent", chart, chart.dimension, f"T*({chart.chart_name})", {"bundle": IndexBundle(f"T({chart.chart_name})", chart.dimension)})


def cotangent_basis(chart: CoordinateChart) -> TensorBasis:
    return TensorBasis(f"T*({chart.chart_name})", "cotangent", chart, chart.dimension, f"T({chart.chart_name})", {"bundle": IndexBundle(f"T({chart.chart_name})", chart.dimension)})


def orthonormal_tangent_basis(chart: CoordinateChart) -> TensorBasis:
    return TensorBasis(f"e({chart.chart_name})", "orthonormal_tangent", chart, chart.dimension, f"theta({chart.chart_name})", {"bundle": IndexBundle(f"e({chart.chart_name})", chart.dimension)})


def orthonormal_cotangent_basis(chart: CoordinateChart) -> TensorBasis:
    return TensorBasis(f"theta({chart.chart_name})", "orthonormal_cotangent", chart, chart.dimension, f"e({chart.chart_name})", {"bundle": IndexBundle(f"e({chart.chart_name})", chart.dimension)})


def dual_basis(basis: TensorBasis) -> TensorBasis:
    return basis.dual()


def basis_dimension(basis: TensorBasis) -> Optional[int]:
    return basis.dimension


def _coords(frame: TensorBasis, coords: Optional[Tuple[sp.Symbol, ...]]) -> Tuple[sp.Symbol, ...]:
    if coords is not None:
        return coords
    if frame.chart is None:
        raise ValueError("Basis must be chart-attached when coords are omitted.")
    return frame.chart.symbols()




def _cleanup_with_coord_assumptions(expr, assumptions, coords: Tuple[sp.Symbol, ...]):
    """Lightweight cleanup for basis transforms.

    Pytest assertion/introspection made the older trigsimp/refine path expose
    branch-sensitive symbolic simplification nontermination for cotangent
    cross-chart transforms.  This routine intentionally avoids global trig
    simplification and performs only cheap structural cleanup plus the simple
    Abs replacements implied by coordinate positivity assumptions.
    """
    out = _basis_simplify_expr(expr)
    if assumptions is not None:
        try:
            for rel in assumptions.atoms(sp.StrictGreaterThan, sp.GreaterThan, sp.StrictLessThan, sp.LessThan):
                lhs = getattr(rel, "lhs", None)
                rhs = getattr(rel, "rhs", None)
                if lhs not in coords or rhs != 0:
                    continue
                if rel.rel_op in {">", ">="}:
                    out = out.xreplace({sp.Abs(lhs): lhs})
                elif rel.rel_op in {"<", "<=",}:
                    out = out.xreplace({sp.Abs(lhs): -lhs})
        except Exception:
            pass
    return _basis_simplify_expr(out)

def _basis_matrix_tnf(frame: TensorBasis, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> TNFMatrix:
    actual_coords = _coords(frame, coords)
    transform_to_chart = frame.metadata.get("transform_to_chart")
    if transform_to_chart is None:
        if frame.chart is not None and _basis_kind_family(frame.kind) in {"tangent", "cotangent"}:
            dim = frame.dimension or frame.chart.dimension
            return tnf_build_matrix(dim, dim, lambda i, j: sp.Integer(1) if i == j else sp.Integer(0))
        raise ValueError("Basis does not carry a transform_to_chart map.")
    matrix = as_tnf_matrix(transform_to_chart(actual_coords))
    return matrix.map_entries(_basis_simplify_expr)


def _basis_kind_family(kind: str) -> str:
    return kind.replace("orthonormal_", "")


def _coordinate_basis_for(chart: CoordinateChart, kind: str) -> TensorBasis:
    family = _basis_kind_family(kind)
    if family == "tangent":
        return tangent_basis(chart)
    if family == "cotangent":
        return cotangent_basis(chart)
    raise NotImplementedError(f"Unsupported basis family {kind!r}.")


def _same_chart_basis_transform(source: TensorBasis, target: TensorBasis, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> TNFMatrix:
    chart = source.chart
    actual_coords = coords or chart.symbols()
    if source.kind == target.kind and source.name == target.name:
        return tnf_build_matrix(chart.dimension, chart.dimension, lambda i, j: sp.Integer(1) if i == j else sp.Integer(0))
    source_tf = source.metadata.get("transform_to_chart") if hasattr(source, "metadata") else None
    target_tf = target.metadata.get("transform_to_chart") if hasattr(target, "metadata") else None
    if source_tf is not None and target.kind == source.kind.replace("orthonormal_", ""):
        return as_tnf_matrix(source_tf(actual_coords)).map_entries(_basis_simplify_expr)
    if target_tf is not None and source.kind == target.kind.replace("orthonormal_", ""):
        return as_tnf_matrix(target_tf(actual_coords)).map_entries(_basis_simplify_expr).inv()
    if source_tf is not None and target_tf is not None:
        source_nf = as_tnf_matrix(source_tf(actual_coords)).map_entries(_basis_simplify_expr)
        target_nf = as_tnf_matrix(target_tf(actual_coords)).map_entries(_basis_simplify_expr)
        return target_nf.inv() @ source_nf
    if chart.is_orthogonal(actual_coords):
        scale_factors = chart.scale_factors(actual_coords)
        scale_nf = tnf_build_matrix(chart.dimension, chart.dimension, lambda i, j: _basis_simplify_expr(scale_factors[i]) if i == j else sp.Integer(0))
        inv_nf = tnf_build_matrix(chart.dimension, chart.dimension, lambda i, j: _basis_simplify_expr(1 / scale_factors[i]) if i == j else sp.Integer(0))
        if source.kind == "tangent" and target.kind == "orthonormal_tangent":
            return scale_nf
        if source.kind == "orthonormal_tangent" and target.kind == "tangent":
            return inv_nf
        if source.kind == "cotangent" and target.kind == "orthonormal_cotangent":
            return inv_nf
        if source.kind == "orthonormal_cotangent" and target.kind == "cotangent":
            return scale_nf
    raise NotImplementedError(f"No implemented basis transformation from {source.kind} to {target.kind}.")


def _chart_basis_transform(source: TensorBasis, target: TensorBasis, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> TNFMatrix:
    if source.chart == target.chart:
        return _same_chart_basis_transform(source, target, coords)
    if source.chart is None or target.chart is None:
        raise NotImplementedError("Transformation matrices require chart-attached bases or an explicit abstract transform.")
    mapping = get_map(source.chart, target.chart)
    if not mapping.inverse_available():
        raise NotImplementedError("A symbolic inverse map is required for cross-chart basis transformations.")
    target_coords = coords or target.chart.symbols()
    source_coords = source.chart.symbols()
    inv_exprs = mapping.inverse_mapping_exprs(target_coords)
    subs = dict(zip(source_coords, inv_exprs))
    assumptions = target.chart.assumptions(target_coords)
    J = mapping.jacobian_tnf(source_coords).subs(subs)
    Jinv = mapping.inverse_jacobian_tnf(target_coords)
    source_coord = _coordinate_basis_for(source.chart, source.kind)
    target_coord = _coordinate_basis_for(target.chart, target.kind)
    left = _same_chart_basis_transform(source, source_coord, source_coords)
    if _basis_kind_family(source.kind) == "tangent":
        middle = J.map_entries(lambda e: _cleanup_with_coord_assumptions(e, assumptions, target_coords))
    elif _basis_kind_family(source.kind) == "cotangent":
        middle = Jinv.transpose().map_entries(lambda e: _cleanup_with_coord_assumptions(e, assumptions, target_coords))
    else:
        raise NotImplementedError(f"Unsupported cross-chart basis family {source.kind!r}.")
    right = _same_chart_basis_transform(target_coord, target, target_coords)
    return right @ middle @ left


def basis_transformation_matrix_tnf(source: TensorBasis, target: TensorBasis, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> TNFMatrix:
    if source.dimension != target.dimension:
        raise ValueError("Basis dimensions must match.")
    return _chart_basis_transform(source, target, coords)


def basis_transformation_matrix(source: TensorBasis, target: TensorBasis, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> sp.Matrix:
    return tnf_matrix_to_sympy(basis_transformation_matrix_tnf(source, target, coords))


def transformed_basis(basis: TensorBasis, mapping: CoordinateMap) -> TensorBasis:
    if basis.chart is None:
        return TensorBasis(basis.name, basis.kind, mapping.target, basis.dimension or mapping.target.dimension, basis.dual_name, dict(basis.metadata))
    if basis.chart != mapping.source:
        raise ValueError("Basis chart must match mapping source.")
    bundle = basis.metadata.get("bundle") if hasattr(basis, "metadata") else None
    metadata = dict(getattr(basis, "metadata", {}) or {})
    dim = basis.dimension or mapping.target.dimension
    if bundle is not None:
        metadata["bundle"] = IndexBundle(getattr(bundle, "name", basis.name), getattr(bundle, "dimension", dim))
    source_tf = metadata.get("transform_to_chart")
    if callable(source_tf):
        family = _basis_kind_family(basis.kind)
        if mapping.inverse_available():
            def _target_tf(target_coords, _source_tf=source_tf, _mapping=mapping, _family=family):
                source_coords = _mapping.source.symbols()
                inv_exprs = _mapping.inverse_mapping_exprs(tuple(target_coords))
                subs = dict(zip(source_coords, inv_exprs))
                source_mat = as_tnf_matrix(_source_tf(source_coords)).subs(subs)
                if _family == "tangent":
                    J = _mapping.jacobian_tnf(source_coords).subs(subs)
                    return tnf_matrix_to_sympy((J @ source_mat).map_entries(_basis_simplify_expr))
                if _family == "cotangent":
                    Jinv = _mapping.inverse_jacobian_tnf(tuple(target_coords))
                    return tnf_matrix_to_sympy((Jinv.transpose() @ source_mat).map_entries(_basis_simplify_expr))
                return tnf_matrix_to_sympy(source_mat.map_entries(_basis_simplify_expr))
            metadata["transform_to_chart"] = _target_tf
        else:
            metadata["source_transform_to_chart"] = source_tf
            metadata.pop("transform_to_chart", None)
    return TensorBasis(basis.name, basis.kind, mapping.target, dim, basis.dual_name, metadata)


def frame_to_chart_matrix_tnf(frame: TensorBasis, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> TNFMatrix:
    return _basis_matrix_tnf(frame, coords)


def frame_to_chart_matrix(frame: TensorBasis, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> sp.Matrix:
    return tnf_matrix_to_sympy(frame_to_chart_matrix_tnf(frame, coords))


def chart_to_frame_matrix_tnf(frame: TensorBasis, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> TNFMatrix:
    return _basis_matrix_tnf(frame, coords).inv()


def chart_to_frame_matrix(frame: TensorBasis, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> sp.Matrix:
    return tnf_matrix_to_sympy(chart_to_frame_matrix_tnf(frame, coords))


def frame_metric_tnf(frame: TensorBasis, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> TNFMatrix:
    if frame.chart is None:
        raise ValueError("Frame must be chart-attached.")
    actual_coords = coords or frame.chart.symbols()
    metric_tnf = frame.chart.metric_tnf(actual_coords)
    frame_tnf = _basis_matrix_tnf(frame, actual_coords)
    return frame_tnf.transpose() @ metric_tnf @ frame_tnf


def frame_metric(frame: TensorBasis, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> sp.Matrix:
    return tnf_matrix_to_sympy(frame_metric_tnf(frame, coords))


def _compute_frame_structure_coefficients(frame: TensorBasis, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> TNFTensorArray:
    if frame.chart is None:
        raise ValueError("Frame must be chart-attached.")
    actual_coords = coords or frame.chart.symbols()
    dim = frame.dimension or frame.chart.dimension
    frame_nf = _basis_matrix_tnf(frame, actual_coords)
    frame_inv_nf = frame_nf.inv()
    def _bracket_chart(a: int, i: int, j: int):
        total = sp.Integer(0)
        for b in range(dim):
            total += frame_nf[b, i] * sp.diff(frame_nf[a, j], actual_coords[b]) - frame_nf[b, j] * sp.diff(frame_nf[a, i], actual_coords[b])
        return _basis_simplify_expr(total)
    return tnf_build_array((dim, dim, dim), lambda idx: _basis_simplify_expr(sum(frame_inv_nf[idx[0], a] * _bracket_chart(a, idx[1], idx[2]) for a in range(dim))))


def frame_connection_coefficients(frame: TensorBasis, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> TNFTensorArray:
    if frame.chart is None:
        raise ValueError("Frame must be chart-attached.")
    actual_coords = coords or frame.chart.symbols()
    gamma = frame.chart.christoffel_symbols(actual_coords)
    if gamma is None:
        raise ValueError("Chart does not define a Levi-Civita connection.")
    dim = frame.chart.dimension
    frame_nf = _basis_matrix_tnf(frame, actual_coords)
    frame_inv_nf = frame_nf.inv()
    def _inner(a: int, i: int, j: int):
        total = sp.Integer(0)
        for b in range(dim):
            total += frame_nf[b, i] * sp.diff(frame_nf[a, j], actual_coords[b])
            for c in range(dim):
                total += frame_nf[b, i] * frame_nf[c, j] * gamma[a, b, c]
        return _basis_simplify_expr(total)
    return tnf_build_array((dim, dim, dim), lambda idx: _basis_simplify_expr(sum(frame_inv_nf[idx[0], a] * _inner(a, idx[1], idx[2]) for a in range(dim))))


def _compute_connection_one_forms(frame: TensorBasis, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> tuple[tuple[tuple[sp.Expr, ...], ...], ...]:
    gamma = frame_connection_coefficients(frame, coords)
    dim = frame.dimension or frame.chart.dimension
    return tuple(tuple(tuple(_basis_simplify_expr(gamma[i, j, k]) for k in range(dim)) for j in range(dim)) for i in range(dim))


def frame_commutator_coefficients(chart: CoordinateChart | TensorBasis, frame: Optional[TensorBasis] = None) -> Any:
    return frame_structure_coefficients(chart if frame is None else frame)


def coframe_connection_one_forms(chart: CoordinateChart | TensorBasis, frame: Optional[TensorBasis] = None) -> Any:
    return connection_one_forms(chart if frame is None else frame)


def _wedge_coeffs_1forms(a, b):
    dim = len(a)
    out = {}
    for i in range(dim):
        for j in range(i + 1, dim):
            coeff = sp.simplify(a[i] * b[j] - a[j] * b[i])
            if coeff != 0:
                out[(i, j)] = coeff
    return out


def _add_twoform_dicts(a, b):
    out = dict(a)
    for key, value in b.items():
        out[key] = sp.simplify(out.get(key, 0) + value)
        if out[key] == 0:
            out.pop(key, None)
    return out


def exterior_derivative_coframe_1form(frame: TensorBasis, coeffs: tuple[sp.Expr, ...], coords: Optional[Tuple[sp.Symbol, ...]] = None) -> dict[tuple[int, int], sp.Expr]:
    actual_coords = coords or frame.chart.symbols()
    frame_nf = _basis_matrix_tnf(frame, actual_coords)
    frame_inv_nf = frame_nf.inv()
    structure = frame_structure_coefficients(frame, actual_coords)
    dim = frame.dimension or frame.chart.dimension
    out = {}
    for i in range(dim):
        grad_chart = [sp.diff(coeffs[i], c) for c in actual_coords]
        directional = [sp.simplify(sum(frame_inv_nf[j, a] * grad_chart[a] for a in range(dim))) for j in range(dim)]
        for j in range(dim):
            for k in range(j + 1, dim):
                coeff = sp.simplify(directional[j] * (1 if i == k else 0) - directional[k] * (1 if i == j else 0))
                if coeff != 0:
                    out[(j, k)] = sp.simplify(out.get((j, k), 0) + coeff)
    for i in range(dim):
        for j in range(dim):
            if i < j:
                coeff = sp.Integer(0)
                for k in range(dim):
                    coeff += -sp.simplify(coeffs[k] * structure[k, i, j])
                if coeff != 0:
                    out[(i, j)] = sp.simplify(out.get((i, j), 0) + coeff)
    return {key: sp.simplify(value) for key, value in out.items() if sp.simplify(value) != 0}


def first_structure_equation_residuals(frame: TensorBasis, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> tuple[dict[tuple[int, int], sp.Expr], ...]:
    dim = frame.dimension or frame.chart.dimension
    omega = connection_one_forms(frame, coords)
    residuals = []
    theta = [tuple(1 if i == j else 0 for j in range(dim)) for i in range(dim)]
    for i in range(dim):
        total = dict(exterior_derivative_coframe_1form(frame, theta[i], coords))
        for j in range(dim):
            total = _add_twoform_dicts(total, _wedge_coeffs_1forms(omega[i][j], theta[j]))
        residuals.append(total)
    return tuple(residuals)


def _compute_curvature_two_forms(frame_or_chart: CoordinateChart | TensorBasis, frame: Optional[TensorBasis] = None, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Any:
    actual_frame = frame_or_chart if frame is None else frame
    dim = actual_frame.dimension or actual_frame.chart.dimension
    omega = connection_one_forms(actual_frame, coords)
    out = {}
    for i in range(dim):
        for j in range(dim):
            total = dict(exterior_derivative_coframe_1form(actual_frame, omega[i][j], coords))
            for k in range(dim):
                total = _add_twoform_dicts(total, _wedge_coeffs_1forms(omega[i][k], omega[k][j]))
            out[(i, j)] = total
    return out


def second_structure_equation_residuals(frame: TensorBasis, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> tuple[tuple[dict[tuple[int, int], sp.Expr], ...], ...]:
    dim = frame.dimension or frame.chart.dimension
    curv = curvature_two_forms(frame, None, coords)
    return tuple(tuple(curv[(i, j)] for j in range(dim)) for i in range(dim))


def _compute_torsion_two_forms(frame_or_chart: CoordinateChart | TensorBasis, frame: Optional[TensorBasis] = None, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Any:
    actual_frame = frame_or_chart if frame is None else frame
    residuals = first_structure_equation_residuals(actual_frame, coords)
    return {i: residuals[i] for i in range(actual_frame.dimension or actual_frame.chart.dimension)}


_FRAME_GEOM_CACHE: dict[tuple[object, ...], object] = {}


def _frame_cache_key(tag: str, frame: TensorBasis, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> tuple[object, ...]:
    actual_coords = tuple(coords) if coords is not None else tuple(frame.chart.symbols())
    return (tag, getattr(frame, "name", str(frame)), frame.chart.metric_name, frame.chart.chart_name, frame.chart.dimension, actual_coords)


_raw_frame_structure_coefficients = _compute_frame_structure_coefficients
_raw_connection_one_forms = _compute_connection_one_forms
_raw_curvature_two_forms = _compute_curvature_two_forms
_raw_torsion_two_forms = _compute_torsion_two_forms


def frame_structure_coefficients(frame: TensorBasis, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> TNFTensorArray:
    key = _frame_cache_key("structure", frame, coords)
    if key not in _FRAME_GEOM_CACHE:
        _FRAME_GEOM_CACHE[key] = _raw_frame_structure_coefficients(frame, coords)
    return _FRAME_GEOM_CACHE[key]


def connection_one_forms(frame: TensorBasis, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> tuple[tuple[tuple[sp.Expr, ...], ...], ...]:
    key = _frame_cache_key("connection_one_forms", frame, coords)
    if key not in _FRAME_GEOM_CACHE:
        _FRAME_GEOM_CACHE[key] = _raw_connection_one_forms(frame, coords)
    return _FRAME_GEOM_CACHE[key]


def curvature_two_forms(frame_or_chart: CoordinateChart | TensorBasis, frame: Optional[TensorBasis] = None, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Any:
    actual_frame = frame_or_chart if frame is None else frame
    key = _frame_cache_key("curvature_two_forms", actual_frame, coords)
    if key not in _FRAME_GEOM_CACHE:
        _FRAME_GEOM_CACHE[key] = _raw_curvature_two_forms(frame_or_chart, frame, coords)
    return _FRAME_GEOM_CACHE[key]


def torsion_two_forms(frame_or_chart: CoordinateChart | TensorBasis, frame: Optional[TensorBasis] = None, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Any:
    actual_frame = frame_or_chart if frame is None else frame
    key = _frame_cache_key("torsion_two_forms", actual_frame, coords)
    if key not in _FRAME_GEOM_CACHE:
        _FRAME_GEOM_CACHE[key] = _raw_torsion_two_forms(frame_or_chart, frame, coords)
    return _FRAME_GEOM_CACHE[key]


def gram_schmidt_frame_tnf(chart: CoordinateChart, frame_matrix, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> TNFMatrix:
    if coords is None:
        coords = chart.symbols()
    G = chart.metric_tnf(coords)
    if G is None:
        raise ValueError("Chart does not define a metric.")
    F = as_tnf_matrix(frame_matrix)
    cols = []
    Gs = G.to_sympy()
    Fs = F.to_sympy()
    from .simplification_core import light_simplify
    for j in range(F.shape[1]):
        v = sp.Matrix(Fs[:, j])
        for q in cols:
            denom = light_simplify((q.T * Gs * q)[0])
            if not scalar_is_zero(denom):
                v = v - light_simplify((q.T * Gs * v)[0] / denom) * q
        norm_sq = light_simplify((v.T * Gs * v)[0])
        # Avoid branch-sensitive Abs simplification in symbolic frames; the
        # chart assumptions/cleanup layer can refine this later if needed.
        cols.append(v / sp.sqrt(norm_sq))
    return as_tnf_matrix(sp.Matrix.hstack(*cols))


def gram_schmidt_frame(chart: CoordinateChart, frame_matrix, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> sp.Matrix:
    return tnf_matrix_to_sympy(gram_schmidt_frame_tnf(chart, frame_matrix, coords))


def orthonormal_frame(chart: CoordinateChart, frame_matrix, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> TensorBasis:
    return frame_basis("orthonormal_frame", chart, lambda c: gram_schmidt_frame_tnf(chart, frame_matrix, coords=c).to_sympy(), orthonormal=True)


def basis_pair_contraction_matrix_tnf(left: TensorBasis, right: TensorBasis, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> TNFMatrix:
    if left.chart != right.chart or left.chart is None:
        raise ValueError('Bases must be attached to the same chart.')
    if left.dimension != right.dimension:
        raise ValueError('Basis dimensions must match.')
    if left.kind.startswith('orthonormal') and right.kind.startswith('orthonormal'):
        n = left.dimension or left.chart.dimension
        return tnf_build_matrix(n, n, lambda i, j: sp.Integer(1) if i == j else sp.Integer(0))
    # default coordinate pairing of a basis with its dual
    n = left.dimension or left.chart.dimension
    if left.kind == 'tangent' and right.kind == 'cotangent' or left.kind == 'cotangent' and right.kind == 'tangent':
        return tnf_build_matrix(n, n, lambda i, j: sp.Integer(1) if i == j else sp.Integer(0))
    try:
        mat = basis_transformation_matrix_tnf(left, dual_basis(right), coords)
        return mat
    except Exception:
        return tnf_build_matrix(n, n, lambda i, j: sp.Integer(1) if i == j else sp.Integer(0))


def bases_compatible(left: TensorBasis, right: TensorBasis) -> bool:
    return left.dimension == right.dimension and left.chart == right.chart and left.kind.replace('orthonormal_', '') == right.kind.replace('orthonormal_', '')


def tensor_bases(obj: Any) -> Tuple[TensorBasis, ...]:
    return tuple(getattr(obj, 'slot_bases', tuple()))



def basis_pair_contraction_matrix(left: TensorBasis, right: TensorBasis, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> sp.Matrix:
    return tnf_matrix_to_sympy(basis_pair_contraction_matrix_tnf(left, right, coords))


def resolve_basis_transform(source: TensorBasis, target: TensorBasis, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> sp.Matrix:
    return basis_transformation_matrix(source, target, coords)

def basis_roundtrip_report(source: TensorBasis, target: TensorBasis, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> BasisTransformationReport:
    mat = basis_transformation_matrix(source, target, coords)
    inv = basis_transformation_matrix(target, source, coords)
    n = mat.rows
    ident = sp.eye(n)
    err = (sp.simplify(inv * mat - ident)).applyfunc(_basis_simplify_expr)
    return BasisTransformationReport(source=source.name, target=target.name, matrix=mat, inverse_matrix=inv, roundtrip_error=err, metadata={"coords": tuple(coords) if coords is not None else None})

def coordinate_basis_frame(chart: CoordinateChart, *, cotangent: bool = False) -> TensorBasis:
    return cotangent_basis(chart) if cotangent else tangent_basis(chart)
