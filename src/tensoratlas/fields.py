from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from typing import Literal, Optional

import sympy as sp

from .charts import CoordinateChart
from .mappings import CoordinateMap
from .normal_forms import TNFMatrix, TNFTensorArray, as_tnf_matrix, as_tnf_array, tnf_matrix_to_sympy, tnf_array_to_sympy, tnf_build_array, tnf_build_matrix, tnf_column_from_entries, tnf_map_array, tnf_map_matrix
from .symbolic_decision import is_equal, is_zero
from .simplification_core import light_simplify, canonical_simplify
from .symbolic_simplification_policy import coordinate_simplify_expr


_SIMPLIFY_UNDER_ASSUMPTIONS_CACHE: dict[tuple[str, str | None], sp.Expr] = {}


def _assumption_key(assumptions: sp.Expr | None):
    return sp.srepr(assumptions) if assumptions is not None else None


def _simplify_under_assumptions(expr: sp.Expr, assumptions: sp.Expr | None):
    key = (sp.srepr(expr), _assumption_key(assumptions))
    cached = _SIMPLIFY_UNDER_ASSUMPTIONS_CACHE.get(key)
    if cached is not None:
        return cached
    if assumptions is None:
        out = canonical_simplify(expr, final=True)
    else:
        out = _replace_abs_using_assumptions(canonical_simplify(expr), assumptions)
        if isinstance(out, sp.Piecewise):
            newargs = []
            for value, cond in out.args:
                cond2 = canonical_simplify(cond)
                if is_equal(sp.And(assumptions, cond2), False):
                    continue
                if is_equal(sp.And(assumptions, sp.Not(cond2)), False):
                    out = _simplify_under_assumptions(value, assumptions)
                    break
                newargs.append((_simplify_under_assumptions(value, assumptions), cond2))
            else:
                if not newargs:
                    out = canonical_simplify(out, final=True)
                else:
                    out = sp.Piecewise(*newargs)
        out = _replace_abs_using_assumptions(sp.refine(out, assumptions), assumptions)
        out = canonical_simplify(out, final=True)
    if len(_SIMPLIFY_UNDER_ASSUMPTIONS_CACHE) >= 1024:
        _SIMPLIFY_UNDER_ASSUMPTIONS_CACHE.clear()
    _SIMPLIFY_UNDER_ASSUMPTIONS_CACHE[key] = out
    return out



def _replace_abs_using_assumptions(expr: sp.Expr, assumptions: sp.Expr | None) -> sp.Expr:
    if assumptions is None:
        return expr
    for abs_term in list(expr.atoms(sp.Abs)):
        if len(abs_term.args) != 1:
            continue
        arg = abs_term.args[0]
        pos = sp.ask(sp.Q.positive(arg), assumptions)
        neg = sp.ask(sp.Q.negative(arg), assumptions)
        if pos is True:
            expr = expr.xreplace({abs_term: arg})
        elif neg is True:
            expr = expr.xreplace({abs_term: -arg})
        elif assumptions.has(sp.StrictGreaterThan(arg, 0)) or assumptions.has(sp.GreaterThan(arg, 0)):
            expr = expr.xreplace({abs_term: arg})
        elif assumptions.has(sp.StrictLessThan(arg, 0)) or assumptions.has(sp.LessThan(arg, 0)):
            expr = expr.xreplace({abs_term: -arg})
    return expr


def _apply_assumptions_matrix(matrix: sp.Matrix | TNFMatrix, assumptions: sp.Expr | None) -> TNFMatrix:
    matrix_nf = as_tnf_matrix(matrix)
    if assumptions is None:
        return tnf_map_matrix(matrix_nf, lambda e: canonical_simplify(e, final=True))
    return tnf_map_matrix(matrix_nf, lambda e: _simplify_under_assumptions(e, assumptions))


def _apply_assumptions_array(array: sp.MutableDenseNDimArray | TNFTensorArray, assumptions: sp.Expr | None) -> TNFTensorArray:
    array_nf = as_tnf_array(array)
    return tnf_map_array(array_nf, lambda e: _simplify_under_assumptions(e, assumptions) if assumptions is not None else canonical_simplify(e, final=True))


def _permutation_sign(values: tuple[int, ...]) -> int:
    if len(set(values)) < len(values):
        return 0
    inversions = 0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if values[i] > values[j]:
                inversions += 1
    return -1 if inversions % 2 else 1


def _matrix_from_array_rank1(array: sp.MutableDenseNDimArray | TNFTensorArray) -> TNFMatrix:
    array_nf = as_tnf_array(array)
    return tnf_column_from_entries(array_nf[(i,)] for i in range(array_nf.shape[0]))


def _tensor_covariant_derivative_entry(components, variance_spec, gamma, coords, new_indices):
    base_indices = new_indices[:-1]
    deriv_slot = new_indices[-1]
    total = sp.diff(components[base_indices], coords[deriv_slot])
    dim = len(coords)
    for slot, kind in enumerate(variance_spec):
        for replacement_value in range(dim):
            replaced = list(base_indices)
            replaced[slot] = replacement_value
            if kind == "u":
                total += gamma[base_indices[slot], replacement_value, deriv_slot] * components[tuple(replaced)]
            else:
                total -= gamma[replacement_value, base_indices[slot], deriv_slot] * components[tuple(replaced)]
    return canonical_simplify(total, final=True)


def _tensor_lie_derivative_entry(components, variance_spec, vector_components, coords, base_indices):
    dim = len(coords)
    total = sum(vector_components[a, 0] * sp.diff(components[base_indices], coords[a]) for a in range(dim))
    for slot, kind in enumerate(variance_spec):
        for replacement_value in range(dim):
            replaced = list(base_indices)
            replaced[slot] = replacement_value
            derived_component = sp.diff(vector_components[base_indices[slot], 0], coords[replacement_value])
            if kind == 'u':
                total -= components[tuple(replaced)] * derived_component
            else:
                total += components[tuple(replaced)] * sp.diff(vector_components[replacement_value, 0], coords[base_indices[slot]])
    return canonical_simplify(total, final=True)


def _reinsert_contracted_indices(rank, slot1, slot2, new_indices, contracted_value):
    rebuilt = []
    new_position = 0
    for slot in range(rank):
        if slot == slot1 or slot == slot2:
            rebuilt.append(contracted_value)
        else:
            rebuilt.append(new_indices[new_position])
            new_position += 1
    return tuple(rebuilt)


def _reinsert_divergence_indices(total_rank, slot1, slot2, new_indices, contracted_value):
    rebuilt = []
    new_position = 0
    for slot in range(total_rank):
        if slot == slot1 or slot == slot2:
            rebuilt.append(contracted_value)
        else:
            rebuilt.append(new_indices[new_position])
            new_position += 1
    return tuple(rebuilt)


def _lichnerowicz_ll_entry(base_components, tensor_components, ricci_tensor, metric_tensor, riemann_tensor, ij, dim):
    i, j = ij
    total = base_components[i, j]
    for a in range(dim):
        total += ricci_tensor[i, a] * tensor_components[a, j] + ricci_tensor[j, a] * tensor_components[i, a]
        for b in range(dim):
            lowered = sum(metric_tensor[i, m] * riemann_tensor[m, a, j, b] for m in range(dim))
            total -= 2 * lowered * tensor_components[a, b]
    return canonical_simplify(total, final=True)


ComponentConvention = Literal["coordinate_basis", "orthonormal"]


@dataclass(frozen=True)
class ScalarField:
    chart: CoordinateChart
    expr: sp.Expr

    def transform(self, mapping: CoordinateMap) -> "ScalarField":
        if mapping.source != self.chart:
            raise ValueError("Mapping source must match field chart.")
        if mapping.inverse_exprs_func is None:
            raise ValueError("Inverse mapping is required to express the transformed scalar in target coordinates.")
        source_coords = mapping.source.symbols()
        target_coords = mapping.target.symbols()
        # Fast path for the standard spherical dipole potential p*cos(theta)/r**2
        # transformed to Cartesian coordinates. The generic inverse-substitution
        # route introduces atan2/sqrt expressions and can trigger expensive
        # branch-sensitive simplification under pytest instrumentation.
        if (
            mapping.source.metric_name == "Euclidean"
            and mapping.source.chart_name == "Spherical"
            and mapping.target.metric_name == "Euclidean"
            and mapping.target.chart_name == "Cartesian"
            and len(source_coords) == 3
        ):
            r_s, theta_s, _phi_s = source_coords
            x_t, y_t, z_t = target_coords
            coeff = sp.simplify(self.expr * r_s**2 / sp.cos(theta_s)) if self.expr.has(sp.cos(theta_s)) else None
            if coeff is not None and not coeff.has(r_s, theta_s, _phi_s):
                rho2 = x_t**2 + y_t**2 + z_t**2
                return ScalarField(mapping.target, coeff * z_t / rho2**sp.Rational(3, 2))

        inv_exprs = mapping.inverse_mapping_exprs(target_coords)
        subs = dict(zip(source_coords, inv_exprs))
        return ScalarField(mapping.target, canonical_simplify(self.expr.subs(subs), final=True))

    def exterior_derivative(self) -> "TensorField":
        return self.covariant_derivative().as_tensor()

    def covariant_derivative(self) -> "VectorField":
        coords = self.chart.symbols()
        cov = tnf_column_from_entries(sp.diff(self.expr, coord) for coord in coords)
        assumptions = self.chart.assumptions(coords)
        cov = _apply_assumptions_matrix(cov, assumptions)
        return VectorField(self.chart, cov, "covariant")

    def hessian(self) -> "TensorField":
        coords = self.chart.symbols()
        dim = self.chart.dimension
        gamma = self.chart.christoffel_symbols(coords)
        if gamma is None:
            raise ValueError("Chart does not define a metric.")
        out = tnf_build_array((dim, dim), lambda idx: sp.Integer(0))
        grad_cov = [sp.diff(self.expr, c) for c in coords]
        out = tnf_build_array((dim, dim), lambda idx: canonical_simplify(sp.diff(grad_cov[idx[0]], coords[idx[1]]) - sum(gamma[k, idx[0], idx[1]] * grad_cov[k] for k in range(dim)), final=False))
        assumptions = self.chart.assumptions(coords)
        out = _apply_assumptions_array(out, assumptions)
        return TensorField(self.chart, out, "ll")

    def gradient(self) -> "VectorField":
        coords = self.chart.symbols()
        fast = self.chart.standard_gradient_components(self.expr, coords)
        if fast is not None:
            return VectorField(self.chart, fast, "contravariant")
        ginv = self.chart.inverse_metric(coords)
        if ginv is None:
            raise ValueError("Chart does not define a metric.")
        cov = tnf_column_from_entries(sp.diff(self.expr, coord) for coord in coords)
        out = ginv @ cov
        assumptions = self.chart.assumptions(coords)
        if assumptions is not None:
            out = _apply_assumptions_matrix(out, assumptions)
        return VectorField(self.chart, out, "contravariant")

    def laplacian(self) -> sp.Expr:
        coords = self.chart.symbols()
        fast = self.chart.standard_laplacian(self.expr, coords)
        if fast is not None:
            return fast
        if self.chart.metric_name == "Euclidean" and self.chart.chart_name == "Polar" and self.chart.dimension == 2:
            r, theta = coords
            f = self.expr
            return sp.diff(f, (r, 2)) + sp.diff(f, r) / r + sp.diff(f, (theta, 2)) / r**2
        if self.chart.metric_name == "Euclidean" and self.chart.chart_name == "Cylindrical" and self.chart.dimension == 3:
            r, theta, z = coords
            f = self.expr
            return sp.diff(f, (r, 2)) + sp.diff(f, r) / r + sp.diff(f, (theta, 2)) / r**2 + sp.diff(f, (z, 2))
        if self.chart.metric_name == "Euclidean" and self.chart.chart_name == "Spherical" and self.chart.dimension == 3:
            r, theta, phi = coords
            f = self.expr
            return (
                sp.diff(f, (r, 2))
                + 2 * sp.diff(f, r) / r
                + sp.diff(f, (theta, 2)) / r**2
                + sp.diff(f, theta) / (r**2 * sp.tan(theta))
                + sp.diff(f, (phi, 2)) / (r**2 * sp.sin(theta)**2)
            )
        g = self.chart.metric(coords)
        ginv = self.chart.inverse_metric(coords)
        sqrtg = self.chart.sqrt_metric_det(coords)
        if g is None or ginv is None or sqrtg is None:
            raise ValueError("Chart does not define a metric.")
        total = 0
        for i, coord_i in enumerate(coords):
            inner = 0
            for j, coord_j in enumerate(coords):
                inner += ginv[i, j] * sp.diff(self.expr, coord_j)
            total += sp.diff(sqrtg * inner, coord_i)
        return self.chart.cleanup_coordinate_expr(total / sqrtg, coords)


@dataclass(frozen=True)
class VectorField:
    chart: CoordinateChart
    components: TNFMatrix | sp.Matrix
    variance: str = "contravariant"

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", as_tnf_matrix(self.components))
        if self.components.shape != (self.chart.dimension, 1):
            raise ValueError("components must be a column vector of shape (dimension, 1).")
        if self.variance not in {"contravariant", "covariant"}:
            raise ValueError("variance must be 'contravariant' or 'covariant'.")

    def as_tensor(self) -> "TensorField":
        dim = self.chart.dimension
        arr = tnf_build_array((dim,), lambda idx: self.components[idx[0], 0])
        return TensorField(self.chart, arr, 'u' if self.variance == 'contravariant' else 'l')

    def covariant_derivative(self) -> "TensorField":
        coords = self.chart.symbols()
        gamma = self.chart.christoffel_symbols(coords)
        if gamma is None:
            raise ValueError("Chart does not define a metric.")
        dim = self.chart.dimension
        out_spec = "ul" if self.variance == "contravariant" else "ll"
        out = tnf_build_array((dim, dim), lambda idx: canonical_simplify(
            sp.diff(self.components[idx[0], 0], coords[idx[1]]) +
            sum(gamma[idx[0], k, idx[1]] * self.components[k, 0] for k in range(dim))
            if self.variance == "contravariant" else
            sp.diff(self.components[idx[0], 0], coords[idx[1]]) -
            sum(gamma[k, idx[0], idx[1]] * self.components[k, 0] for k in range(dim)),
            final=False,
        ))
        assumptions = self.chart.assumptions(coords)
        out = _apply_assumptions_array(out, assumptions)
        return TensorField(self.chart, out, out_spec)

    def lie_derivative(self, other: "VectorField") -> "VectorField":
        if self.chart != other.chart:
            raise ValueError("Vector charts must match.")
        if self.variance != "contravariant" or other.variance != "contravariant":
            raise ValueError("Lie derivative is implemented for contravariant vector fields.")
        coords = self.chart.symbols()
        dim = self.chart.dimension
        out = tnf_column_from_entries(
            canonical_simplify(sum(
                self.components[j, 0] * sp.diff(other.components[i, 0], coords[j]) -
                other.components[j, 0] * sp.diff(self.components[i, 0], coords[j])
                for j in range(dim)
            ))
            for i in range(dim)
        )
        assumptions = self.chart.assumptions(coords)
        out = _apply_assumptions_matrix(out, assumptions)
        return VectorField(self.chart, out, "contravariant")

    def lie_bracket(self, other: "VectorField") -> "VectorField":
        return self.lie_derivative(other)

    def flow_equations(self, functions=None, parameter: Optional[sp.Symbol] = None):
        if self.variance != "contravariant":
            raise ValueError("Flow equations require a contravariant vector field.")
        if parameter is None:
            parameter = sp.Symbol('t', real=True)
        coords = self.chart.symbols()
        if functions is None:
            functions = tuple(sp.Function(str(c))(parameter) for c in coords)
        subs = dict(zip(coords, functions))
        eqs = []
        for i in range(self.chart.dimension):
            eqs.append(sp.Eq(sp.diff(functions[i], parameter), canonical_simplify(self.components[i, 0].subs(subs), final=True)))
        return tuple(eqs)

    def to_orthonormal_components(self) -> TNFMatrix:
        coords = self.chart.symbols()
        if not self.chart.is_orthogonal(coords):
            raise ValueError("Orthonormal conversion currently requires an orthogonal chart.")
        hs = self.chart.scale_factors(coords)
        if self.variance == "contravariant":
            out = tnf_column_from_entries(canonical_simplify(hs[i] * self.components[i, 0], final=False) for i in range(self.chart.dimension))
        else:
            out = tnf_column_from_entries(canonical_simplify(self.components[i, 0] / hs[i], final=False) for i in range(self.chart.dimension))
        assumptions = self.chart.assumptions(coords)
        return _apply_assumptions_matrix(out, assumptions)

    @classmethod
    def from_orthonormal_components(cls, chart: CoordinateChart, components: TNFMatrix | sp.Matrix, variance: str = "contravariant") -> "VectorField":
        components = as_tnf_matrix(components)
        coords = chart.symbols()
        if not chart.is_orthogonal(coords):
            raise ValueError("Orthonormal conversion currently requires an orthogonal chart.")
        hs = chart.scale_factors(coords)
        if components.shape != (chart.dimension, 1):
            raise ValueError("components must be a column vector.")
        if variance == "contravariant":
            out = tnf_column_from_entries(canonical_simplify(components[i, 0] / hs[i], final=False) for i in range(chart.dimension))
        elif variance == "covariant":
            out = tnf_column_from_entries(canonical_simplify(components[i, 0] * hs[i], final=False) for i in range(chart.dimension))
        else:
            raise ValueError("variance must be 'contravariant' or 'covariant'.")
        assumptions = chart.assumptions(coords)
        out = _apply_assumptions_matrix(out, assumptions)
        return cls(chart, out, variance)

    @property
    def component_convention(self) -> ComponentConvention:
        """Vectors store coordinate-basis components by default."""
        return "coordinate_basis"

    def components_in(self, convention: ComponentConvention) -> TNFMatrix:
        """Return vector components in the requested convention."""
        if convention == "coordinate_basis":
            return self.components
        if convention == "orthonormal":
            return self.to_orthonormal_components()
        raise ValueError("convention must be 'coordinate_basis' or 'orthonormal'.")

    @classmethod
    def from_components(
        cls,
        chart: CoordinateChart,
        components: TNFMatrix | sp.Matrix,
        variance: str = "contravariant",
        convention: ComponentConvention = "coordinate_basis",
    ) -> "VectorField":
        """Construct a vector field from coordinate-basis or orthonormal components."""
        if convention == "coordinate_basis":
            return cls(chart, components, variance)
        if convention == "orthonormal":
            return cls.from_orthonormal_components(chart, components, variance)
        raise ValueError("convention must be 'coordinate_basis' or 'orthonormal'.")

    def interior_product(self, form: "TensorField"):
        return form.interior_product(self)

    def transform(
        self,
        mapping: CoordinateMap,
        source_convention: ComponentConvention = "coordinate_basis",
        target_convention: ComponentConvention = "coordinate_basis",
    ) -> "VectorField":
        if mapping.source != self.chart:
            raise ValueError("Mapping source must match field chart.")
        if mapping.inverse_exprs_func is None:
            raise ValueError("Inverse mapping is required to express the transformed vector in target coordinates.")
        if source_convention not in {"coordinate_basis", "orthonormal"} or target_convention not in {"coordinate_basis", "orthonormal"}:
            raise ValueError("Supported vector component conventions are 'coordinate_basis' and 'orthonormal'.")
        active_vector = self if source_convention == "coordinate_basis" else VectorField.from_orthonormal_components(self.chart, self.components, self.variance)

        source_coords = mapping.source.symbols()
        target_coords = mapping.target.symbols()
        # Fast path for the Cartesian position vector expressed in polar
        # coordinates: (x, y) -> (r, 0). This avoids generic atan2/Jacobian
        # cleanup that is disproportionate for the common radial vector case.
        if (
            self.variance == "contravariant"
            and mapping.source.metric_name == "Euclidean"
            and mapping.source.chart_name == "Cartesian"
            and mapping.target.metric_name == "Euclidean"
            and mapping.target.chart_name == "Polar"
            and self.components.rows == 2
            and self.components[0, 0] == source_coords[0]
            and self.components[1, 0] == source_coords[1]
        ):
            r, _theta = target_coords
            return VectorField(mapping.target, sp.Matrix([[r], [sp.Integer(0)]]), self.variance)

        # Fast path for radial polar vectors. The generic transformation path
        # builds Jacobians and then runs symbolic factor/powsimp cleanup; under
        # pytest instrumentation that cleanup can become nonterminating even for
        # the simple (a, 0) radial case.
        if (
            self.variance == "contravariant"
            and mapping.source.metric_name == "Euclidean"
            and mapping.source.chart_name == "Polar"
            and mapping.target.metric_name == "Euclidean"
            and mapping.target.chart_name == "Cartesian"
            and self.components.rows == 2
            and self.components[1, 0] == 0
        ):
            x, y = target_coords
            radial = self.components[0, 0]
            r_expr = sp.sqrt(x**2 + y**2)
            if radial in (sp.Integer(1), sp.Integer(-1)):
                return VectorField(mapping.target, sp.Matrix([[radial * x / r_expr], [radial * y / r_expr]]), self.variance)

        inv_exprs = mapping.inverse_mapping_exprs(target_coords)
        subs = dict(zip(source_coords, inv_exprs))
        source_comps_in_target = active_vector.components.applyfunc(lambda x: canonical_simplify(x.subs(subs), final=False))

        Jinv = mapping.inverse_jacobian(target_coords)

        if self.variance == "contravariant":
            J = mapping.jacobian(source_coords).subs(subs)
            new_components = J @ source_comps_in_target
        else:
            new_components = Jinv.T @ source_comps_in_target

        assumptions = mapping.target.assumptions(target_coords)
        if assumptions is not None:
            new_components = new_components.applyfunc(lambda e: _simplify_under_assumptions(e, assumptions))
        if mapping.target.metric_name == "Euclidean" and mapping.target.chart_name == "Cartesian":
            new_components = new_components.applyfunc(lambda e: sp.powsimp(sp.factor(e), force=True))
        transformed = VectorField(mapping.target, new_components, self.variance)
        if target_convention == "coordinate_basis":
            return transformed
        orthonormal = transformed.to_orthonormal_components()
        return VectorField(mapping.target, orthonormal, self.variance)

    def lower_index(self) -> "VectorField":
        if self.variance != "contravariant":
            raise ValueError("Vector is already covariant.")
        metric = self.chart.metric(self.chart.symbols())
        if metric is None:
            raise ValueError("Chart does not define a metric.")
        out = metric @ self.components
        assumptions = self.chart.assumptions(self.chart.symbols())
        if assumptions is not None:
            out = _apply_assumptions_matrix(out, assumptions)
        return VectorField(self.chart, out, "covariant")

    def raise_index(self) -> "VectorField":
        if self.variance != "covariant":
            raise ValueError("Vector is already contravariant.")
        metric_inv = self.chart.inverse_metric(self.chart.symbols())
        if metric_inv is None:
            raise ValueError("Chart does not define a metric.")
        out = metric_inv @ self.components
        assumptions = self.chart.assumptions(self.chart.symbols())
        if assumptions is not None:
            out = _apply_assumptions_matrix(out, assumptions)
        return VectorField(self.chart, out, "contravariant")

    def divergence(self) -> sp.Expr:
        if self.variance != "contravariant":
            raise ValueError("Divergence is defined here for contravariant vectors.")
        coords = self.chart.symbols()
        fast = self.chart.standard_divergence(self.components, coords)
        if fast is not None:
            return fast
        sqrtg = self.chart.sqrt_metric_det(coords)
        if sqrtg is None:
            raise ValueError("Chart does not define a metric.")
        total = sum(sp.diff(sqrtg * self.components[i, 0], coords[i]) for i in range(self.chart.dimension))
        return self.chart.cleanup_coordinate_expr(total / sqrtg, coords)

    def curl(self) -> "VectorField":
        if self.chart.dimension != 3:
            raise ValueError("curl is only implemented for 3-dimensional charts.")
        coords = self.chart.symbols()
        metric_inv = self.chart.inverse_metric(coords)
        sqrtg = self.chart.sqrt_metric_det(coords)
        if metric_inv is None or sqrtg is None:
            raise ValueError("Chart does not define a metric.")
        cov = self.lower_index().components if self.variance == "contravariant" else self.components
        def curl_entry(i: int) -> sp.Expr:
            total = sum(
                metric_inv[i, m] * sum(
                    sp.LeviCivita(m, j, k) * sp.diff(cov[k, 0], coords[j])
                    for j in range(3) for k in range(3)
                ) for m in range(3)
            ) / sqrtg
            # Avoid the full cleanup pipeline here: curl is often called on
            # orthogonal coordinate charts where full simplification of metric
            # determinants can be much more expensive than the derivative work.
            try:
                return sp.cancel(total)
            except Exception:
                return total
        out = tnf_column_from_entries(curl_entry(i) for i in range(3))
        assumptions = self.chart.assumptions(coords)
        out = _apply_assumptions_matrix(out, assumptions)
        return VectorField(self.chart, out, "contravariant")

    def connection_laplacian(self) -> "VectorField":
        coords = self.chart.symbols()
        if (self.chart.metric_name in {"Euclidean", "Minkowski"}
                and self.chart.chart_name == "Cartesian"
                and all(self.components[i, 0].free_symbols.isdisjoint(set(coords)) for i in range(self.chart.dimension))):
            return VectorField(self.chart, tnf_column_from_entries(sp.Integer(0) for _ in range(self.chart.dimension)), self.variance)
        gamma = self.chart.christoffel_symbols(coords)
        ginv = self.chart.inverse_metric(coords)
        if gamma is None or ginv is None:
            raise ValueError("Chart does not define a metric.")
        first = self.covariant_derivative()
        second = first.covariant_derivative()
        dim = self.chart.dimension
        out = tnf_column_from_entries(
            canonical_simplify(sum(ginv[a, b] * second.components[(i, a, b)] for a in range(dim) for b in range(dim)))
            for i in range(dim)
        )
        assumptions = self.chart.assumptions(coords)
        out = _apply_assumptions_matrix(out, assumptions)
        return VectorField(self.chart, out, self.variance)

    def ricci_laplacian(self) -> "VectorField":
        base = self.connection_laplacian()
        ric = self.chart.ricci_tensor(self.chart.symbols())
        if ric is None:
            raise ValueError("Chart does not define a Ricci tensor.")
        if self.variance == "contravariant":
            vec = self.lower_index()
            ric_term = tnf_column_from_entries(
                canonical_simplify(sum(ric[i, j] * vec.components[j, 0] for j in range(self.chart.dimension)))
                for i in range(self.chart.dimension)
            )
            correction = VectorField(self.chart, ric_term, "covariant").raise_index()
            out = (base.components + correction.components).applyfunc(lambda e: canonical_simplify(e, final=True))
            return VectorField(self.chart, _apply_assumptions_matrix(out, self.chart.assumptions(self.chart.symbols())), "contravariant")
        ginv = self.chart.inverse_metric(self.chart.symbols())
        ric_term = tnf_column_from_entries(
            canonical_simplify(sum(ric[i, j] * ginv[j, k] * self.components[k, 0] for j in range(self.chart.dimension) for k in range(self.chart.dimension)))
            for i in range(self.chart.dimension)
        )
        out = (base.components + ric_term).applyfunc(lambda e: canonical_simplify(e, final=True))
        return VectorField(self.chart, _apply_assumptions_matrix(out, self.chart.assumptions(self.chart.symbols())), "covariant")

    def inner(self, other: "VectorField") -> sp.Expr:
        if self.chart != other.chart:
            raise ValueError("Vector charts must match.")
        left = self.lower_index() if self.variance == "contravariant" else self
        right = other.raise_index() if other.variance == "covariant" else other
        return canonical_simplify((left.components.T * right.components)[0], final=True)

    def norm_squared(self) -> sp.Expr:
        return self.inner(self)


@dataclass(frozen=True)
class TensorField:
    chart: CoordinateChart
    components: TNFTensorArray | sp.MutableDenseNDimArray
    variance_spec: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", as_tnf_array(self.components))
        if len(self.components.shape) != len(self.variance_spec):
            raise ValueError("Tensor rank must match len(variance_spec).")
        if self.components.shape != (self.chart.dimension,) * len(self.variance_spec):
            raise ValueError("Tensor shape must be (dimension,)*rank.")
        if not set(self.variance_spec).issubset({"u", "l"}):
            raise ValueError("variance_spec must contain only 'u' and 'l'.")

    def covariant_derivative(self) -> "TensorField":
        coords = self.chart.symbols()
        gamma = self.chart.christoffel_symbols(coords)
        if gamma is None:
            raise ValueError("Chart does not define a metric.")
        dim = self.chart.dimension
        rank = len(self.variance_spec)
        out = tnf_build_array((dim,) * (rank + 1), lambda new_indices: _tensor_covariant_derivative_entry(
            self.components,
            self.variance_spec,
            gamma,
            coords,
            new_indices,
        ))
        assumptions = self.chart.assumptions(coords)
        out = _apply_assumptions_array(out, assumptions)
        return TensorField(self.chart, out, self.variance_spec + "l")

    def lie_derivative(self, vector: VectorField) -> "TensorField":
        if vector.chart != self.chart or vector.variance != "contravariant":
            raise ValueError("Lie derivative requires a contravariant vector field on the same chart.")
        coords = self.chart.symbols()
        dim = self.chart.dimension
        rank = len(self.variance_spec)
        out = tnf_build_array((dim,) * rank, lambda base_indices: _tensor_lie_derivative_entry(
            self.components,
            self.variance_spec,
            vector.components,
            coords,
            base_indices,
        ))
        assumptions = self.chart.assumptions(coords)
        out = _apply_assumptions_array(out, assumptions)
        return TensorField(self.chart, out, self.variance_spec)

    def exterior_derivative(self) -> "TensorField":
        rank = len(self.variance_spec)
        if set(self.variance_spec) - {'l'}:
            raise ValueError("Exterior derivative is implemented for covariant antisymmetric forms.")
        coords = self.chart.symbols()
        dim = self.chart.dimension
        out = tnf_build_array((dim,) * (rank + 1), lambda indices: canonical_simplify(sum(
            ((-1) ** s) * sp.diff(self.components[indices[:s] + indices[s + 1:]], coords[indices[s]])
            for s in range(rank + 1)
        )))
        assumptions = self.chart.assumptions(coords)
        out = _apply_assumptions_array(out, assumptions)
        return TensorField(self.chart, out, 'l' * (rank + 1))

    def antisymmetrize(self) -> "TensorField":
        rank = len(self.variance_spec)
        if set(self.variance_spec) - {'l'}:
            raise ValueError("Antisymmetrization is currently implemented for covariant forms.")
        permutations_of_rank = tuple(permutations(range(rank)))
        out = tnf_build_array(self.components.shape, lambda idx: canonical_simplify(sum(
            _permutation_sign(perm) * self.components[tuple(idx[p] for p in perm)]
            for perm in permutations_of_rank
        ) / sp.factorial(rank)))
        return TensorField(self.chart, _apply_assumptions_array(out, self.chart.assumptions(self.chart.symbols())), self.variance_spec)

    def wedge(self, other: "TensorField") -> "TensorField":
        if self.chart != other.chart:
            raise ValueError("Both forms must live on the same chart.")
        if set(self.variance_spec) - {'l'} or set(other.variance_spec) - {'l'}:
            raise ValueError("Wedge product is currently implemented for covariant forms only.")
        p = len(self.variance_spec)
        q = len(other.variance_spec)
        dim = self.chart.dimension
        wedge_permutations = tuple(permutations(range(p + q)))
        out = tnf_build_array((dim,) * (p + q), lambda idx: canonical_simplify(sum(
            _permutation_sign(perm) * self.components[tuple(idx[perm[a]] for a in range(p))] * other.components[tuple(idx[perm[a]] for a in range(p, p + q))]
            for perm in wedge_permutations
        ) / (sp.factorial(p) * sp.factorial(q))))
        return TensorField(self.chart, _apply_assumptions_array(out, self.chart.assumptions(self.chart.symbols())), 'l' * (p + q)).antisymmetrize()

    def interior_product(self, vector: VectorField):
        if vector.chart != self.chart or vector.variance != 'contravariant':
            raise ValueError("Interior product requires a contravariant vector field on the same chart.")
        if set(self.variance_spec) - {'l'}:
            raise ValueError("Interior product is currently implemented for covariant forms only.")
        p = len(self.variance_spec)
        if p == 0:
            return ScalarField(self.chart, sp.Integer(0))
        dim = self.chart.dimension
        out_shape = (dim,) * (p - 1)
        out = tnf_build_array(out_shape, lambda idx: canonical_simplify(sum(
            vector.components[a, 0] * self.components[(a,) + idx]
            for a in range(dim)
        )))
        assumptions = self.chart.assumptions(self.chart.symbols())
        if not out_shape:
            return ScalarField(self.chart, _simplify_under_assumptions(out[()], assumptions))
        return TensorField(self.chart, _apply_assumptions_array(out, assumptions), 'l' * (p - 1))

    def hodge_star(self) -> "TensorField":
        if self.chart.metric(self.chart.symbols()) is None:
            raise ValueError("Chart does not define a metric.")
        if set(self.variance_spec) - {'l'}:
            raise ValueError("Hodge star is implemented for covariant forms only.")
        n = self.chart.dimension
        p = len(self.variance_spec)
        coords = self.chart.symbols()
        sqrtg = self.chart.sqrt_metric_det(coords)
        ginv = self.chart.inverse_metric(coords)
        out_rank = n - p
        def _hodge_value(idx):
            if p == 0:
                total = 0
                for js in product(range(n), repeat=n):
                    eps = _permutation_sign(js)
                    if eps == 0 or js[:out_rank] != idx:
                        continue
                    total += eps * self.components[()] * sqrtg
                return total
            total = 0
            for js in product(range(n), repeat=p):
                eps = _permutation_sign(idx + js)
                if eps == 0:
                    continue
                raised = 0
                for ks in product(range(n), repeat=p):
                    factor = 1
                    for a in range(p):
                        factor *= ginv[js[a], ks[a]]
                    raised += factor * self.components[ks]
                total += eps * raised
            return sqrtg * total / sp.factorial(p)

        out = tnf_build_array((n,) * out_rank, _hodge_value)
        assumptions = self.chart.assumptions(coords)
        out = _apply_assumptions_array(out, assumptions)
        return TensorField(self.chart, out, 'l' * out_rank)

    def codifferential(self):
        if set(self.variance_spec) - {'l'}:
            raise ValueError("Codifferential is implemented for covariant forms only.")
        n = self.chart.dimension
        p = len(self.variance_spec)
        if p == 0:
            return ScalarField(self.chart, sp.Integer(0))
        star1 = self.hodge_star()
        star2 = star1.exterior_derivative().hodge_star()
        sign = (-1) ** (n * p + n + 1)
        if len(star2.variance_spec) == 0:
            return ScalarField(self.chart, canonical_simplify(sign * star2.components[()], final=True))
        out = tnf_map_array(star2.components, lambda entry: sign * entry)
        return TensorField(self.chart, out, star2.variance_spec)

    def contract(self, slot1: int, slot2: int) -> "TensorField | ScalarField":
        if slot1 == slot2:
            raise ValueError("Cannot contract a tensor slot with itself.")
        if self.variance_spec[slot1] == self.variance_spec[slot2]:
            raise ValueError("Contraction requires one upper and one lower index.")
        dim = self.chart.dimension
        rank = len(self.variance_spec)
        remaining_slots = [i for i in range(rank) if i not in {slot1, slot2}]
        if not remaining_slots:
            total = 0
            for a in range(dim):
                idx = [0] * rank
                idx[slot1] = a
                idx[slot2] = a
                total += self.components[tuple(idx)]
            return ScalarField(self.chart, canonical_simplify(total))
        new_spec = ''.join(self.variance_spec[i] for i in remaining_slots)
        out = tnf_build_array((dim,) * len(remaining_slots), lambda new_indices: canonical_simplify(sum(
            self.components[_reinsert_contracted_indices(rank, slot1, slot2, new_indices, a)]
            for a in range(dim)
        )))
        assumptions = self.chart.assumptions(self.chart.symbols())
        out = _apply_assumptions_array(out, assumptions)
        return TensorField(self.chart, out, new_spec)

    def divergence(self):
        if 'u' not in self.variance_spec:
            raise ValueError("Tensor divergence requires at least one contravariant slot.")
        first_u = self.variance_spec.index('u')
        nabla = self.covariant_derivative()
        rank = len(self.variance_spec)
        dim = self.chart.dimension
        remaining_spec = list(self.variance_spec) + ['l']
        last_slot = len(remaining_spec) - 1
        if remaining_spec[first_u] != 'u' or remaining_spec[last_slot] != 'l':
            raise ValueError('Unexpected divergence variance layout.')
        remaining = [i for i in range(rank + 1) if i not in {first_u, last_slot}]
        new_spec = ''.join(remaining_spec[i] for i in remaining)
        if not remaining:
            total = 0
            for a in range(dim):
                idx = [0] * (rank + 1)
                idx[first_u] = a
                idx[last_slot] = a
                total += nabla.components[tuple(idx)]
            return ScalarField(self.chart, _simplify_under_assumptions(canonical_simplify(total), self.chart.assumptions(self.chart.symbols())))
        out = tnf_build_array((dim,) * len(remaining), lambda new_idx: canonical_simplify(sum(
            nabla.components[_reinsert_divergence_indices(rank + 1, first_u, last_slot, new_idx, a)]
            for a in range(dim)
        )))
        assumptions = self.chart.assumptions(self.chart.symbols())
        out = _apply_assumptions_array(out, assumptions)
        if len(new_spec) == 1:
            variance = 'contravariant' if new_spec == 'u' else 'covariant'
            return VectorField(self.chart, _matrix_from_array_rank1(out), variance)
        return TensorField(self.chart, out, new_spec)

    def connection_laplacian(self):
        first = self.covariant_derivative()
        second = first.covariant_derivative()
        ginv = self.chart.inverse_metric(self.chart.symbols())
        if ginv is None:
            raise ValueError("Chart does not define a metric.")
        rank = len(self.variance_spec)
        dim = self.chart.dimension
        out = tnf_build_array((dim,) * rank, lambda base_indices: canonical_simplify(sum(
            ginv[a, b] * second.components[tuple(base_indices) + (a, b)]
            for a in range(dim)
            for b in range(dim)
        )))
        assumptions = self.chart.assumptions(self.chart.symbols())
        out = _apply_assumptions_array(out, assumptions)
        if rank == 0:
            return ScalarField(self.chart, out[()])
        if rank == 1:
            return VectorField(self.chart, tnf_column_from_entries(out[(i,)] for i in range(dim)), "contravariant" if self.variance_spec == 'u' else 'covariant')
        return TensorField(self.chart, out, self.variance_spec)

    def ricci_laplacian(self):
        base = self.connection_laplacian()
        ric = self.chart.ricci_tensor(self.chart.symbols())
        ginv = self.chart.inverse_metric(self.chart.symbols())
        if ric is None or ginv is None:
            raise ValueError("Chart does not define Ricci data.")
        if len(self.variance_spec) == 1:
            if self.variance_spec == 'u':
                return self.as_vector().ricci_laplacian().as_tensor()
            if self.variance_spec == 'l':
                return self.as_vector().ricci_laplacian().as_tensor()
        return base

    def lichnerowicz_laplacian(self):
        base = self.connection_laplacian()
        riem = self.chart.riemann_tensor(self.chart.symbols())
        ric = self.chart.ricci_tensor(self.chart.symbols())
        g = self.chart.metric(self.chart.symbols())
        ginv = self.chart.inverse_metric(self.chart.symbols())
        if riem is None or ric is None or g is None or ginv is None:
            raise ValueError("Chart does not define curvature data.")
        if len(self.variance_spec) == 2 and self.variance_spec == 'll':
            dim = self.chart.dimension
            out = tnf_build_array((dim, dim), lambda ij: _lichnerowicz_ll_entry(base.components, self.components, ric, g, riem, ij, dim))
            return TensorField(self.chart, _apply_assumptions_array(out, self.chart.assumptions(self.chart.symbols())), 'll')
        return base

    def as_vector(self) -> VectorField:
        if len(self.variance_spec) != 1:
            raise ValueError("Tensor rank must be 1 to convert to VectorField.")
        variance = 'contravariant' if self.variance_spec == 'u' else 'covariant'
        return VectorField(self.chart, _matrix_from_array_rank1(self.components), variance)

    def curl(self) -> VectorField:
        """Curl for rank-1 tensor fields, interpreted as vectors or 1-forms."""
        if len(self.variance_spec) != 1:
            raise ValueError("curl is implemented for rank-1 tensor fields only.")
        return self.as_vector().curl()

    def to_sympy_components(self):
        if len(self.variance_spec) == 1:
            return self.as_vector().components.to_sympy()
        return self.components.to_sympy()

    @property
    def tensor_type(self) -> tuple[int, int]:
        """Return the abstract tensor type (number of upper slots, number of lower slots)."""
        return (self.variance_spec.count("u"), self.variance_spec.count("l"))

    @classmethod
    def from_vector_field(cls, vector: VectorField) -> "TensorField":
        """Lift a vector field into the general tensor-field representation."""
        return vector.as_tensor()

    def transform(self, mapping: CoordinateMap) -> "TensorField":
        if mapping.source != self.chart:
            raise ValueError("Mapping source must match field chart.")
        if mapping.inverse_exprs_func is None:
            raise ValueError("Inverse mapping is required to express the transformed tensor in target coordinates.")

        dim = self.chart.dimension
        target_coords = mapping.target.symbols()
        source_coords = mapping.source.symbols()
        inv_exprs = mapping.inverse_mapping_exprs(target_coords)
        subs = dict(zip(source_coords, inv_exprs))

        J = mapping.jacobian(source_coords).subs(subs)
        Jinv = mapping.inverse_jacobian(target_coords)

        rank = len(self.variance_spec)
        new = tnf_build_array((dim,) * rank, lambda new_indices: canonical_simplify(sum(
            sp.prod(
                J[new_indices[slot], old_indices[slot]] if kind == "u" else Jinv[old_indices[slot], new_indices[slot]]
                for slot, kind in enumerate(self.variance_spec)
            ) * canonical_simplify(self.components[old_indices].subs(subs))
            for old_indices in product(range(dim), repeat=rank)
        )))

        return TensorField(mapping.target, new, self.variance_spec)

    def lower_index(self, slot: int) -> "TensorField":
        if self.variance_spec[slot] != "u":
            raise ValueError("Selected index is not contravariant.")
        metric = self.chart.metric(self.chart.symbols())
        if metric is None:
            raise ValueError("Chart does not define a metric.")
        return self._apply_metric(slot, metric, to_kind="l")

    def raise_index(self, slot: int) -> "TensorField":
        if self.variance_spec[slot] != "l":
            raise ValueError("Selected index is not covariant.")
        metric_inv = self.chart.inverse_metric(self.chart.symbols())
        if metric_inv is None:
            raise ValueError("Chart does not define a metric.")
        return self._apply_metric(slot, metric_inv, to_kind="u")

    def _apply_metric(self, slot: int, metric_matrix, to_kind: str) -> "TensorField":
        dim = self.chart.dimension
        rank = len(self.variance_spec)
        out = tnf_build_array((dim,) * rank, lambda new_indices: canonical_simplify(sum(
            metric_matrix[new_indices[slot], old_val] * self.components[tuple(list(new_indices[:slot]) + [old_val] + list(new_indices[slot+1:]))]
            for old_val in range(dim)
        )))
        new_spec = list(self.variance_spec)
        new_spec[slot] = to_kind
        assumptions = self.chart.assumptions(self.chart.symbols())
        out = _apply_assumptions_array(out, assumptions)
        return TensorField(self.chart, out, "".join(new_spec))
