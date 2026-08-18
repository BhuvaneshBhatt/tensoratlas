"""Structural variational and perturbative tensor workflows.

The routines in this module deliberately operate on the backend-light semantic
kernel.  They provide bounded product-rule variation, integration by parts with
optional boundary bookkeeping, common metric-variation identities, and fixed
order tensor perturbation expansion.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb
from typing import Iterable, Mapping, Sequence

from .derivatives import DerivativeOperator
from .indices import AbstractIndex, IndexType
from .manifolds import TensorKernelError
from .tensor_expr import TensorExpr, TensorFactor, TensorTerm, canonicalize
from .tensor_heads import TensorHead


def _fresh_index(base: str, index_type: IndexType, used: set[tuple[str, IndexType]], *, variance: str = "up") -> AbstractIndex:
    counter = 1
    while True:
        name = f"{base}{counter}"
        if (name, index_type) not in used:
            return AbstractIndex(name, index_type, variance)
        counter += 1


def _used_indices(expr: TensorExpr | TensorFactor) -> set[tuple[str, IndexType]]:
    out: set[tuple[str, IndexType]] = set()
    factors: list[TensorFactor] = []
    if isinstance(expr, TensorFactor):
        factors = [expr]
    else:
        for term in expr.terms:
            factors.extend(term.factors)
    for factor in factors:
        for idx in factor.indices:
            out.add((idx.name, idx.index_type))
    return out


def _single_factor(expr: TensorExpr) -> TensorFactor:
    canonical = canonicalize(expr)
    if len(canonical.terms) != 1 or len(canonical.terms[0].factors) != 1 or canonical.terms[0].coefficient != 1:
        raise TensorKernelError("Expected expression containing exactly one tensor factor.")
    return canonical.terms[0].factors[0]


def _copy_head(head: TensorHead, *, role: str | None = None) -> TensorHead:
    return TensorHead(
        head.name,
        head.index_types,
        symmetry=head.symmetry.kind,
        variance=head.variance,
        commutative=head.commutative,
        role=role or head.role,
    )


def variation_head(field: TensorHead, *, prefix: str = "d") -> TensorHead:
    """Return a tensor head representing the variation of ``field``."""
    return TensorHead(
        f"{prefix}{field.name}",
        field.index_types,
        symmetry=field.symmetry.kind,
        role=field.role,
        commutative=field.commutative,
        variance=field.variance,
    )


def vary(
    expression: TensorExpr,
    replacements: Mapping[TensorHead, TensorHead | TensorExpr] | None = None,
    *,
    prefix: str = "d",
) -> TensorExpr:
    """Take a first structural variation using the product rule."""
    expr = canonicalize(expression)
    mapping = dict(replacements or {})
    vary_all = replacements is None
    terms: list[TensorTerm] = []
    for term in expr.terms:
        for pos, factor in enumerate(term.factors):
            replacement = mapping.get(factor.head)
            if replacement is None and not vary_all:
                continue
            if replacement is None:
                replacement_expr = variation_head(factor.head, prefix=prefix)(*factor.indices)
            elif isinstance(replacement, TensorHead):
                replacement_expr = replacement(*factor.indices)
            else:
                replacement_expr = replacement
            for rterm in replacement_expr.terms:
                factors = term.factors[:pos] + rterm.factors + term.factors[pos + 1 :]
                terms.append(TensorTerm(term.coefficient * rterm.coefficient, factors))
    return canonicalize(TensorExpr(tuple(terms)))


def variation_of_factor(factor: TensorFactor, replacement_head: TensorHead | None = None, *, prefix: str = "d") -> TensorExpr:
    """Return the structural variation of a single tensor factor."""
    return (replacement_head or variation_head(factor.head, prefix=prefix))(*factor.indices)


def inverse_metric_variation(
    inverse_metric_factor: TensorFactor,
    metric: TensorHead,
    metric_variation: TensorHead | None = None,
) -> TensorExpr:
    """Return ``delta g^ab = -g^ac g^bd delta g_cd`` structurally."""
    if inverse_metric_factor.head.role != "inverse_metric" or len(inverse_metric_factor.indices) != 2:
        raise TensorKernelError("Expected an inverse-metric factor with two contravariant indices.")
    if metric.role != "metric":
        raise TensorKernelError("Expected a covariant metric head.")
    left, right = inverse_metric_factor.indices
    if not (left.is_up and right.is_up):
        raise TensorKernelError("Inverse metric variation requires upper indices.")
    used = _used_indices(inverse_metric_factor)
    c = _fresh_index("v", left.index_type, used, variance="up")
    used.add((c.name, c.index_type))
    d = _fresh_index("v", right.index_type, used, variance="up")
    dg = metric_variation or variation_head(metric)
    inert_inverse = TensorHead(inverse_metric_factor.head.name, inverse_metric_factor.head.index_types, symmetry="none", variance=(None, None), role="tensor")
    inert_dg = TensorHead(dg.name, dg.index_types, symmetry="none", variance=(None, None), role="tensor")
    return canonicalize(-(inert_inverse(left, c) * inert_inverse(right, d) * inert_dg(-c, -d)))


def metric_density_variation(
    density: TensorHead,
    inverse_metric: TensorHead,
    metric_variation: TensorHead,
    index_type: IndexType,
) -> TensorExpr:
    """Return ``delta sqrt(|g|) = 1/2 sqrt(|g|) g^ab delta g_ab``."""
    if density.rank != 0:
        raise TensorKernelError("Metric density head must be scalar/rank zero.")
    if inverse_metric.role != "inverse_metric" or inverse_metric.index_types[0] != index_type:
        raise TensorKernelError("Expected an inverse metric for the supplied index type.")
    a = AbstractIndex("v1", index_type, "up")
    b = AbstractIndex("v2", index_type, "up")
    inert_inverse = TensorHead(inverse_metric.name, inverse_metric.index_types, symmetry="none", variance=(None, None), role="tensor")
    inert_metric_variation = TensorHead(metric_variation.name, metric_variation.index_types, symmetry="none", variance=(None, None), role="tensor")
    return canonicalize(Fraction(1, 2) * density() * inert_inverse(a, b) * inert_metric_variation(-a, -b))


def derivative_of_expression(operator: DerivativeOperator, expression: TensorExpr, derivative_index: AbstractIndex) -> TensorExpr:
    """Apply a structural derivative to an expression."""
    return operator.apply(expression, derivative_index, expand_products=True)


@dataclass(frozen=True, slots=True)
class BoundaryTerm:
    """A total-derivative term produced by integration by parts."""

    derivative_index: AbstractIndex
    expression: TensorExpr


@dataclass(frozen=True, slots=True)
class IntegrationByPartsResult:
    """Bulk and boundary terms from one structural integration-by-parts pass."""

    bulk: TensorExpr
    boundary_terms: tuple[BoundaryTerm, ...]

    @property
    def boundary_expression(self) -> TensorExpr:
        if not self.boundary_terms:
            return TensorExpr.zero()
        terms: list[TensorTerm] = []
        for boundary in self.boundary_terms:
            total = boundary.expression
            terms.extend(total.terms)
        return canonicalize(TensorExpr(tuple(terms)))


def integrate_by_parts_once_with_boundary(
    expression: TensorExpr,
    operator: DerivativeOperator,
    variation: TensorHead,
) -> IntegrationByPartsResult:
    """Move one derivative off a variation while retaining boundary records."""
    expr = canonicalize(expression)
    out_terms: list[TensorTerm] = []
    boundaries: list[BoundaryTerm] = []
    for term in expr.terms:
        moved = False
        for pos, factor in enumerate(term.factors):
            if factor.head.name != f"{operator.name}{variation.name}" or len(factor.indices) == 0:
                continue
            derivative_index = factor.indices[0]
            base_indices = factor.indices[1:]
            base_var = variation(*base_indices)
            rest = TensorExpr((TensorTerm(term.coefficient, term.factors[:pos] + term.factors[pos + 1 :]),))
            boundary_expr = canonicalize(rest * base_var)
            boundaries.append(BoundaryTerm(derivative_index, boundary_expr))
            differentiated_rest = operator.apply(rest, derivative_index, expand_products=True)
            rewritten = -differentiated_rest * base_var
            out_terms.extend(rewritten.terms)
            moved = True
            break
        if not moved:
            out_terms.append(term)
    return IntegrationByPartsResult(canonicalize(TensorExpr(tuple(out_terms))), tuple(boundaries))


def integrate_by_parts_once(expression: TensorExpr, operator: DerivativeOperator, variation: TensorHead) -> TensorExpr:
    """Move one derivative off a variation, discarding boundary terms."""
    return integrate_by_parts_once_with_boundary(expression, operator, variation).bulk


@dataclass(frozen=True, slots=True)
class EulerLagrangeResult:
    """Structural Euler-Lagrange split for one varied field."""

    varied_expression: TensorExpr
    integrated_expression: TensorExpr
    coefficient: TensorExpr
    boundary_terms: tuple[BoundaryTerm, ...] = ()


def coefficient_of_factor(expression: TensorExpr, factor_head: TensorHead) -> TensorExpr:
    """Extract the coefficient multiplying one occurrence of ``factor_head``."""
    expr = canonicalize(expression)
    coeff_terms: list[TensorTerm] = []
    for term in expr.terms:
        for pos, factor in enumerate(term.factors):
            if factor.head == factor_head:
                remaining = term.factors[:pos] + term.factors[pos + 1 :]
                coeff_terms.append(TensorTerm(term.coefficient, remaining))
                break
    return canonicalize(TensorExpr(tuple(coeff_terms)))


def euler_lagrange_expression(
    lagrangian: TensorExpr,
    field: TensorHead,
    operator: DerivativeOperator,
    *,
    variation: TensorHead | None = None,
    keep_boundary: bool = False,
) -> EulerLagrangeResult:
    """Return a structural Euler-Lagrange coefficient after one IBP pass."""
    delta = variation or variation_head(field)
    varied = vary(lagrangian, {field: delta})
    ibp = integrate_by_parts_once_with_boundary(varied, operator, delta)
    coeff = coefficient_of_factor(ibp.bulk, delta)
    return EulerLagrangeResult(varied, ibp.bulk, coeff, ibp.boundary_terms if keep_boundary else ())


@dataclass(frozen=True, slots=True)
class PerturbationRule:
    """A replacement ``base -> base + parameter * perturbation``."""

    base: TensorHead
    perturbation: TensorHead
    parameter: str = "eps"


@dataclass(frozen=True, slots=True)
class PerturbativeTerm:
    """One term in a perturbative expansion with an integer order."""

    order: int
    coefficient: Fraction
    factors: tuple[TensorFactor, ...]

    def to_tensor_term(self) -> TensorTerm:
        return TensorTerm(self.coefficient, self.factors)


@dataclass(frozen=True, slots=True)
class PerturbativeExpression:
    """A sparse fixed-order perturbative expansion."""

    terms: tuple[PerturbativeTerm, ...]

    def truncate(self, max_order: int) -> "PerturbativeExpression":
        return PerturbativeExpression(tuple(term for term in self.terms if term.order <= max_order))

    def order(self, order: int) -> TensorExpr:
        return canonicalize(TensorExpr(tuple(term.to_tensor_term() for term in self.terms if term.order == order)))

    def as_dict(self) -> dict[int, TensorExpr]:
        orders = sorted({term.order for term in self.terms})
        return {order: self.order(order) for order in orders}


def _cartesian_factor_choices(choices: Sequence[Sequence[tuple[int, TensorFactor]]]) -> Iterable[tuple[tuple[int, TensorFactor], ...]]:
    if not choices:
        yield ()
        return
    first, *rest = choices
    for item in first:
        for tail in _cartesian_factor_choices(rest):
            yield (item,) + tail


def perturbative_expand(expression: TensorExpr, rules: Sequence[PerturbationRule], *, max_order: int = 1) -> PerturbativeExpression:
    """Expand tensor factors under fixed-order additive perturbations."""
    expr = canonicalize(expression)
    rule_by_head = {rule.base: rule for rule in rules}
    out: list[PerturbativeTerm] = []
    for term in expr.terms:
        choices: list[list[tuple[int, TensorFactor]]] = []
        for factor in term.factors:
            rule = rule_by_head.get(factor.head)
            if rule is None:
                choices.append([(0, factor)])
            else:
                choices.append([(0, factor), (1, TensorFactor(rule.perturbation, factor.indices))])
        for selected in _cartesian_factor_choices(choices):
            order = sum(item[0] for item in selected)
            if order <= max_order:
                out.append(PerturbativeTerm(order, term.coefficient, tuple(item[1] for item in selected)))
    return PerturbativeExpression(tuple(out)).truncate(max_order)


def binomial_perturbation_coefficients(power: int, max_order: int) -> tuple[Fraction, ...]:
    """Return coefficients for ``(base + eps*perturbation)^power`` up to order."""
    if power < 0:
        raise TensorKernelError("Only non-negative integer powers are supported by this helper.")
    return tuple(Fraction(comb(power, order)) for order in range(min(power, max_order) + 1))


def christoffel_variation(
    metric_variation: TensorHead,
    inverse_metric: TensorHead,
    operator: DerivativeOperator,
    upper: AbstractIndex,
    first_lower: AbstractIndex,
    second_lower: AbstractIndex,
) -> TensorExpr:
    """Return ``delta Gamma^a_bc = 1/2 g^ad(D_b dg_cd + D_c dg_bd - D_d dg_bc)``."""
    if metric_variation.role != "metric" or inverse_metric.role != "inverse_metric":
        raise TensorKernelError("Connection variation expects metric variation and inverse metric heads.")
    if not (upper.is_up and first_lower.is_down and second_lower.is_down):
        raise TensorKernelError("Connection variation expects indices (^a, _b, _c).")
    if upper.index_type != first_lower.index_type or upper.index_type != second_lower.index_type:
        raise TensorKernelError("Connection variation indices must share one index type.")
    if operator.index_type != upper.index_type:
        raise TensorKernelError("Derivative operator index type does not match the connection indices.")
    used = {(idx.name, idx.index_type) for idx in (upper, first_lower, second_lower)}
    dummy = _fresh_index("v", upper.index_type, used, variance="up")
    left = operator.derivative_factor(_single_factor(metric_variation(second_lower, -dummy)), first_lower)
    middle = operator.derivative_factor(_single_factor(metric_variation(first_lower, -dummy)), second_lower)
    right = operator.derivative_factor(_single_factor(metric_variation(first_lower, second_lower)), -dummy)
    inert_inverse = TensorHead(inverse_metric.name, inverse_metric.index_types, symmetry="none", variance=(None, None), role="tensor")
    return canonicalize(Fraction(1, 2) * inert_inverse(upper, dummy) * (left + middle - right))


def ricci_variation(connection_variation: TensorHead, operator: DerivativeOperator, first_lower: AbstractIndex, second_lower: AbstractIndex) -> TensorExpr:
    """Return ``delta Ric_ab = D_c delta Gamma^c_ab - D_b delta Gamma^c_ac``."""
    if connection_variation.rank != 3 or connection_variation.variance != ("up", "down", "down"):
        raise TensorKernelError("Ricci variation expects a connection-variation head with variance (^a,_b,_c).")
    if not (first_lower.is_down and second_lower.is_down):
        raise TensorKernelError("Ricci variation expects two covariant free indices.")
    if first_lower.index_type != second_lower.index_type or operator.index_type != first_lower.index_type:
        raise TensorKernelError("Ricci variation index types do not match.")
    used = {(first_lower.name, first_lower.index_type), (second_lower.name, second_lower.index_type)}
    dummy = _fresh_index("v", first_lower.index_type, used, variance="up")
    term1 = _single_factor(connection_variation(dummy, first_lower, second_lower))
    term2 = _single_factor(connection_variation(dummy, first_lower, -dummy))
    return canonicalize(operator.derivative_factor(term1, -dummy) - operator.derivative_factor(term2, second_lower))


def scalar_curvature_variation(
    inverse_metric: TensorHead,
    ricci: TensorHead,
    ricci_variation_head: TensorHead,
    metric_variation: TensorHead,
    first: AbstractIndex,
    second: AbstractIndex,
) -> TensorExpr:
    """Return ``delta R = g^ab delta Ric_ab - Ric^ab delta g_ab``."""
    if inverse_metric.role != "inverse_metric" or ricci.rank != 2 or ricci_variation_head.rank != 2:
        raise TensorKernelError("Scalar-curvature variation expects inverse metric and rank-two Ricci heads.")
    if metric_variation.role != "metric":
        raise TensorKernelError("Metric variation must be a covariant metric head.")
    if not (first.is_up and second.is_up):
        raise TensorKernelError("Scalar-curvature variation helper expects two contravariant dummy seeds.")
    used = {(first.name, first.index_type), (second.name, second.index_type)}
    c = _fresh_index("v", first.index_type, used, variance="up")
    used.add((c.name, c.index_type))
    d = _fresh_index("v", first.index_type, used, variance="up")
    inert_inverse = TensorHead(inverse_metric.name, inverse_metric.index_types, symmetry="none", variance=(None, None), role="tensor")
    inert_metric_variation = TensorHead(metric_variation.name, metric_variation.index_types, symmetry="none", variance=(None, None), role="tensor")
    return canonicalize(
        inert_inverse(first, second) * ricci_variation_head(-first, -second)
        - inert_inverse(first, c) * inert_inverse(second, d) * ricci(-c, -d) * inert_metric_variation(-first, -second)
    )


def einstein_hilbert_metric_variation_density(
    density: TensorHead,
    inverse_metric: TensorHead,
    einstein: TensorHead,
    metric_variation: TensorHead,
    first: AbstractIndex,
    second: AbstractIndex,
) -> TensorExpr:
    """Return the Einstein-Hilbert bulk variation for covariant ``delta g_ab``."""
    if density.rank != 0:
        raise TensorKernelError("Density must be scalar/rank zero.")
    if inverse_metric.role != "inverse_metric" or einstein.rank != 2 or metric_variation.role != "metric":
        raise TensorKernelError("Expected inverse metric, Einstein tensor, and metric variation heads.")
    if not (first.is_up and second.is_up):
        raise TensorKernelError("Einstein-Hilbert helper expects contravariant dummy seeds.")
    used = {(first.name, first.index_type), (second.name, second.index_type)}
    c = _fresh_index("v", first.index_type, used, variance="up")
    used.add((c.name, c.index_type))
    d = _fresh_index("v", first.index_type, used, variance="up")
    inert_inverse = TensorHead(inverse_metric.name, inverse_metric.index_types, symmetry="none", variance=(None, None), role="tensor")
    inert_metric_variation = TensorHead(metric_variation.name, metric_variation.index_types, symmetry="none", variance=(None, None), role="tensor")
    return canonicalize(-density() * inert_inverse(first, c) * inert_inverse(second, d) * einstein(-c, -d) * inert_metric_variation(-first, -second))


def stress_energy_from_metric_variation(density: TensorHead, matter_variation_coefficient: TensorExpr) -> TensorExpr:
    """Return the structural stress-energy density coefficient ``-2 * coefficient``.

    This helper intentionally leaves division by the volume density to caller
    convention; the returned expression is the tensor-density numerator.
    """
    if density.rank != 0:
        raise TensorKernelError("Density must be scalar/rank zero.")
    return canonicalize(-2 * density() * matter_variation_coefficient)


def metric_gauge_variation(
    vector_field: TensorHead,
    operator: DerivativeOperator,
    first_lower: AbstractIndex,
    second_lower: AbstractIndex,
) -> TensorExpr:
    """Return the linearized diffeomorphism variation ``D_a xi_b + D_b xi_a``."""
    if vector_field.rank != 1:
        raise TensorKernelError("Gauge generator must be a rank-one covector/vector head.")
    if not (first_lower.is_down and second_lower.is_down):
        raise TensorKernelError("Metric gauge variation expects covariant indices.")
    if operator.index_type != first_lower.index_type or first_lower.index_type != second_lower.index_type:
        raise TensorKernelError("Gauge variation index types do not match.")
    xi_a = _single_factor(vector_field(first_lower))
    xi_b = _single_factor(vector_field(second_lower))
    return canonicalize(operator.derivative_factor(xi_b, first_lower) + operator.derivative_factor(xi_a, second_lower))


def expand_metric_perturbation_to_order(expression: TensorExpr, metric: TensorHead, perturbation: TensorHead, *, max_order: int) -> dict[int, TensorExpr]:
    """Return each fixed perturbative order for ``metric -> metric + eps*h``."""
    return perturbative_expand(expression, [PerturbationRule(metric, perturbation)], max_order=max_order).as_dict()
