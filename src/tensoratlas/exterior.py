from __future__ import annotations

from itertools import product

import sympy as sp

from .fields import TensorField, VectorField, _apply_assumptions_array
from .normal_forms import tnf_build_array


def exterior_derivative(form: TensorField) -> TensorField:
    """Return the exterior derivative of a covariant differential form."""
    return form.exterior_derivative()


def wedge(left: TensorField, right: TensorField) -> TensorField:
    """Return the wedge product of two covariant forms."""
    return left.wedge(right)


def interior_product(vector: VectorField, form: TensorField):
    """Contract a contravariant vector into the leading slot of a form."""
    return form.interior_product(vector)


def hodge_star(form: TensorField) -> TensorField:
    """Return the Hodge dual of a covariant p-form in an oriented Riemannian chart."""
    if set(form.variance_spec) - {"l"}:
        raise ValueError("Hodge star requires a covariant differential form.")
    coords = form.chart.symbols()
    metric_inv = form.chart.inverse_metric(coords)
    sqrtg = form.chart.sqrt_metric_det(coords)
    if metric_inv is None or sqrtg is None:
        raise ValueError("Chart does not define a metric.")
    dim = form.chart.dimension
    rank = len(form.variance_spec)
    out_rank = dim - rank
    out_shape = (dim,) * out_rank
    out = tnf_build_array(out_shape, lambda out_idx: _hodge_star_entry(form, metric_inv, sqrtg, rank, dim, out_idx))
    assumptions = form.chart.assumptions(coords)
    return TensorField(form.chart, _apply_assumptions_array(out, assumptions), 'l' * out_rank)


def _hodge_star_entry(form: TensorField, metric_inv, sqrtg, rank: int, dim: int, out_idx):
    total = sp.Integer(0)
    for lowered_idx in product(range(dim), repeat=rank) if rank else [()]:
        raised_total = sp.Integer(0)
        for raised_idx in product(range(dim), repeat=rank) if rank else [()]:
            weight = sp.Integer(1)
            for pos in range(rank):
                weight *= metric_inv[lowered_idx[pos], raised_idx[pos]]
            comp = form.components[raised_idx] if rank else form.components[()]
            raised_total += weight * comp
        total += sp.LeviCivita(*(lowered_idx + out_idx)) * raised_total
    return sp.simplify(sqrtg * total / sp.factorial(rank))


def codifferential(form: TensorField) -> TensorField:
    """Return the codifferential of a covariant form using d and Hodge star."""
    dim = form.chart.dimension
    rank = len(form.variance_spec)
    if rank == 0:
        raise ValueError("Codifferential is defined on positive-degree forms.")
    result = hodge_star(exterior_derivative(hodge_star(form)))
    sign = (-1) ** (dim * (rank + 1) + 1)
    out = result.components.applyfunc(lambda entry: sp.simplify(sign * entry))
    assumptions = form.chart.assumptions(form.chart.symbols())
    return TensorField(form.chart, _apply_assumptions_array(out, assumptions), result.variance_spec)


def de_rham_laplacian(form: TensorField) -> TensorField:
    """Return the Hodge Laplacian Δ = dδ + δd on covariant forms."""
    terms = []
    if len(form.variance_spec) > 0:
        terms.append(exterior_derivative(codifferential(form)))
    if len(form.variance_spec) < form.chart.dimension:
        terms.append(codifferential(exterior_derivative(form)))
    if not terms:
        raise ValueError("The zero-dimensional Hodge Laplacian is not defined for this form.")
    out = terms[0].components
    for term in terms[1:]:
        out = out + term.components
    out = out.applyfunc(sp.simplify)
    assumptions = form.chart.assumptions(form.chart.symbols())
    return TensorField(form.chart, _apply_assumptions_array(out, assumptions), form.variance_spec)
