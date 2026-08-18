import sympy as sp

from tensoratlas.relativity import (
    christoffel_component,
    christoffel_symbols,
    flrw_metric,
    nonzero_components,
    ricci_tensor,
    scalar_curvature,
    schwarzschild_metric,
    two_sphere_metric,
)


def test_two_sphere_scalar_curvature():
    model = two_sphere_metric()
    R = model.parameters[0]
    ric = ricci_tensor(model)
    scalar = scalar_curvature(model, ricci=ric)
    assert sp.simplify(scalar - 2 / R**2) == 0


def test_metric_model_inverse_metric_is_cached():
    model = two_sphere_metric()
    assert model.inverse_metric is model.inverse_metric


def test_schwarzschild_ricci_vanishes():
    ric = ricci_tensor(schwarzschild_metric())
    assert nonzero_components(ric) == {}


def test_flrw_representative_christoffel_component():
    model = flrw_metric()
    t = model.coordinates[0]
    a = sp.Function("a")(t)
    assert sp.simplify(christoffel_component(model, 1, 0, 1) - sp.diff(a, t) / a) == 0


def test_selective_simplification_can_be_disabled():
    model = two_sphere_metric()
    gamma = christoffel_symbols(model, simplify=False)
    assert len(gamma) == 2
