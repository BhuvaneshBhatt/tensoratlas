from fractions import Fraction

from tensoratlas.core import (
    Basis,
    ComponentTensor,
    CoordinateSystem,
    DifferentialForm,
    Manifold,
    TensorHead,
    christoffel_variation,
    covariant_derivative,
    echelon_identity_basis,
    form_inner_product,
    integrate_by_parts_once_with_boundary,
    is_closed,
    metric_gauge_variation,
    normal_form,
    pullback_from_coordinate_functions,
    stress_energy_from_metric_variation,
    wedge_hodge_inner_product,
    weyl_ricci_scalar_decomposition_identity,
)


def test_noncoordinate_exterior_derivative_uses_structure_coefficients():
    manifold = Manifold("G", 2)
    coords = CoordinateSystem("u", manifold, ("u", "v"))
    basis = Basis("e", coords, kind="noncoordinate", structure_coefficients={(0, 0, 1): 3})
    theta0 = DifferentialForm.basis_one_form(basis, 0)
    dtheta0 = theta0.exterior_derivative()
    assert dtheta0.component(0, 1) == -3


def test_form_inner_product_and_closed_helper():
    manifold = Manifold("M", 2)
    coords = CoordinateSystem("x", manifold, ("x", "y"))
    basis = coords.coordinate_basis()
    metric = ComponentTensor(TensorHead.metric("g", basis.index_type), basis, {(0, 0): 1, (1, 1): 1})
    dx = DifferentialForm.basis_one_form(basis, 0)
    dy = DifferentialForm.basis_one_form(basis, 1)
    assert form_inner_product(dx, dx, metric) == 1
    assert form_inner_product(dx, dy, metric) == 0
    assert wedge_hodge_inner_product(dx, dx, metric).degree == 2
    assert is_closed(dx)


def test_coordinate_function_pullback_of_one_form():
    import sympy as sp
    manifold = Manifold("M", 2)
    old = CoordinateSystem("x", manifold, ("x", "y"))
    new = CoordinateSystem("u", manifold, ("u", "v"))
    old_basis = old.coordinate_basis()
    new_basis = new.coordinate_basis()
    x, y = sp.symbols("x y")
    u, v = sp.symbols("u v")
    dx = DifferentialForm.basis_one_form(old_basis, 0)
    pulled = pullback_from_coordinate_functions(dx, new_basis, (u**2, v), (u, v))
    assert sp.simplify(pulled.component(0) - 2*u) == 0
    assert pulled.component(1) == 0


def test_echelon_identity_basis_and_normal_form():
    manifold = Manifold("M", 3)
    itype = manifold.index_type("T")
    a, b = itype.indices("a b", variance="up")
    S = TensorHead("S", (itype,))
    T = TensorHead("T", (itype,))
    U = TensorHead("U", (itype,))
    id1 = (S(a) + T(a))
    id2 = (T(a) + U(a))
    basis = echelon_identity_basis([
        __import__("tensoratlas.core", fromlist=["identity_from_expression"]).identity_from_expression(id1),
        __import__("tensoratlas.core", fromlist=["identity_from_expression"]).identity_from_expression(id2),
    ])
    reduced = basis.reduce(S(a))
    assert "U" in repr(reduced)
    assert repr(normal_form(S(a), basis.identities)) == repr(reduced)


def test_weyl_ricci_scalar_identity_reduces_riemann_template():
    manifold = Manifold("M", 4)
    itype = manifold.index_type("T")
    a, b, c, d = itype.indices("a b c d", variance="down")
    R = TensorHead.riemann("R", itype, variance=("down", "down", "down", "down"))
    C = TensorHead.weyl("C", itype, variance=("down", "down", "down", "down"))
    Ric = TensorHead.ricci("Ric", itype)
    scal = TensorHead.scalar_curvature("Scal")
    g = TensorHead.metric("g", itype)
    identity = weyl_ricci_scalar_decomposition_identity(R, C, Ric, scal, g, (a, b, c, d), dimension=4)
    reduced = normal_form(R(a, b, c, d), [identity])
    text = repr(reduced)
    assert "C(" in text and "Ric(" in text and "Scal" in text


def test_ibp_boundary_and_metric_gauge_variation_helpers():
    manifold = Manifold("M", 4)
    itype = manifold.index_type("T")
    a, b = itype.indices("a b", variance="down")
    phi = TensorHead("phi", (itype,), variance=("down",))
    dphi = TensorHead("dphi", (itype,), variance=("down",))
    D = covariant_derivative("D", itype)
    varied = D.derivative_factor(next(iter(dphi(a).terms)).factors[0], b)
    result = integrate_by_parts_once_with_boundary(varied, D, dphi)
    assert result.boundary_terms
    xi = TensorHead("xi", (itype,), variance=("down",))
    gauge = metric_gauge_variation(xi, D, a, b)
    assert "Dxi" in repr(gauge)


def test_stress_energy_density_helper():
    rho = TensorHead.scalar("rho")
    manifold = Manifold("M", 2)
    itype = manifold.index_type("T")
    a = itype.index("a", variance="up")
    V = TensorHead("V", (itype,))
    expr = stress_energy_from_metric_variation(rho, V(a))
    assert repr(expr).startswith("-2*")
