import pytest
import sympy as sp

from tensoratlas.relativity import geodesic_equations, geodesic_rhs, minkowski_metric, two_sphere_metric


def test_geodesic_equations_returns_one_equation_per_coordinate():
    equations = geodesic_equations(two_sphere_metric())
    assert len(equations) == 2


def test_geodesic_rhs_rejects_wrong_state_length():
    x0, v0 = sp.symbols("x0 v0")
    with pytest.raises(ValueError, match="state length"):
        geodesic_rhs(two_sphere_metric(), (x0, v0))


def test_geodesic_rhs_accepts_numpy_module_name():
    model = minkowski_metric(2, names=("t", "x"))
    t, x, vt, vx = sp.symbols("t x vt vx")
    rhs = geodesic_rhs(model, (t, x, vt, vx), modules="numpy")
    assert rhs(0.0, 1.0, 2.0, 3.0) == (2.0, 3.0, 0, 0)
