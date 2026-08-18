import sympy as sp

from tensoratlas import (
    TensorField,
    coframe_basis,
    connection_one_forms,
    coordinate_chart,
    curvature_two_forms,
    first_structure_equation_residuals,
    frame_basis,
    frame_metric,
    second_structure_equation_residuals,
    torsion_two_forms,
    wedge,
    hodge_star,
)


def _polar_orthonormal_frame(chart):
    def tangent_to_chart(coords):
        r, theta = coords
        return sp.Matrix([[1, 0], [0, 1 / r]])

    def coframe_to_chart(coords):
        r, theta = coords
        return sp.Matrix([[1, 0], [0, r]])

    frame = frame_basis("e", chart, tangent_to_chart, orthonormal=True, dual_name="theta")
    coframe = coframe_basis("theta", chart, coframe_to_chart, orthonormal=True, dual_name="e")
    return frame, coframe


def test_frame_metric_and_connection_forms_on_polar_orthonormal_frame():
    polar = coordinate_chart("Euclidean", "Polar", 2)
    r, theta = polar.symbols()
    frame, _ = _polar_orthonormal_frame(polar)
    metric = frame_metric(frame)
    assert metric[0, 0] == 1 and metric[1, 1] == 1
    omega = connection_one_forms(frame)
    assert sp.simplify(omega[0][1][1] + 1 / r) == 0
    assert sp.simplify(omega[1][0][0]) == 0


def test_frame_based_curvature_and_structure_equations_return_consistent_shapes():
    polar = coordinate_chart("Euclidean", "Polar", 2)
    frame, _ = _polar_orthonormal_frame(polar)
    curvature = curvature_two_forms(frame)
    torsion = torsion_two_forms(frame)
    first = first_structure_equation_residuals(frame)
    second = second_structure_equation_residuals(frame)
    assert isinstance(curvature, dict)
    assert isinstance(torsion, dict)
    assert len(first) == 2
    assert len(second) == 2 and len(second[0]) == 2
    assert (0, 1) in curvature


def test_hodge_star_and_wedge_stay_consistent_in_cartesian_plane():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    dx = TensorField(cart, sp.MutableDenseNDimArray([1, 0]), "l")
    dy = TensorField(cart, sp.MutableDenseNDimArray([0, 1]), "l")
    area = wedge(dx, dy)
    assert area.components[0, 1] == 1
    star_dx = hodge_star(dx)
    assert tuple(star_dx.components) == (0, 1)
