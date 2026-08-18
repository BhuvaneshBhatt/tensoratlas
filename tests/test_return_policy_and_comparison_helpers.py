from __future__ import annotations

import sympy as sp

from tensoratlas import (
    simplify_object,
    coordinate_chart,
    coordinate_map,
    list_public_return_policies,
    matrix_equal,
    matrix_is_zero,
    paired_public_api,
    public_return_policy,
    returns_sympy,
    returns_tnormal_forms,
    scalar_equal,
    scalar_is_zero,
    tensor_equal,
    tensor_is_zero,
)
from tensoratlas.normal_forms import tnf_build_array


def test_public_return_policy_pairs_are_consistent():
    pairs = {item.name: item.paired_with for item in list_public_return_policies()}
    assert returns_sympy("CoordinateMap.jacobian")
    assert returns_tnormal_forms("CoordinateMap.jacobian_tnf")
    assert paired_public_api("CoordinateMap.jacobian") == "CoordinateMap.jacobian_tnf"
    assert pairs["frame_metric"] == "frame_metric_tnf"
    assert public_return_policy("basis_transformation_matrix_tnf").returns == "tnf"


def test_simplification_levels_and_simplify_object_cover_scalars_and_matrices():
    x = sp.Symbol("x")
    structural = simplify_object((x**2 - 1) / (x - 1), level="cheap")
    presentation = simplify_object((x**2 - 1) / (x - 1), level="strong")
    assert structural == x + 1
    assert presentation == x + 1
    matrix = sp.Matrix([[(x**2 - 1) / (x - 1), 0], [0, 1]])
    cleaned = simplify_object(matrix, level="strong")
    assert cleaned == sp.Matrix([[x + 1, 0], [0, 1]])


def test_scalar_matrix_and_tensor_comparison_helpers():
    x = sp.Symbol("x")
    assert scalar_equal((x**2 - 1) / (x - 1), x + 1)
    assert scalar_is_zero(sp.sin(sp.pi))
    left = sp.Matrix([[(x**2 - 1) / (x - 1), 0], [0, 1]])
    right = sp.Matrix([[x + 1, 0], [0, 1]])
    assert matrix_equal(left, right)
    assert matrix_is_zero(sp.Matrix([[0, 0], [0, 0]]))
    arr1 = tnf_build_array((2, 2), lambda idx: left[idx])
    arr2 = tnf_build_array((2, 2), lambda idx: right[idx])
    assert tensor_equal(arr1, arr2)
    assert tensor_is_zero(tnf_build_array((2, 2), lambda idx: sp.Integer(0)))


def test_cross_chart_pairing_remains_consistent_under_policy_helpers():
    polar = coordinate_chart("Euclidean", "Polar", 2)
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    mapping = coordinate_map(polar, cart)
    jac_tnf = mapping.jacobian_tnf()
    jac_sympy = mapping.jacobian()
    assert matrix_equal(jac_tnf, jac_sympy)
