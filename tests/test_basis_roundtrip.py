from __future__ import annotations

import sympy as sp

from tensoratlas import (
    basis_transformation_matrix,
    basis_transformation_matrix_tnf,
    coordinate_chart,
    coordinate_map,
    cotangent_basis,
    gram_schmidt_frame,
    gram_schmidt_frame_tnf,
    matrix_equal,
    orthonormal_tangent_basis,
    tangent_basis,
)


def test_cross_chart_basis_transformations_match_sympy_and_tnf_paths():
    polar = coordinate_chart("Euclidean", "Polar", 2)
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    coordinate_map(polar, cart)
    sympy_mat = basis_transformation_matrix(tangent_basis(polar), tangent_basis(cart))
    tnf_mat = basis_transformation_matrix_tnf(tangent_basis(polar), tangent_basis(cart))
    assert matrix_equal(sympy_mat, tnf_mat)
    sympy_cot = basis_transformation_matrix(cotangent_basis(polar), cotangent_basis(cart))
    tnf_cot = basis_transformation_matrix_tnf(cotangent_basis(polar), cotangent_basis(cart))
    assert matrix_equal(sympy_cot, tnf_cot)
    ortho = basis_transformation_matrix_tnf(orthonormal_tangent_basis(polar), tangent_basis(cart))
    assert ortho.shape == (2, 2)


def test_gram_schmidt_paths_agree_for_symbolic_input_frame():
    polar = coordinate_chart("Euclidean", "Polar", 2)
    r, theta = polar.symbols()
    frame = sp.Matrix([[1, 0], [0, 1 / r]])
    sympy_frame = gram_schmidt_frame(polar, frame)
    tnf_frame = gram_schmidt_frame_tnf(polar, frame)
    assert matrix_equal(sympy_frame, tnf_frame)
