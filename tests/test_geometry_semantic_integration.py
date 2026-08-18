from __future__ import annotations

import sympy as sp

from tensoratlas import (
    coordinate_chart,
    orthonormal_tangent_basis,
    clifford_algebra,
    spin_connection,
    exterior_form_nf,
    unified_tensor_normal_form,
    compare_unified_normal_forms,
    semantic_ir_for_object,
    canonical_semantic_form,
)


def test_exterior_forms_use_canonical_semantic_core():
    x = sp.Symbol("x")
    left = exterior_form_nf({(1, 0): x}, dimension=3, basis_labels=("dx", "dy", "dz"))
    right = exterior_form_nf({(0, 1): -x}, dimension=3, basis_labels=("dx", "dy", "dz"))

    left_nf = unified_tensor_normal_form(left)
    right_nf = unified_tensor_normal_form(right)

    assert left_nf.layer == "exterior"
    assert right_nf.layer == "exterior"
    assert left_nf.semantic_form is not None
    assert left_nf.key == right_nf.key
    assert compare_unified_normal_forms(left, right)


def test_spin_connection_uses_semantic_core_stably():
    chart = coordinate_chart("Euclidean", "Cartesian", 2)
    frame = orthonormal_tangent_basis(chart)
    conn1 = spin_connection(frame)
    conn2 = spin_connection(frame)

    sem1 = canonical_semantic_form(semantic_ir_for_object(conn1))
    sem2 = canonical_semantic_form(semantic_ir_for_object(conn2))

    assert sem1.key == sem2.key
    assert sem1.ir.layer == "spin_connection"


def test_clifford_and_frame_objects_have_semantic_layer():
    chart = coordinate_chart("Euclidean", "Cartesian", 2)
    frame = orthonormal_tangent_basis(chart)
    cliff = clifford_algebra(2, (2, 0, 0), basis_labels=("1", "2"))

    f_nf = unified_tensor_normal_form(frame)
    c_nf = unified_tensor_normal_form(cliff)

    assert f_nf.semantic_form is not None
    assert c_nf.semantic_form is not None
    assert f_nf.layer == "frame"
    assert c_nf.layer == "clifford"
