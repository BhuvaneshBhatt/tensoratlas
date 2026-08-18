
from __future__ import annotations

import sympy as sp

from tensoratlas import (
    clifford_algebra,
    gamma_string,
    semantic_equivalent_objects,
    tangent_basis,
    cotangent_basis,
    transformed_basis,
    basis_transformation_matrix,
    basis_roundtrip_report,
)
from tensoratlas.semantic_ops import GammaStringExpr
from tensoratlas.semantic_rewrite import semantic_match, svar
from tensoratlas.tensor_core import TensorObject
from tensoratlas.charts import get_chart
from tensoratlas.mappings import CoordinateMap
from tensoratlas.tensor_indices import TensorIndex, IndexedTensor, IndexedTensorExpr, alpha_rename_dummies


def test_gamma_string_semantic_equivalence():
    cliff = clifford_algebra(3, (3, 0, 0))
    a = gamma_string(cliff, [0, 0])
    b = sp.Integer(1)
    assert semantic_equivalent_objects(a, b)


def test_indexed_dummy_renaming_match():
    chart = get_chart("Euclidean", "Cartesian", 2)
    tb = tangent_basis(chart)
    cb = cotangent_basis(chart)
    T = TensorObject(chart=chart, components=sp.MutableDenseNDimArray.zeros(2,2), variance_spec="ul", slot_bases=(tb, cb), name="T")
    expr1 = IndexedTensor(T, (TensorIndex("i", "u"), TensorIndex("i", "l")))
    expr2 = IndexedTensor(T, (TensorIndex("j", "u"), TensorIndex("j", "l")))
    pat = svar("x")
    env = semantic_match(expr1, pat)
    assert env is not None
    assert semantic_equivalent_objects(env["x"], expr2)


def test_indexed_expr_semantic_equivalence_under_dummy_renaming():
    chart = get_chart("Euclidean", "Cartesian", 2)
    tb = tangent_basis(chart)
    cb = cotangent_basis(chart)
    T = TensorObject(chart=chart, components=sp.MutableDenseNDimArray.zeros(2,2), variance_spec="ul", slot_bases=(tb, cb), name="T")
    A = IndexedTensor(T, (TensorIndex("i", "u"), TensorIndex("i", "l")))
    B = IndexedTensor(T, (TensorIndex("j", "u"), TensorIndex("j", "l")))
    e1 = IndexedTensorExpr("add", (A, A))
    e2 = IndexedTensorExpr("add", (B, B))
    assert semantic_equivalent_objects(e1, e2)


def test_coordinate_basis_has_identity_transform():
    chart = get_chart("Euclidean", "Cartesian", 2)
    tb = tangent_basis(chart)
    cb = cotangent_basis(chart)
    assert basis_transformation_matrix(tb, tb) == sp.eye(2)
    assert basis_transformation_matrix(cb, cb) == sp.eye(2)


def test_transformed_basis_preserves_transform_callable():
    x, y = sp.symbols("x y", real=True)
    u, v = sp.symbols("u v", real=True)
    source = get_chart("Euclidean", "Cartesian", 2)
    target = get_chart("Euclidean", "Cartesian", 2)
    mapping = CoordinateMap(source, target, lambda c: (c[0] + 1, c[1] - 2), inverse_exprs_func=lambda c: (c[0] - 1, c[1] + 2))
    tb = tangent_basis(source)
    moved = transformed_basis(tb, mapping)
    rep = basis_roundtrip_report(moved, tangent_basis(target))
    assert rep.roundtrip_error == sp.zeros(2)
