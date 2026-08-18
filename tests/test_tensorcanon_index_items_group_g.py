
from tensoratlas import (
    coordinate_chart, TensorObject, indexed, indices,
    to_indexed_tensor_form, normalize_indexed_expression,
    stronger_indexed_equal, indexed_equal, IndexedTensorExpr
)
from tensoratlas.tensor_algebra import kronecker_delta_tensor, metric_tensor, permutation_tensor

def test_symmetry_normalization_sign_stable():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    eps = TensorObject.from_tensor_field(permutation_tensor(chart, "lll"), name="eps", symmetry_metadata={"antisymmetric": ((0,1,2),)})
    i,j,k = indices("i_ j_ k_")
    a,b,c = indices("a_ b_ c_")
    nf1 = to_indexed_tensor_form(indexed(eps, i, j, k))
    nf2 = to_indexed_tensor_form(indexed(eps, a, b, c))
    assert nf1 == nf2

def test_mixed_special_tensor_chain_boundary():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    d = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name="δ")
    gU = TensorObject.from_tensor_field(metric_tensor(chart, "uu"), name="gU")
    gL = TensorObject.from_tensor_field(metric_tensor(chart, "ll"), name="gL")
    eps = TensorObject.from_tensor_field(permutation_tensor(chart, "lll"), name="eps", symmetry_metadata={"antisymmetric": ((0,1,2),)})
    a,b,c,i,j = indices("a^ b_ c_ i^ j_")
    expr = IndexedTensorExpr("tensor_product", (
        indexed(d, a, b),
        indexed(gU, i, a),
        indexed(gL, b, j),
        indexed(eps, j, c, b),
    ))
    out = normalize_indexed_expression(expr)
    assert out is not None

def test_dummy_renaming_invariance_basic():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    d = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name="δ")
    i,j = indices("i^ j_")
    a,b = indices("a^ b_")
    assert indexed_equal(indexed(d, i, j), indexed(d, a, b))

def test_symmetric_metric_equality():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    g = TensorObject.from_tensor_field(metric_tensor(chart, "ll"), name="g", symmetry_metadata={"symmetric": ((0,1),)})
    i,j = indices("i_ j_")
    assert stronger_indexed_equal(indexed(g, i, j), indexed(g, j, i))
