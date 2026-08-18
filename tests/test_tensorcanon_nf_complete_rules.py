
import sympy as sp
from tensoratlas import coordinate_chart, TensorObject, indexed, indices, stronger_indexed_equal, indexed_equal, to_indexed_tensor_form, normalize_indexed_expression, last_normalization_diagnostics
from tensoratlas.tensor_algebra import metric_tensor, kronecker_delta_tensor

def test_symmetric_leaf_nf_identity():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    g = TensorObject.from_tensor_field(metric_tensor(chart, "ll"), name="g", symmetry_metadata={"symmetric": ((0,1),)})
    i,j = indices("i_ j_")
    nf1 = to_indexed_tensor_form(indexed(g, i, j))
    nf2 = to_indexed_tensor_form(indexed(g, j, i))
    assert nf1 == nf2

def test_symmetric_leaf_equality():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    g = TensorObject.from_tensor_field(metric_tensor(chart, "ll"), name="g", symmetry_metadata={"symmetric": ((0,1),)})
    i,j = indices("i_ j_")
    assert stronger_indexed_equal(indexed(g, i, j), indexed(g, j, i))
    assert indexed_equal(indexed(g, i, j), indexed(g, j, i))

def test_delta_leaf_equality_still_works():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    d = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name="δ")
    i,j = indices("i^ j_")
    a,b = indices("a^ b_")
    assert indexed_equal(indexed(d, i, j), indexed(d, a, b))

def test_nf_normalize_boundary_and_diag():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    d = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name="δ")
    i,j = indices("i^ j_")
    out = normalize_indexed_expression(indexed(d, i, j))
    diag = last_normalization_diagnostics()
    assert out is not None
    assert diag is not None
