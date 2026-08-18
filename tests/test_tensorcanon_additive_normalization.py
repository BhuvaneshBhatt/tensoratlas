
import sympy as sp
from tensoratlas import (
    coordinate_chart, TensorObject, indexed, indices,
    normalize_indexed_expression, IndexedNormalizationConfig,
    indexed_signature, indexed_equal, stronger_indexed_equal,
    optimizer_report, abstract_layer, is_abstract_layer, to_component_layer,
    index_space, build_contraction_plan, to_indexed_tensor_form,
    special_tensor_normalize, render_indexed_tensor_form, last_normalization_diagnostics
)
from tensoratlas.tensor_algebra import metric_tensor, permutation_tensor, kronecker_delta_tensor

def test_normal_form_is_more_central():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    d = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name="δ")
    i,j = indices("i^ j_")
    nf = to_indexed_tensor_form(indexed(d, i, j))
    txt = render_indexed_tensor_form(nf)
    assert nf.terms
    assert isinstance(txt, str)

def test_unified_special_tensor_engine():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    d = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name="δ")
    g = TensorObject.from_tensor_field(metric_tensor(chart, "ll"), name="g", symmetry_metadata={"symmetric": ((0,1),)})
    i,j,k,l = indices("i^ j_ k_ l_")
    res = special_tensor_normalize([indexed(d, i, j), indexed(g, k, l)], sp.Integer(1))
    assert res.plan.estimated_cost > 0

def test_typed_tensor_space_by_dimension_compatibility():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    d = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name="δ")
    T = index_space("T(Cartesian)", 3)
    i,j = indices("i^ j_")
    i = type(i)(i.name, i.variance, T)
    j = type(j)(j.name, j.variance, T)
    expr = indexed(d, i, j)
    assert expr is not None

def test_symmetry_canonicalization_and_equality():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    g = TensorObject.from_tensor_field(metric_tensor(chart, "ll"), name="g", symmetry_metadata={"symmetric": ((0,1),)})
    i,j = indices("i_ j_")
    assert stronger_indexed_equal(indexed(g,i,j), indexed(g,j,i))

def test_diagnostics_exist():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    d = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name="δ")
    i,j = indices("i^ j_")
    _ = normalize_indexed_expression(indexed(d, i, j))
    diag = last_normalization_diagnostics()
    assert diag is not None
