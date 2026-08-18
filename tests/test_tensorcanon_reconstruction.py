
import sympy as sp
from tensoratlas import (
    coordinate_chart, TensorObject, indexed, indices,
    normalize_indexed_expression, IndexedNormalizationConfig,
    indexed_signature, indexed_equal, stronger_indexed_equal,
    optimizer_prepass, optimizer_report, abstract_layer,
    is_abstract_layer, is_component_layer, to_component_layer,
    build_contraction_plan, to_indexed_tensor_form
)
from tensoratlas.tensor_algebra import metric_tensor, permutation_tensor, kronecker_delta_tensor

def test_normal_form_is_centralish():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    d = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name="δ")
    i,j = indices("i^ j_")
    nf = to_indexed_tensor_form(indexed(d, i, j))
    assert nf.terms

def test_contraction_plan_exists():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    d = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name="δ")
    g = TensorObject.from_tensor_field(metric_tensor(chart, "ll"), name="g", symmetry_metadata={"symmetric": ((0,1),)})
    i,j,k,l = indices("i^ j_ k_ l_")
    f1 = indexed(d, i, j)
    f2 = indexed(g, k, l)
    plan = build_contraction_plan([f1, f2])
    assert plan.ordered_factors
    assert plan.estimated_cost > 0

def test_optimizer_and_layering():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    eps = TensorObject.from_tensor_field(permutation_tensor(chart, "lll"), name="eps", symmetry_metadata={"antisymmetric": ((0,1,2),)})
    i,j,k = indices("i_ i_ k_")
    rep = optimizer_report(indexed(eps, i, j, k))
    assert rep is not None
    a = abstract_layer(indexed(eps, i, j, k))
    c = to_component_layer(a)
    assert is_abstract_layer(a)
    assert is_component_layer(c)

def test_symmetric_metric_still_equal():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    g = TensorObject.from_tensor_field(metric_tensor(chart, "ll"), name="g", symmetry_metadata={"symmetric": ((0,1),)})
    i,j = indices("i_ j_")
    assert stronger_indexed_equal(indexed(g,i,j), indexed(g,j,i))

def test_cached_signature_path():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    d = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name="δ")
    i,j = indices("i^ j_")
    a,b = indices("a^ b_")
    assert indexed_equal(indexed(d,i,j), indexed(d,a,b))
