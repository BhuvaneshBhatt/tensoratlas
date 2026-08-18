
import sympy as sp
from tensoratlas import (
    coordinate_chart, TensorObject, indexed, indices,
    normalize_indexed_expression, IndexedNormalizationConfig,
    indexed_signature, indexed_equal, stronger_indexed_equal,
    optimizer_prepass, optimizer_report, abstract_layer, component_layer,
    is_abstract_layer, is_component_layer, to_component_layer
)
from tensoratlas.tensor_algebra import metric_tensor, permutation_tensor, kronecker_delta_tensor

def test_optimizer_prepass_and_report():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    d = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name="δ")
    i,j = indices("i^ j_")
    expr = indexed(d, i, j)
    pre = optimizer_prepass(expr)
    rep = optimizer_report(expr)
    assert pre is not None
    assert rep.original_kind in {"tensor", "product", "add", "IndexedTensor"}

def test_abstract_component_layer_markers():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    d = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name="δ")
    i,j = indices("i^ j_")
    expr = indexed(d, i, j)
    a = abstract_layer(expr)
    c = component_layer(expr)
    assert is_abstract_layer(a)
    assert is_component_layer(c)
    assert is_component_layer(to_component_layer(a))

def test_immutable_cached_signature_path():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    d = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name="δ")
    i,j = indices("i^ j_")
    a,b = indices("a^ b_")
    cfg = IndexedNormalizationConfig()
    s1 = indexed_signature(indexed(d,i,j), cfg)
    s2 = indexed_signature(indexed(d,a,b), cfg)
    assert s1 == s2
    assert indexed_equal(indexed(d,i,j), indexed(d,a,b))

def test_zero_detection_precedes_unsafe_validation():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    eps = TensorObject.from_tensor_field(permutation_tensor(chart, "lll"), name="eps", symmetry_metadata={"antisymmetric": ((0,1,2),)})
    i,j,k = indices("i_ i_ k_")
    z = normalize_indexed_expression(indexed(eps, i, j, k))
    assert hasattr(z, "expr") and sp.simplify(z.expr) == 0

def test_tiered_normalization_still_handles_metric_epsilon():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    g = TensorObject.from_tensor_field(metric_tensor(chart, "ll"), name="g", symmetry_metadata={"symmetric": ((0,1),)})
    i,j = indices("i_ j_")
    assert stronger_indexed_equal(indexed(g,i,j), indexed(g,j,i))
