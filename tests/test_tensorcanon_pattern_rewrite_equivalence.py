
import sympy as sp
from tensoratlas import coordinate_chart, TensorObject, indexed, indices, normalize_indexed_expression, IndexedNormalizationConfig, indexed_signature, indexed_equal, stronger_indexed_equal
from tensoratlas.tensor_algebra import metric_tensor, permutation_tensor, kronecker_delta_tensor

def test_tiered_simplification_and_normal_form():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    d = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name="δ")
    i,j = indices("i^ j_")
    expr = indexed(d, i, j)
    sig1 = indexed_signature(expr, IndexedNormalizationConfig())
    sig2 = indexed_signature(expr, IndexedNormalizationConfig())
    assert sig1 == sig2

def test_scalar_tensor_split_collection():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    g = TensorObject.from_tensor_field(metric_tensor(chart, "ll"), name="g", symmetry_metadata={"symmetric": ((0,1),)})
    i,j = indices("i_ j_")
    expr1 = indexed(g,i,j)
    expr2 = indexed(g,j,i)
    assert stronger_indexed_equal(expr1, expr2)

def test_contraction_planning_zero_detection():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    eps = TensorObject.from_tensor_field(permutation_tensor(chart, "lll"), name="eps", symmetry_metadata={"antisymmetric": ((0,1,2),)})
    i,j,k = indices("i_ i_ k_")
    z = normalize_indexed_expression(indexed(eps, i, j, k))
    assert hasattr(z, "expr") and sp.simplify(z.expr) == 0

def test_delay_component_expansion_still_allows_metric_epsilon():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    gU = TensorObject.from_tensor_field(metric_tensor(chart, "uu"), name="gU")
    epsL = TensorObject.from_tensor_field(permutation_tensor(chart, "lll"), name="eps")
    i,a,b,c = indices("i^ a^ b_ c_")
    expr = indexed(gU, i, a) * indexed(epsL, a.dual(), b, c)
    norm = normalize_indexed_expression(expr, IndexedNormalizationConfig())
    assert norm is not None

def test_cached_equality_path():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    d = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name="δ")
    i,j = indices("i^ j_")
    a,b = indices("a^ b_")
    assert indexed_equal(indexed(d,i,j), indexed(d,a,b))
