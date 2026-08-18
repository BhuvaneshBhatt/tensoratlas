
import random
import sympy as sp
from tensoratlas import (
    coordinate_chart, TensorObject, indexed, indices,
    indexed_equal, stronger_indexed_equal, to_indexed_tensor_form,
    build_contraction_graph, build_contraction_plan, special_tensor_normalize,
    index_space, last_normalization_diagnostics, normalize_indexed_expression,
    optimizer_report
)
from tensoratlas.tensor_algebra import metric_tensor, permutation_tensor, kronecker_delta_tensor
from tensoratlas.basis import frame_basis, frame_structure_coefficients, connection_one_forms


def test_normal_form_drives_equality():
    chart = coordinate_chart('Euclidean','Cartesian',3)
    d = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name='δ')
    i,j = indices('i^ j_')
    a,b = indices('a^ b_')
    nf1 = to_indexed_tensor_form(indexed(d,i,j))
    nf2 = to_indexed_tensor_form(indexed(d,a,b))
    assert nf1 == nf2
    assert indexed_equal(indexed(d,i,j), indexed(d,a,b))


def test_contraction_graph_and_plan():
    chart = coordinate_chart('Euclidean','Cartesian',3)
    d = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name='δ')
    g = TensorObject.from_tensor_field(metric_tensor(chart,'ll'), name='g', symmetry_metadata={'symmetric': ((0,1),)})
    i,j,k,l = indices('i^ j_ k_ l_')
    factors=[indexed(d,i,j), indexed(g,k,l)]
    graph = build_contraction_graph(factors)
    plan = build_contraction_plan(factors)
    assert isinstance(graph, dict)
    assert plan.estimated_cost > 0


def test_unified_special_tensor_normalize():
    chart = coordinate_chart('Euclidean','Cartesian',3)
    gU = TensorObject.from_tensor_field(metric_tensor(chart,'uu'), name='gU')
    epsL = TensorObject.from_tensor_field(permutation_tensor(chart,'lll'), name='eps')
    i,a,b,c = indices('i^ a^ b_ c_')
    factors=[indexed(gU,i,a), indexed(epsL,a.dual(),b,c)]
    out_factors, scalar, plan = special_tensor_normalize(factors, sp.Integer(1))
    assert plan.estimated_cost > 0
    assert out_factors is not None


def test_symmetry_cached_equality():
    chart = coordinate_chart('Euclidean','Cartesian',3)
    g = TensorObject.from_tensor_field(metric_tensor(chart,'ll'), name='g', symmetry_metadata={'symmetric': ((0,1),)})
    i,j = indices('i_ j_')
    assert stronger_indexed_equal(indexed(g,i,j), indexed(g,j,i))


def test_typed_index_space_compatible_name_dim():
    chart = coordinate_chart('Euclidean','Cartesian',3)
    d = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name='δ')
    T = index_space('T(Cartesian)', 3)
    i,j = indices('i^ j_')
    i = type(i)(i.name, i.variance, T)
    j = type(j)(j.name, j.variance, T)
    expr = indexed(d, i, j)
    assert expr is not None


def test_lazy_geometry_caches():
    chart = coordinate_chart('Euclidean','Cartesian',3)
    fr = frame_basis('e', chart, lambda coords: sp.eye(chart.dimension))
    a = frame_structure_coefficients(fr)
    b = frame_structure_coefficients(fr)
    c = connection_one_forms(fr)
    d = connection_one_forms(fr)
    assert a is b
    assert c is d


def test_diagnostics_and_randomized_dummy_invariance():
    chart = coordinate_chart('Euclidean','Cartesian',3)
    d = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name='δ')
    names = ['a','b','c','i','j','k','p','q']
    for _ in range(5):
        n1, n2 = random.sample(names, 2)
        expr1 = indexed(d, type(indices('i^')[0])(n1,'u',None), type(indices('j_')[0])(n2,'l',None))
        expr2 = indexed(d, type(indices('i^')[0])('x','u',None), type(indices('j_')[0])('y','l',None))
        assert indexed_equal(expr1, expr2)
    _ = normalize_indexed_expression(indexed(d, *indices('i^ j_')))
    diag = last_normalization_diagnostics()
    assert diag is not None
