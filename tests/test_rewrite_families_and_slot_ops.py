import sympy as sp

from tensoratlas import (
    IndexType,
    Index,
    riemann_tensor_head,
    list_rewrite_families,
    abstract_expand,
    indexed_expand,
    coordinate_chart,
    tensor_from_components,
    change_tensor_basis,
    change_tensor_slots,
    normalize_contraction_graph,
    contraction_graph_key,
    tensor_product_dispatch,
    tensor_transpose_dispatch,
    tensor_contract_dispatch,
    tensor_wedge_dispatch,
    component_to_abstract,
    abstract_contraction_graph_key,
)
from tensoratlas.tensor_core import TensorObject
from tensoratlas.normal_forms import tnf_build_array
from tensoratlas import indices, indexed
from tensoratlas.tensor_core import TensorExpr
from tensoratlas.basis import tangent_basis, cotangent_basis, dual_basis


def test_rewrite_family_registry_and_abstract_expand_surface():
    families = {f.name for f in list_rewrite_families("abstract")}
    assert {"linearity", "metric_delta", "multiterm", "invariant", "all"}.issubset(families)
    V = IndexType("V", dimension=4)
    a, b, c, d = [Index(ch, V, "u") for ch in "abcd"]
    R = riemann_tensor_head("R", V.to_sympy())
    expr = R(a.to_sympy(), b.to_sympy(), c.to_sympy(), d.to_sympy())
    out = abstract_expand(expr, families=("linearity", "multiterm"), dimension=4)
    assert out is not None


def test_indexed_expand_surface_on_simple_indexed_product():
    chart = coordinate_chart('Euclidean', 'Cartesian', 2)
    A = tensor_from_components(chart, [[1, 0], [0, 2]], 'ul', name='A')
    i, j_down = indices('i^ j_')
    expr = indexed(A, i, j_down)
    out = indexed_expand(expr, families=("canonicalize",))
    assert out is not None


def test_basis_and_slot_change_wrappers_and_dispatch_helpers():
    chart = coordinate_chart('Euclidean', 'Cartesian', 2)
    arr = tnf_build_array((2, 2), lambda idx: sp.Integer(10 * idx[0] + idx[1]))
    obj = tensor_from_components(chart, arr, 'ul', name='T')
    base = TensorObject.from_tensor_field(obj.to_tensor_field())
    tb = tangent_basis(chart)
    cb = cotangent_basis(chart)
    changed = change_tensor_basis(base, (tb, cb))
    assert len(changed.slot_bases) == 2
    permuted = change_tensor_slots(base, perm=(1, 0))
    assert permuted.variance_spec == 'lu'
    tp = tensor_product_dispatch(base, base)
    tt = tensor_transpose_dispatch(base, (1, 0))
    tc = tensor_contract_dispatch(tp, (1, 2))
    wedge = tensor_wedge_dispatch(tensor_from_components(chart, [1, 2], 'l'), tensor_from_components(chart, [3, 4], 'l'))
    assert tt is not None and tc is not None and wedge is not None


def test_contraction_graph_normalization_keys_are_stable():
    chart = coordinate_chart('Euclidean', 'Cartesian', 3)
    A = tensor_from_components(chart, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 'ul', name='A')
    B = tensor_from_components(chart, [[2, 0, 0], [0, 2, 0], [0, 0, 2]], 'ul', name='B')
    i, j_down, j_up, k_down = indices('i^ j_ j^ k_')
    expr = TensorExpr('tensor_product', (indexed(A, i, j_down), indexed(B, j_up, k_down)))
    g = normalize_contraction_graph(expr)
    assert 'summary' in g and g['summary']['contraction_edges'] >= 1
    assert contraction_graph_key(expr) == contraction_graph_key(expr)


def test_component_bridge_exposes_abstract_contraction_graph_key():
    chart = coordinate_chart('Euclidean', 'Cartesian', 2)
    obj = tensor_from_components(chart, [[1, 0], [0, 1]], 'll', name='G')
    abstract = component_to_abstract(obj)
    key = abstract_contraction_graph_key(abstract.expr)
    assert isinstance(key, tuple)
