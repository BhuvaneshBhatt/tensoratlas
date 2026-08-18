import sympy as sp

from sympy.tensor.tensor import tensor_indices

from tensoratlas import (
    abstract_index_type,
    ricci_tensor_head,
    abstract_hypergraph_signature,
    canonical_reduce_by_hypergraph,
    canonical_reduce_contraction_graph,
    coordinate_chart,
    tensor_from_components,
    indices,
    indexed,
)


def _idx(itype, names):
    parts = names.split()
    out = []
    for n in parts:
        made = tensor_indices(n, itype)
        out.append(made[0] if isinstance(made, tuple) else made)
    return tuple(out)


def test_abstract_hypergraph_signature_is_factor_permutation_stable():
    L = abstract_index_type("Lh")
    S = ricci_tensor_head("Sg", L)
    T = ricci_tensor_head("Tg", L)
    a, b, c = _idx(L, "a b c")
    expr1 = S(a, b) * T(-b, c)
    expr2 = T(-b, c) * S(a, b)
    assert abstract_hypergraph_signature(expr1) == abstract_hypergraph_signature(expr2)


def test_canonical_reduce_by_hypergraph_matches_under_factor_permutation():
    L = abstract_index_type("Lr")
    S = ricci_tensor_head("Sr", L)
    T = ricci_tensor_head("Tr", L)
    a, b, c = _idx(L, "a b c")
    expr1 = S(a, b) * T(-b, c)
    expr2 = T(-b, c) * S(a, b)
    red1 = canonical_reduce_by_hypergraph(expr1)
    red2 = canonical_reduce_by_hypergraph(expr2)
    assert str(red1.expr) == str(red2.expr)


def test_indexed_hypergraph_reducer_is_factor_permutation_stable():
    chart = coordinate_chart('Euclidean', 'Cartesian', 2)
    A = tensor_from_components(chart, [[1, 0], [0, 2]], 'ul', name='A')
    B = tensor_from_components(chart, [[3, 0], [0, 4]], 'ul', name='B')
    i_up, j_down, j_up, k_down = indices('i^ j_ j^ k_')
    expr1 = indexed(A, i_up, j_down) * indexed(B, j_up, k_down)
    expr2 = indexed(B, j_up, k_down) * indexed(A, i_up, j_down)
    red1 = canonical_reduce_contraction_graph(expr1)
    red2 = canonical_reduce_contraction_graph(expr2)
    assert str(red1) == str(red2)
