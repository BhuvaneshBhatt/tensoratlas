from sympy.tensor.tensor import tensor_indices

from tensoratlas import (
    abstract_index_type,
    fully_symmetric_head,
    riemann_tensor_head,
    canonical_reduce_by_hypergraph,
    build_operator_tree,
    expand_operator_tree,
    commute_operator_tree,
    connection,
    covariant_derivative_operator,
    index_type,
    tableau_projector,
    apply_tableau_projector,
    decompose_irreducible,
    representation_reduce,
    HypergraphCanonizationReport,
)


def _idx(itype, names):
    parts = names.split()
    out = []
    for n in parts:
        made = tensor_indices(n, itype)
        out.append(made[0] if isinstance(made, tuple) else made)
    return tuple(out)


def test_hypergraph_canonizer_report_and_dummy_partition_stable():
    L = abstract_index_type("Lg")
    R = riemann_tensor_head("Rg", L)
    a, b, c, d = _idx(L, "a b c d")
    expr1 = R(a, b, -c, -d) * R(c, d, -a, -b)
    expr2 = R(c, d, -a, -b) * R(a, b, -c, -d)
    red1, rep1 = canonical_reduce_by_hypergraph(expr1, with_report=True)
    red2, rep2 = canonical_reduce_by_hypergraph(expr2, with_report=True)
    assert isinstance(rep1, HypergraphCanonizationReport)
    assert str(red1.expr) == str(red2.expr)
    assert rep1.free_index_partition == rep2.free_index_partition
    assert rep1.dummy_index_partition == rep2.dummy_index_partition


def test_operator_tree_expands_and_commutator_returns_expr_wrapper():
    V = index_type("Vop")
    conn = connection("G", V)
    nabla = covariant_derivative_operator(V, connection=conn)
    T = fully_symmetric_head("Tg", [V.to_sympy(), V.to_sympy()])
    a, b, c, d, e = _idx(V.to_sympy(), "a b c d e")
    tree = build_operator_tree(T(a, b), (c,), operator=nabla)
    expanded = expand_operator_tree(tree, operator=nabla)
    assert hasattr(expanded, "expr")
    tree2 = build_operator_tree(T(a, b), (d, e), operator=nabla)
    commuted = commute_operator_tree(tree2, connection=conn)
    assert hasattr(commuted, "expr")


def test_tableau_projector_and_irreducible_decomposition_workflow():
    V = abstract_index_type("Vp")
    A = fully_symmetric_head("Ap", [V, V, V, V])
    a, b, c, d = _idx(V, "a b c d")
    expr = A(a, b, c, d)
    proj = tableau_projector(((0, 1), (2, 3)))
    projected = apply_tableau_projector(expr, proj)
    assert hasattr(projected, "expr")
    report = decompose_irreducible(expr, [((0, 1), (2, 3)), ((0, 2), (1, 3))])
    assert len(report.projectors) == 2
    assert len(report.components) == 2
    reduced, rep = representation_reduce(expr, [((0, 1), (2, 3))], with_report=True)
    assert hasattr(reduced, "expr")
    assert len(rep.components) == 1
