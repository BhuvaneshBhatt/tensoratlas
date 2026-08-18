import sympy as sp
from sympy.tensor.tensor import tensor_indices

from tensoratlas import (
    IndexType,
    Index,
    metric,
    fully_symmetric_head,
    covariant_derivative_operator,
    apply_covariant_derivative,
    build_operator_tree,
    operator_normal_form,
    compose_operator_trees,
    connection,
    canonical_tensor_expression,
)


def _idx(itype, names):
    out = []
    for n in names.split():
        made = tensor_indices(n, itype)
        out.append(made[0] if isinstance(made, tuple) else made)
    return tuple(out)


def test_second_covariant_derivative_product_rule_has_cross_terms():
    V = IndexType('Vp', dimension=3)
    nabla = covariant_derivative_operator(V)
    T = fully_symmetric_head('Tpr', [V.to_sympy()])
    S = fully_symmetric_head('Spr', [V.to_sympy()])
    a, b, c, d = _idx(V.to_sympy(), 'a b c d')
    expr = T(a) * S(b)
    out = apply_covariant_derivative(expr, (-c, -d), operator=nabla).expr
    text = str(out)
    assert text.count('nabla_Tpr') >= 2 or 'nabla_nabla_Tpr' in text
    assert 'nabla_Tpr' in text and 'nabla_Spr' in text


def test_nonmetric_connection_derivative_of_metric_uses_nonmetricity_head():
    V = IndexType('Vm', dimension=4)
    conn = connection('Gm', V, metric=metric(V), metric_compatible=False, non_metricity_name='Qm')
    nabla = covariant_derivative_operator(V, connection=conn)
    g = metric(V).to_sympy()
    a, b, c = _idx(V.to_sympy(), 'a b c')
    out = apply_covariant_derivative(g(a, b), (-c,), operator=nabla, connection=conn).expr
    assert 'Qm' in str(out)


def test_operator_normal_form_flattens_nested_same_operator_trees():
    V = IndexType('Vo', dimension=2)
    nabla = covariant_derivative_operator(V)
    T = fully_symmetric_head('Top', [V.to_sympy()])
    a, b, c = _idx(V.to_sympy(), 'a b c')
    inner = build_operator_tree(T(a), (b,), operator=nabla)
    outer = build_operator_tree(inner, (c,), operator=nabla)
    nf = operator_normal_form(outer)
    assert len(nf.derivative_indices) == 2


def test_compose_operator_trees_keeps_flattened_form():
    V = IndexType('Vc', dimension=2)
    nabla = covariant_derivative_operator(V)
    T = fully_symmetric_head('Tco', [V.to_sympy()])
    a, b, c = _idx(V.to_sympy(), 'a b c')
    inner = build_operator_tree(T(a), (b,), operator=nabla)
    outer = build_operator_tree(T(a), (c,), operator=nabla)
    composed = compose_operator_trees(outer, inner)
    nf = operator_normal_form(composed)
    assert len(nf.derivative_indices) == 2


def test_canonical_tensor_expression_is_idempotent_on_simple_input():
    V = IndexType('Vi', dimension=3)
    T = fully_symmetric_head('Tid', [V.to_sympy(), V.to_sympy()])
    a, b = _idx(V.to_sympy(), 'a b')
    expr = T(a, b)
    first = canonical_tensor_expression(expr).expr
    second = canonical_tensor_expression(first).expr
    assert first == second
