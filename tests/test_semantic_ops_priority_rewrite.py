from __future__ import annotations

import sympy as sp

from tensoratlas import (
    clifford_algebra,
    exterior_form_nf,
    hodge_expr,
    codifferential_expr,
    interior_expr,
    lie_expr,
    gamma_string,
    gamma_string_to_sympy,
    semantic_match,
    semantic_operator_rules,
    semantic_rewrite,
    semantic_execute,
    semantic_layer_of,
)
from tensoratlas.charts import CoordinateChart
from tensoratlas.basis import orthonormal_tangent_basis
from tensoratlas.geometry_workflows import metric_signature_from_chart, exterior_execution_pipeline, spin_execution_pipeline


def test_semantic_operator_layers_and_rules_evaluate():
    x, y = sp.symbols('x y')
    alpha = exterior_form_nf({(0,): x, (1,): y}, dimension=2, basis_labels=('dx', 'dy'))
    op = hodge_expr(alpha, metric_signature=(1, 1))
    assert semantic_layer_of(op) == 'exterior_operator'
    env = semantic_match(op, __import__('tensoratlas').spat('hodge', bind='op'))
    assert env is not None
    rewritten, report = semantic_rewrite(op, semantic_operator_rules(), layer='exterior_operator')
    assert type(rewritten).__name__ == 'ExteriorFormNF'
    assert report.steps and report.steps[0].rule == 'eval_hodge'


def test_codiff_interior_lie_are_semantic_core_rewriteable():
    x, y = sp.symbols('x y')
    alpha = exterior_form_nf({(0,): x**2, (1,): x*y}, dimension=2, basis_labels=('dx', 'dy'))
    ops = [
        codifferential_expr(alpha, (x, y), metric_signature=(1, 1)),
        interior_expr((1, 0), alpha),
        lie_expr((x, 0), alpha, (x, y)),
    ]
    names = ['eval_codifferential', 'eval_interior', 'eval_lie']
    for op, name in zip(ops, names):
        rewritten, report = semantic_rewrite(op, semantic_operator_rules(), layer='exterior_operator')
        assert report.steps and report.steps[0].rule == name
        assert semantic_execute(rewritten).ir.layer in {'exterior', 'abstract'}


def test_gamma_string_is_semantic_node_not_only_nc_product():
    cl = clifford_algebra(2, (2, 0, 0), basis_labels=('0', '1'))
    gs = gamma_string(cl, [0, 0, 1], scalar=sp.Integer(3))
    sem = semantic_execute(gs)
    assert sem.ir.layer == 'gamma_string'
    rewritten, report = semantic_rewrite(gs, semantic_operator_rules(), layer='gamma_string')
    assert report.steps and report.steps[0].rule == 'eval_gamma_string'
    # gamma0*gamma0 -> +1 in Euclidean signature
    assert sp.expand(rewritten - 3 * gamma_string_to_sympy(gamma_string(cl, [1]))) == 0


def test_priority_b_execution_pipelines():
    x, y = sp.symbols('x y')
    chart = CoordinateChart(metric_name='Euclidean', chart_name='Cartesian', dimension=2, coordinate_names=('x', 'y'), metric_func=lambda c: sp.diag(1, 1))
    sig = metric_signature_from_chart(chart)
    assert sig == (1, 1)
    alpha = exterior_form_nf({(0,): x, (1,): y}, dimension=2, basis_labels=('dx', 'dy'))
    erep = exterior_execution_pipeline(alpha, chart=chart, vector_components=(1, 0), coordinates=(x, y))
    assert erep.signature == (1, 1)
    assert type(erep.hodge).__name__ == 'ExteriorFormNF'
    frame = orthonormal_tangent_basis(chart)
    srep = spin_execution_pipeline(sp.Symbol('psi', commutative=False), frame, coordinates=(x, y))
    assert srep.signature == (1, 1)
    assert srep.spin_connection_name.startswith('Spin(')
