from __future__ import annotations

import sympy as sp

from tensoratlas import (
    SemanticRewriteRule,
    clifford_algebra,
    compile_semantic_ir,
    semantic_execute,
    semantic_match,
    semantic_normalize_object,
    semantic_rewrite,
    spat,
    svar,
    unified_reduce_with_trace,
)


def test_semantic_core_commutative_normalization_is_order_invariant():
    x, y = sp.symbols('x y')
    left = semantic_execute(y + x)
    right = semantic_execute(x + y)
    assert left.key == right.key
    assert semantic_normalize_object(y + x) == x + y



def test_semantic_rewrite_engine_applies_pattern_rules_to_sympy_scalars():
    x = sp.symbols('x')
    a = svar('a')
    rules = (
        SemanticRewriteRule('drop_add_zero_rhs', spat('add', a, 0), a),
        SemanticRewriteRule('drop_mul_one_rhs', spat('mul', a, 1), a),
    )
    expr = sp.Mul(sp.Add(x, 0, evaluate=False), 1, evaluate=False)
    out, report = semantic_rewrite(expr, rules)
    assert out == x
    assert report.steps
    assert report.steps[0].rule == 'drop_mul_one_rhs'



def test_semantic_match_exposes_bindings():
    x = sp.symbols('x')
    env = semantic_match(sp.Add(x, 1, evaluate=False), spat('add', svar('lhs'), 1))
    assert env is not None
    assert env['lhs'] == x



def test_semantic_core_compiles_clifford_objects_into_typed_ir():
    cl = clifford_algebra(3, (3, 0, 0), name='Cl3', basis_labels=('e1', 'e2', 'e3'))
    ir = compile_semantic_ir(cl)
    assert ir.layer == 'clifford'
    assert ir.root is not None
    assert ir.root.kind == 'clifford_algebra'
    assert semantic_execute(cl).ir.root.kind == 'clifford_algebra'



def test_unified_reduce_with_trace_includes_semantic_rewrite_step():
    x = sp.symbols('x')
    a = svar('a')
    rules = (
        SemanticRewriteRule('drop_add_zero_rhs', spat('add', a, 0), a),
        SemanticRewriteRule('drop_mul_one_rhs', spat('mul', a, 1), a),
    )
    expr = sp.Mul(sp.Add(x, 0, evaluate=False), 1, evaluate=False)
    reduced, trace = unified_reduce_with_trace(expr, semantic_rules=rules)
    assert reduced == x
    assert trace.steps
    assert trace.steps[0].name == 'semantic_rewrite'
