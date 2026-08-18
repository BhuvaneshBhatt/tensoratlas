
from __future__ import annotations

from tensoratlas.ir_curvature_rewriting import (
    IR_CURVATURE_RULES,
    ir_curvature_symbol,
    ir_curvature_linear_combo,
    compile_curvature_symbol_to_ir,
    execute_ir_curvature_rewriting,
)

def test_ir_curvature_rules_exist():
    assert len(IR_CURVATURE_RULES) >= 5
    assert IR_CURVATURE_RULES[0].metadata["terminating"] is True

def test_compile_curvature_symbol_to_ir():
    node = ir_curvature_symbol("Riemann", 4, name="R")
    out = compile_curvature_symbol_to_ir(node)
    assert out.kind == "curvature_symbol"
    assert out.metadata["family"] == "Riemann"

def test_riemann_rewrites_on_ir():
    node = ir_curvature_symbol("Riemann", 4, name="R")
    rep = execute_ir_curvature_rewriting(node)
    assert len(rep.applied_rules) >= 1
    assert rep.rewritten_ir.kind in {"curvature_contraction", "curvature_linear_combo"}

def test_ricci_rewrites_on_ir():
    node = ir_curvature_symbol("Ricci", 4, name="Ric")
    rep = execute_ir_curvature_rewriting(node)
    assert rep.rewritten_ir.kind in {"curvature_contraction", "curvature_linear_combo"}

def test_curvature_combo_normalizes():
    a = ir_curvature_symbol("Weyl", 4, name="W")
    combo = ir_curvature_linear_combo(a)
    rep = execute_ir_curvature_rewriting(combo)
    assert rep.rewritten_ir.kind == "curvature_symbol" or "ir_rewrite_curvature_singleton_combo" in rep.applied_rules
