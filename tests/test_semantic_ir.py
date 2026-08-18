
from __future__ import annotations
import sympy as sp

from tensoratlas.charts import get_chart
from tensoratlas.basis import tangent_basis, cotangent_basis
from tensoratlas.tensor_core import TensorObject
from tensoratlas.tensor_indices import TensorIndex, IndexedTensor, IndexedTensorExpr
from tensoratlas.geometry_components import component_tensor_field
from tensoratlas.exterior_geometry import ExteriorFormNF
from tensoratlas.semantic_ir import (
    compile_tensor_expr,
    normalize_tensor_expr,
    materialize_tensor_expr,
    compile_tensor_expr_report,
    execute_tensor_expr,
)

def _tensor(name: str, variance: str, *, symmetry=None):
    chart = get_chart("Euclidean", "Cartesian", 2)
    tb = tangent_basis(chart); cb = cotangent_basis(chart)
    slot_bases = tuple(tb if v == "u" else cb for v in variance)
    arr = sp.MutableDenseNDimArray.zeros(*([2] * len(variance)))
    return TensorObject(chart=chart, components=arr, variance_spec=variance, slot_bases=slot_bases, name=name, symmetry_metadata=dict(symmetry or {}))

def test_compile_indexed_tensor_expr():
    T = _tensor("T", "ul")
    t = IndexedTensor(T, (TensorIndex("i","u"), TensorIndex("j","l")))
    ir = compile_tensor_expr(t)
    assert ir.kind == "indexed_tensor"
    assert ir.metadata["tensor_name"] == "T"

def test_normalize_indexed_expr_ir_ordering():
    T = _tensor("T", "ul")
    U = _tensor("U", "ul")
    t1 = IndexedTensor(T, (TensorIndex("i","u"), TensorIndex("i","l")))
    t2 = IndexedTensor(U, (TensorIndex("a","u"), TensorIndex("a","l")))
    e = IndexedTensorExpr("add", (t2, t1))
    ir = normalize_tensor_expr(compile_tensor_expr(e))
    assert ir.kind == "indexed_expr:add"
    assert len(ir.children) == 2

def test_materialize_exterior_form_ir():
    form = ExteriorFormNF(2, {(0,): sp.Integer(1), (1,): sp.Integer(2)}, basis_labels=("e0","e1"), metadata={})
    ir = compile_tensor_expr(form)
    out = materialize_tensor_expr(ir)
    assert isinstance(out, ExteriorFormNF)
    assert out.terms[(0,)] == 1

def test_execute_component_tensor_expr():
    chart = get_chart("Euclidean", "Cartesian", 2)
    field = component_tensor_field("V", chart, "u", [1, 2])
    rep = execute_tensor_expr(field)
    assert rep.ir_kind == "component_tensor"
    assert rep.materialized.name == "V"

def test_compile_tensor_expr_report_has_fingerprint():
    x = sp.Symbol("x")
    rep = compile_tensor_expr_report(x + 1)
    assert isinstance(rep.semantic_fingerprint, tuple)
