
from __future__ import annotations
import sympy as sp
from tensoratlas.charts import get_chart
from tensoratlas.basis import tangent_basis, cotangent_basis
from tensoratlas.tensor_core import TensorObject
from tensoratlas.tensor_indices import TensorIndex, IndexedTensor, IndexedTensorExpr
from tensoratlas.semantic_core import compile_semantic_node
from tensoratlas.semantic_matching import indexed_identity_rewrite_signatures, indexed_graph_equivalent

def _tensor(name: str, variance: str, *, symmetry=None):
    chart = get_chart("Euclidean", "Cartesian", 2)
    tb = tangent_basis(chart); cb = cotangent_basis(chart)
    slot_bases = tuple(tb if v == "u" else cb for v in variance)
    arr = sp.MutableDenseNDimArray.zeros(*([2] * len(variance)))
    return TensorObject(chart=chart, components=arr, variance_spec=variance, slot_bases=slot_bases, name=name, symmetry_metadata=dict(symmetry or {}))

def test_weyl_tracefree_rewrite_awareness():
    C = _tensor("C", "ulul", symmetry={"weyl": True})
    expr1 = IndexedTensor(C, (TensorIndex("a","u"), TensorIndex("b","l"), TensorIndex("a","u"), TensorIndex("d","l")))
    expr2 = IndexedTensor(C, (TensorIndex("x","u"), TensorIndex("y","l"), TensorIndex("x","u"), TensorIndex("z","l")))
    assert any("weyl_tracefree" in repr(s) for s in indexed_identity_rewrite_signatures(compile_semantic_node(expr1)))
    assert indexed_graph_equivalent(expr1, expr2)

def test_ricci_symmetry_rewrite_awareness():
    R = _tensor("Ricci", "ll", symmetry={"ricci_symmetric": True, "symmetric": ((0,1),)})
    expr1 = IndexedTensor(R, (TensorIndex("a","l"), TensorIndex("b","l")))
    expr2 = IndexedTensor(R, (TensorIndex("b","l"), TensorIndex("a","l")))
    assert indexed_graph_equivalent(expr1, expr2)
    assert any("ricci_symmetric" in repr(s) for s in indexed_identity_rewrite_signatures(compile_semantic_node(expr1)))

def test_metric_trace_rewrite_awareness():
    g = _tensor("g", "ll", symmetry={"metric": True, "symmetric": ((0,1),)})
    T = _tensor("T", "ul")
    expr1 = IndexedTensorExpr("tensor_product", (IndexedTensor(g, (TensorIndex("a","l"), TensorIndex("b","l"))), IndexedTensor(T, (TensorIndex("a","u"), TensorIndex("c","l")))))
    expr2 = IndexedTensorExpr("tensor_product", (IndexedTensor(g, (TensorIndex("x","l"), TensorIndex("y","l"))), IndexedTensor(T, (TensorIndex("x","u"), TensorIndex("z","l")))))
    assert indexed_graph_equivalent(expr1, expr2)
    assert any("metric_trace" in repr(s) for s in indexed_identity_rewrite_signatures(compile_semantic_node(expr1)))

def test_epsilon_delta_rewrite_awareness():
    eps = _tensor("eps", "lll", symmetry={"epsilon": True, "antisymmetric": ((0,1,2),)})
    delta = _tensor("delta", "ul", symmetry={"delta": True})
    expr1 = IndexedTensorExpr("tensor_product", (IndexedTensor(eps, tuple(TensorIndex(x, "l") for x in "abc")), IndexedTensor(delta, (TensorIndex("a","u"), TensorIndex("d","l")))))
    expr2 = IndexedTensorExpr("tensor_product", (IndexedTensor(eps, tuple(TensorIndex(x, "l") for x in "xyz")), IndexedTensor(delta, (TensorIndex("x","u"), TensorIndex("w","l")))))
    assert indexed_graph_equivalent(expr1, expr2)
    assert any("epsilon_delta" in repr(s) for s in indexed_identity_rewrite_signatures(compile_semantic_node(expr1)))

def test_metric_raise_lower_rewrite_awareness():
    g = _tensor("g", "ll", symmetry={"metric": True, "symmetric": ((0,1),)})
    V = _tensor("V", "u")
    expr1 = IndexedTensorExpr("tensor_product", (IndexedTensor(g, (TensorIndex("a","l"), TensorIndex("b","l"))), IndexedTensor(V, (TensorIndex("a","u"),))))
    expr2 = IndexedTensorExpr("tensor_product", (IndexedTensor(g, (TensorIndex("x","l"), TensorIndex("y","l"))), IndexedTensor(V, (TensorIndex("x","u"),))))
    assert indexed_graph_equivalent(expr1, expr2)
    assert any("metric_raise_lower" in repr(s) for s in indexed_identity_rewrite_signatures(compile_semantic_node(expr1)))

def test_broader_multiterm_pair_exchange_awareness():
    R = _tensor("R", "llll", symmetry={"riemann": True, "antisymmetric": ((0,1),(2,3)), "pair_symmetric": (((0,1),(2,3)),), "bianchi": True})
    expr1 = IndexedTensor(R, tuple(TensorIndex(x, "l") for x in "abcd"))
    expr2 = IndexedTensor(R, tuple(TensorIndex(x, "l") for x in ("c","d","a","b")))
    assert indexed_graph_equivalent(expr1, expr2)
    sigs = indexed_identity_rewrite_signatures(compile_semantic_node(expr1))
    assert any("riemann_pair_family" in repr(s) or "riemann_antisym_family" in repr(s) for s in sigs)
