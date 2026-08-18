
from __future__ import annotations
import sympy as sp

from tensoratlas.charts import get_chart
from tensoratlas.basis import tangent_basis, cotangent_basis
from tensoratlas.tensor_core import TensorObject
from tensoratlas.tensor_indices import TensorIndex, IndexedTensor, IndexedTensorExpr
from tensoratlas.critical_pair_rewrite_engine import (
    analyze_critical_pairs,
    semantic_native_indexed_geometry_reduce,
)

def _tensor(name: str, variance: str, *, symmetry=None):
    chart = get_chart("Euclidean", "Cartesian", 2)
    tb = tangent_basis(chart); cb = cotangent_basis(chart)
    slot_bases = tuple(tb if v == "u" else cb for v in variance)
    arr = sp.MutableDenseNDimArray.zeros(*([2] * len(variance)))
    return TensorObject(chart=chart, components=arr, variance_spec=variance, slot_bases=slot_bases, name=name, symmetry_metadata=dict(symmetry or {}))

def test_critical_pair_analysis_reports_pairs():
    rep = analyze_critical_pairs()
    assert rep.metadata["pair_count"] > 0
    assert any(p.overlap_kind in {"same_family", "declared_conflict", "tensor_identity_family"} for p in rep.pairs)

def test_semantic_native_indexed_geometry_reduce_runs():
    T = _tensor("T", "ul")
    term = IndexedTensor(T, (TensorIndex("i","u"), TensorIndex("i","l")))
    expr = IndexedTensorExpr("add", (term, term))
    rep = semantic_native_indexed_geometry_reduce(expr)
    assert isinstance(rep.semantic_fingerprint, tuple)
    assert len(rep.reduced_terms) >= 1
