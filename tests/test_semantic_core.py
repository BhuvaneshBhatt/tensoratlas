import sympy as sp

from tensoratlas.abstract_tensor import AbstractNormalForm, BridgeConversionReport, canonical_tensor_normal_form
from tensoratlas.semantic_core import canonical_semantic_form, semantic_ir
from tensoratlas.unified_reduction import unified_tensor_normal_form


def test_abstract_normal_form_populates_semantic_key():
    x = sp.Symbol("x")
    nf = canonical_tensor_normal_form(x + x)
    assert isinstance(nf, AbstractNormalForm)
    assert nf.semantic_key


def test_unified_normal_form_uses_semantic_form_for_abstract_scalars():
    x = sp.Symbol("x")
    left = unified_tensor_normal_form(x + x)
    right = unified_tensor_normal_form(2 * x)
    assert left.semantic_form is not None
    assert right.semantic_form is not None
    assert left.key == right.key


def test_bridge_report_has_formal_relation_metadata():
    report = BridgeConversionReport("abstract", "indexed", sp.Symbol("x"), sp.Symbol("x"))
    assert report.relation == "formal_via_canonical_core"


def test_semantic_core_metadata_affects_key_stably():
    form1 = canonical_semantic_form(semantic_ir(sp.Symbol("x"), layer="abstract", metadata={"policy": "p"}))
    form2 = canonical_semantic_form(semantic_ir(sp.Symbol("x"), layer="abstract", metadata={"policy": "p"}))
    assert form1.key == form2.key
