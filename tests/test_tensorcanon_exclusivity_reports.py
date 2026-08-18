
from tensoratlas import (
    coordinate_chart, TensorObject, indexed, indices,
    to_indexed_tensor_form, normalize_indexed_expression,
    tnf_helper_audit, tnf_exclusivity_report, indexed_equal
)
from tensoratlas.tensor_algebra import kronecker_delta_tensor, metric_tensor, permutation_tensor, tensor_product

def test_exclusivity_report_populates():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    d = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name="δ")
    i,j = indices("i^ j_")
    _ = to_indexed_tensor_form(indexed(d, i, j))
    rep = tnf_exclusivity_report()
    assert "to_indexed_tensor_form" in rep.helpers_seen
    assert "to_indexed_tensor_form" in rep.parse_only_helpers

def test_runtime_audit_sees_nf_and_boundary_helpers():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    g = TensorObject.from_tensor_field(metric_tensor(chart, "ll"), name="g", symmetry_metadata={"symmetric": ((0,1),)})
    i,j = indices("i_ j_")
    _ = normalize_indexed_expression(indexed(g, i, j))
    audit = tnf_helper_audit()
    assert "normalize_indexed_expression" in audit
    assert audit["normalize_indexed_expression"].category == "boundary_only"

def test_mixed_special_tensor_chain_boundary():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    d = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name="δ")
    gU = TensorObject.from_tensor_field(metric_tensor(chart, "uu"), name="gU")
    gL = TensorObject.from_tensor_field(metric_tensor(chart, "ll"), name="gL")
    eps = TensorObject.from_tensor_field(permutation_tensor(chart, "lll"), name="eps", symmetry_metadata={"antisymmetric": ((0,1,2),)})
    a,b,c,i,j = indices("a^ b_ c_ i^ j_")
    expr = tensor_product(indexed(d, a, b), indexed(gU, i, a), indexed(gL, b, j), indexed(eps, j.dual(), c, b))
    out = normalize_indexed_expression(expr)
    assert out is not None

def test_dummy_renaming_invariance_basic():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    d = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name="δ")
    i,j = indices("i^ j_")
    a,b = indices("a^ b_")
    assert indexed_equal(indexed(d, i, j), indexed(d, a, b))
