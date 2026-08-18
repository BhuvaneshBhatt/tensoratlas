
from tensoratlas import (
    coordinate_chart, TensorObject, indexed, indices,
    to_indexed_tensor_form, normalize_indexed_expression, tnf_helper_audit
)
from tensoratlas.tensor_algebra import metric_tensor, kronecker_delta_tensor

def test_tnf_helper_audit_populates():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    d = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name="δ")
    i,j = indices("i^ j_")
    _ = to_indexed_tensor_form(indexed(d, i, j))
    audit = tnf_helper_audit()
    assert "to_indexed_tensor_form" in audit
    assert audit["to_indexed_tensor_form"].tnf_output is True

def test_tnf_helper_audit_marks_boundary_adapter():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    g = TensorObject.from_tensor_field(metric_tensor(chart, "ll"), name="g", symmetry_metadata={"symmetric": ((0,1),)})
    i,j = indices("i_ j_")
    _ = normalize_indexed_expression(indexed(g, i, j))
    audit = tnf_helper_audit()
    assert "normalize_tree_boundary_adapter" not in audit or audit["normalize_tree_boundary_adapter"].category == "boundary_only"
