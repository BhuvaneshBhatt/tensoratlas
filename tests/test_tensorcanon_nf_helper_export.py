from tensoratlas import tnf_helper_audit, TNFHelperAuditRecord, coordinate_chart, TensorObject, indexed, indices, to_indexed_tensor_form
from tensoratlas.tensor_algebra import kronecker_delta_tensor

def test_tnf_helper_audit_root_export_and_population():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    d = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name="δ")
    i,j = indices("i^ j_")
    _ = to_indexed_tensor_form(indexed(d, i, j))
    audit = tnf_helper_audit()
    assert isinstance(audit, dict)
    assert "to_indexed_tensor_form" in audit
    rec = audit["to_indexed_tensor_form"]
    assert isinstance(rec, TNFHelperAuditRecord)
    assert rec.tnf_output is True
