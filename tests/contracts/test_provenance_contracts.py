from tensoratlas.abstract_tensor import (
    riemann_tensor_head,
    apply_curvature_identity_library,
    abstract_index_type,
)
from tensoratlas.contracts.provenance import check_provenance_contract
from sympy.tensor.tensor import tensor_indices


def test_curvature_identity_report_satisfies_provenance_contract():
    L = abstract_index_type("LR")
    R = riemann_tensor_head("RProv", L)
    a, b, c, d = tensor_indices("a b c d", L)
    _, report = apply_curvature_identity_library(R(a, b, c, d), library='core', with_report=True, max_rounds=1)
    assert report.steps
    step = report.steps[0]
    step_payload = dict(step.provenance) | {
        'before': step.before_expr,
        'after': step.after_expr,
        'identity_name': step.identity_name,
        'before_fingerprint': step.before_fingerprint,
        'after_fingerprint': step.after_fingerprint,
    }
    result = check_provenance_contract(step_payload)
    assert result.has_before
    assert result.has_after
    assert result.has_rule_family
    assert result.has_semantic_delta
