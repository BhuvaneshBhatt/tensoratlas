from tensoratlas.abstract_tensor import fully_symmetric_head, canonical_tensor_expression, abstract_index_type
from tensoratlas.contracts.normal_forms import check_normal_form_contract
from tensoratlas.canonical_keys import structural_key
from sympy.tensor.tensor import tensor_indices


def test_canonical_tensor_expression_satisfies_normal_form_contract():
    L = abstract_index_type("L")
    T = fully_symmetric_head("TC", [L, L])
    a, b = tensor_indices("a b", L)
    expr = T(a, b)
    result = check_normal_form_contract(expr, canonical_tensor_expression, equivalent=lambda e: T(b, a))
    assert result.idempotent
    assert result.stable_under_equivalence
    assert result.normalized_key == structural_key(canonical_tensor_expression(expr))
