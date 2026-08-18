import random

import sympy as sp

from tensoratlas import (
    TensorField,
    TensorObject,
    coordinate_chart,
    alpha_rename_dummies,
    indexed_equal,
    indexed_signature,
    normalize_indexed_expression,
    to_indexed_tensor_form,
    indices,
    IndexedTensor,
)


def _identity_tensor():
    cart = coordinate_chart("Euclidean", "Cartesian", 3)
    field = TensorField(cart, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "ul")
    return TensorObject.from_tensor_field(field)


def test_nf_signature_is_stable_under_alpha_renaming():
    tensor = _identity_tensor()
    i, j = indices("i^ j_")
    expr = IndexedTensor(tensor, (i, j))
    renamed = alpha_rename_dummies(expr, prefix="q")
    assert indexed_signature(expr) == indexed_signature(renamed)
    assert indexed_equal(expr, renamed)


def test_nf_roundtrip_is_idempotent_on_sampled_indexed_inputs():
    for seed in range(5):
        random.seed(seed)
        tensor = _identity_tensor()
        i, j = indices("i^ j_")
        expr = IndexedTensor(tensor, (i, j))
        normalized_once = normalize_indexed_expression(expr)
        normalized_twice = normalize_indexed_expression(normalized_once)
        assert indexed_equal(normalized_once, normalized_twice)
        nf = to_indexed_tensor_form(expr)
        assert nf == to_indexed_tensor_form(normalized_once)
