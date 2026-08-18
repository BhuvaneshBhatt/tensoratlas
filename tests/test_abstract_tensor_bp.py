import pytest

from tensoratlas import (
    AbstractTensorCanonicalizationError,
    abstract_index_type,
    abstract_tensor_head,
    fully_symmetric_head,
    fully_antisymmetric_head,
    riemann_tensor_head,
    butler_portugal_canonicalize,
)
from sympy.tensor.tensor import tensor_indices


def test_butler_portugal_canonicalizes_antisymmetric_expression():
    lor = abstract_index_type('Lorentz', dummy_name='L')
    a0, a1, a2 = tensor_indices('a0,a1,a2', lor)
    A = fully_antisymmetric_head('A', [lor, lor])
    expr = A(a1, a0)
    out = butler_portugal_canonicalize(expr)
    assert out == -A(a0, a1)


def test_butler_portugal_canonicalizes_symmetric_expression():
    lor = abstract_index_type('Lorentz', dummy_name='L')
    a0, a1 = tensor_indices('a0,a1', lor)
    S = fully_symmetric_head('S', [lor, lor])
    expr = S(a1, a0)
    out = butler_portugal_canonicalize(expr)
    assert out == S(a0, a1)


def test_riemann_tensor_head_encodes_monoterm_symmetries():
    lor = abstract_index_type('Lorentz', dummy_name='L')
    a, b, c, d = tensor_indices('a,b,c,d', lor)
    R = riemann_tensor_head('R', lor)
    expr = butler_portugal_canonicalize(R(b, a, c, d) + R(a, b, d, c) + 2 * R(a, b, c, d))
    assert expr == 0


def test_invalid_riemann_rank_rejected():
    lor = abstract_index_type('Lorentz', dummy_name='L')
    with pytest.raises(AbstractTensorCanonicalizationError):
        abstract_tensor_head('R', [lor, lor], symmetry='riemann')
