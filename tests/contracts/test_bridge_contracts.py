import sympy as sp

from tensoratlas.abstract_tensor import indexed_to_abstract, abstract_to_indexed
from tensoratlas.contracts.bridges import check_bridge_contract
from tensoratlas.canonical_keys import structural_key
from tensoratlas import coordinate_chart, cotangent_basis, indexed, indices, TensorObject


def test_abstract_indexed_bridge_preserves_structural_key_for_simple_tensor():
    cart = coordinate_chart('Euclidean', 'Cartesian', 2)
    arr = sp.MutableDenseNDimArray.zeros(2, 2)
    arr[0, 0] = 1
    arr[1, 1] = 2
    T = TensorObject(cart, arr, 'll', (cotangent_basis(cart), cotangent_basis(cart)), name='T')
    i, j = indices('i_ j_')
    leaf = indexed(T, i, j)
    result = check_bridge_contract(
        leaf,
        indexed_to_abstract,
        lambda abstract: abstract_to_indexed(abstract, tensor_registry={'T': T}),
        canonical_key=structural_key,
    )
    assert result.preserves_canonical_key
