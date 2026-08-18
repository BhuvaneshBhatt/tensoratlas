
from __future__ import annotations

import sympy as sp

from tensoratlas.charts import get_chart
from tensoratlas.basis import tangent_basis, cotangent_basis
from tensoratlas.tensor_core import TensorObject
from tensoratlas.tensor_indices import TensorIndex, IndexedTensor
from tensoratlas.semantic_rewrite import semantic_match, spat, svar
from tensoratlas.semantic_matching import indexed_tensor_orbit_specs, semantic_equivalent_objects


def _sym_tensor(rank2_symmetry: str):
    chart = get_chart("Euclidean", "Cartesian", 2)
    tb = tangent_basis(chart)
    cb = cotangent_basis(chart)
    sym = {rank2_symmetry: ((0, 1),)}
    arr = sp.MutableDenseNDimArray.zeros(2, 2)
    return TensorObject(chart=chart, components=arr, variance_spec="ll", slot_bases=(cb, cb), name="A", symmetry_metadata=sym)


def test_indexed_tensor_orbit_specs_symmetric_has_swap():
    A = _sym_tensor("symmetric")
    obj = IndexedTensor(A, (TensorIndex("i", "l"), TensorIndex("j", "l")))
    specs = indexed_tensor_orbit_specs(obj)
    assert (0, 1) in [perm for perm, _ in specs]
    assert (1, 0) in [perm for perm, _ in specs]


def test_semantic_match_indexed_tensor_through_symmetric_orbit():
    A = _sym_tensor("symmetric")
    obj = IndexedTensor(A, (TensorIndex("i", "l"), TensorIndex("j", "l")))
    env = semantic_match(obj, spat("indexed_tensor", svar("a"), svar("b")))
    assert env is not None
    # Now require reversed order; symmetric orbit should still match.
    env2 = semantic_match(obj, spat("indexed_tensor", TensorIndex("j", "l"), TensorIndex("i", "l")))
    assert env2 is not None


def test_semantic_match_tracks_antisymmetric_orbit_sign():
    A = _sym_tensor("antisymmetric")
    obj = IndexedTensor(A, (TensorIndex("i", "l"), TensorIndex("j", "l")))
    env = semantic_match(obj, spat("indexed_tensor", TensorIndex("j", "l"), TensorIndex("i", "l"), bind="m"))
    assert env is not None
    assert env.get("__orbit_sign__") == -1
    assert env.get("m__orbit_sign__") == -1
