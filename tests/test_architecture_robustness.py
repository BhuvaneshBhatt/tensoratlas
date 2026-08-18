from __future__ import annotations

import sympy as sp

from tensoratlas import (
    ScalarField,
    VectorField,
    coordinate_chart,
    tensor_from_components,
    tensor_to_structured,
    tensor_from_structured,
    tensor_metadata,
    tensor_rebuild_like,
    tensor_roundtrip_structured,
    tensor_interop_report,
    sparse_tensor,
)


def test_tensorobject_structured_roundtrip_preserves_metadata() -> None:
    chart = coordinate_chart("Euclidean", "Cartesian", 2)
    x, y = chart.symbols()
    tensor = tensor_from_components(
        chart,
        [[x, y], [y, x + y]],
        "uu",
        name="T",
        symmetry_metadata={"symmetric": ((0, 1),)},
        domain_metadata={"domain": "demo", "units": "arb"},
    )
    structured = tensor_to_structured(tensor)
    rebuilt = tensor_from_structured(chart, structured, "uu", tensor.slot_bases, name=tensor.name)
    assert rebuilt.components == tensor.components
    assert rebuilt.symmetry_metadata == tensor.symmetry_metadata
    assert rebuilt.domain_metadata == tensor.domain_metadata


def test_tensor_rebuild_like_preserves_public_kind() -> None:
    chart = coordinate_chart("Euclidean", "Cartesian", 2)
    v = VectorField(chart, sp.Matrix([1, 2]))
    rebuilt = tensor_rebuild_like(v, sp.Matrix([3, 4]))
    assert isinstance(rebuilt, VectorField)
    assert rebuilt.components[0, 0] == sp.Integer(3)
    assert rebuilt.components[1, 0] == sp.Integer(4)


def test_tensor_roundtrip_structured_for_scalar_and_vector() -> None:
    chart = coordinate_chart("Euclidean", "Cartesian", 2)
    x, y = chart.symbols()
    s = ScalarField(chart, x + y)
    v = VectorField(chart, sp.Matrix([x, y]))
    s2 = tensor_roundtrip_structured(s)
    v2 = tensor_roundtrip_structured(v)
    assert isinstance(s2, ScalarField)
    assert isinstance(v2, VectorField)
    assert s2.expr == s.expr
    assert v2.components == v.components


def test_tensor_interop_report_flags_roundtrip_success() -> None:
    chart = coordinate_chart("Euclidean", "Cartesian", 2)
    tensor = tensor_from_components(
        chart,
        [[1, 0], [0, 2]],
        "ud",
        domain_metadata={"space": "V"},
    )
    rep = tensor_interop_report(tensor)
    assert rep["roundtrip_dense_equal"] is True
    assert rep["roundtrip_domain_metadata_equal"] is True
    assert rep["structured_shape"] == (2, 2)


def test_sparse_structured_roundtrip_report() -> None:
    arr = sparse_tensor((3, 3), {(0, 0): 1, (2, 1): sp.Symbol("a")}, domain_metadata={"layout": "sparse"})
    rep = tensor_interop_report(arr)
    assert rep["structured_nonzero_entries"] == 2
    assert rep["roundtrip_domain_metadata_equal"] is True


def test_tensor_metadata_contains_expected_keys() -> None:
    chart = coordinate_chart("Euclidean", "Cartesian", 2)
    tensor = tensor_from_components(chart, [[1, 2], [3, 4]], "uu", name="A")
    meta = tensor_metadata(tensor)
    assert meta["kind"] == "TensorObject"
    assert meta["rank"] == 2
    assert meta["shape"] == (2, 2)
    assert meta["name"] == "A"
