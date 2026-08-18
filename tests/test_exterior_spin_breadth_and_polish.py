from __future__ import annotations

import sympy as sp

from tensoratlas import (
    coordinate_chart,
    manifold,
    chart_definition,
    tangent_bundle,
    cotangent_bundle,
    riemannian_metric_from_chart,
    levi_civita_connection,
    spin_structure,
    spinor_bundle,
    clifford_algebra,
    gamma_generators,
    gamma_anticommutator,
    clifford_reduce,
    geometry_to_data,
    geometry_from_data,
    export_geometry_archive,
    import_geometry_archive,
)


def test_spinor_bundle_dimension_matches_even_dim_formula():
    M = manifold("M", 4)
    S = spinor_bundle(M, (1, 3, 0), chirality="dirac")
    assert S.complex_dimension == 4
    assert S.spin_structure.signature == (1, 3, 0)


def test_clifford_anticommutator_and_reduction():
    cl = clifford_algebra(3, (3, 0, 0), basis_labels=("1", "2", "3"))
    g1, g2, g3 = gamma_generators(cl)
    assert gamma_anticommutator(g1, g1, cl) == 2
    assert gamma_anticommutator(g1, g2, cl) == 0
    assert clifford_reduce(g2 * g1, cl) == -g1 * g2
    assert clifford_reduce(g1 * g1 + g2 * g2 + g3 * g3, cl) == 3


def test_geometry_roundtrip_via_data_dict():
    chart = coordinate_chart("Euclidean", "Polar", 2)
    M = manifold("Plane", 2).with_chart(chart)
    chart_def = chart_definition(M, chart)
    bundle = tangent_bundle(M)
    metric = riemannian_metric_from_chart(chart, manifold_name="Plane")
    connection = levi_civita_connection(metric)

    for obj in (M, chart, chart_def, bundle, metric, connection, spin_structure(M, (2, 0, 0))):
        rebuilt = geometry_from_data(geometry_to_data(obj))
        assert type(rebuilt) is type(obj)


def test_geometry_archive_export_import(tmp_path):
    chart = coordinate_chart("Euclidean", "Cartesian", 2)
    M = manifold("Plane", 2).with_chart(chart)
    archive_path = tmp_path / "geom.json"
    export_geometry_archive([M, spinor_bundle(M, (2, 0, 0))], archive_path, metadata={"suite": "exterior-spin"})
    archive = import_geometry_archive(archive_path)
    assert archive.version == "geometry-archive-v1"
    assert archive.metadata["suite"] == "exterior-spin"
    assert len(archive.objects) == 2
