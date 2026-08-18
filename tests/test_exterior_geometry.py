from __future__ import annotations

import sympy as sp

from tensoratlas import (
    coordinate_chart,
    frame_basis,
    orthonormal_tangent_basis,
    clifford_algebra,
    spin_connection,
    gamma_frame_generators,
    spin_covariant_components,
    dirac_operator,
    exterior_form_nf,
    wedge_exterior_forms,
    exterior_derivative_nf,
    exterior_identity_report,
    geometry_to_data,
    geometry_from_data,
    export_geometry_archive,
    import_geometry_archive,
)


def test_spin_connection_and_dirac_operator_smoke():
    chart = coordinate_chart('Euclidean', 'Cartesian', 2)
    frame = orthonormal_tangent_basis(chart)
    cliff = clifford_algebra(2, (2, 0, 0), basis_labels=('1', '2'))
    x, y = chart.symbols()
    psi = sp.Function('psi')(x, y)

    conn = spin_connection(frame)
    gens = gamma_frame_generators(frame, cliff)
    comps = spin_covariant_components(psi, frame, cliff, spin_conn=conn)
    Dpsi = dirac_operator(psi, frame, cliff, spin_conn=conn)

    assert len(gens) == 2
    assert len(comps) == 2
    assert Dpsi.has(sp.Derivative(psi, x)) or Dpsi.has(sp.Derivative(psi, y))


def test_exterior_nf_canonicalization_and_identities():
    x, y, z = sp.symbols('x y z')
    alpha = exterior_form_nf({(1, 0): x}, dimension=3, basis_labels=('dx', 'dy', 'dz'))
    beta = exterior_form_nf({(0,): z, (2,): 1}, dimension=3, basis_labels=('dx', 'dy', 'dz'))
    assert alpha.terms[(0, 1)] == -x

    wedge1 = wedge_exterior_forms(alpha, beta)
    wedge2 = wedge_exterior_forms(beta, alpha)
    assert wedge1.degree == 3
    assert wedge1.terms == wedge2.scale((-1) ** (alpha.degree * beta.degree)).terms

    report = exterior_identity_report(alpha, beta, (x, y, z))
    assert report.d_squared_zero
    assert report.graded_leibniz_holds
    assert report.associativity_holds


def test_basis_and_frame_roundtrip_archive(tmp_path):
    chart = coordinate_chart('Euclidean', 'Polar', 2)
    r, th = chart.symbols()
    frame = frame_basis('epolar', chart, lambda c: sp.Matrix([[1, 0], [0, 1 / c[0]]]), orthonormal=True)

    basis_payload = geometry_to_data(frame)
    rebuilt_basis = geometry_from_data(basis_payload)
    assert rebuilt_basis.name == frame.name
    assert rebuilt_basis.kind == frame.kind
    assert rebuilt_basis.chart.chart_name == frame.chart.chart_name
    assert sp.Matrix(rebuilt_basis.metadata['transform_to_chart'](rebuilt_basis.chart.symbols())) == sp.Matrix(frame.metadata['transform_to_chart'](frame.chart.symbols()))

    archive_path = tmp_path / 'exterior_geometry_archive.json'
    export_geometry_archive([frame], archive_path, metadata={'suite': 'exterior-geometry'})
    archive = import_geometry_archive(archive_path)
    assert archive.version == 'geometry-archive-v1'
    assert archive.metadata['suite'] == 'exterior-geometry'
    rebuilt = archive.objects[0]
    assert rebuilt.name == frame.name
