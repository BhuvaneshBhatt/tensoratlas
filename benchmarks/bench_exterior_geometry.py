from __future__ import annotations

import sympy as sp

from tensoratlas import (
    coordinate_chart,
    clifford_algebra,
    orthonormal_tangent_basis,
    spin_connection,
    dirac_operator,
    exterior_form_nf,
    exterior_derivative_nf,
    export_geometry_archive,
    import_geometry_archive,
)


def benchmark_spin_connection():
    chart = coordinate_chart('Euclidean', 'Polar', 2)
    frame = orthonormal_tangent_basis(chart)
    return spin_connection(frame)


def benchmark_dirac_operator(tmp_path=None):
    chart = coordinate_chart('Euclidean', 'Cartesian', 2)
    frame = orthonormal_tangent_basis(chart)
    cliff = clifford_algebra(2, (2, 0, 0), basis_labels=('1', '2'))
    x, y = chart.symbols()
    psi = sp.Function('psi')(x, y)
    return dirac_operator(psi, frame, cliff)


def benchmark_exterior_nf():
    x, y, z = sp.symbols('x y z')
    omega = exterior_form_nf({(1, 0): x, (2,): y}, dimension=3)
    return exterior_derivative_nf(omega, (x, y, z))


def benchmark_archive_roundtrip(tmp_path):
    chart = coordinate_chart('Euclidean', 'Polar', 2)
    frame = orthonormal_tangent_basis(chart)
    path = tmp_path / 'exterior_geometry_geom.json'
    export_geometry_archive([frame], path)
    return import_geometry_archive(path)
