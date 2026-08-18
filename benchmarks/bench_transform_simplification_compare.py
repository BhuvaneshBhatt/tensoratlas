from __future__ import annotations

import pathlib
import sys

import sympy as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from _common import print_report, run_case
from tensoratlas import (
    CleanupPhase,
    basis_transformation_matrix,
    basis_transformation_matrix_tnf,
    cleanup_phase,
    coordinate_chart,
    coordinate_map,
    cotangent_basis,
    gradient,
    matrix_equal,
    orthonormal_tangent_basis,
    scalar_equal,
    tangent_basis,
)
from tensoratlas.fields import ScalarField


def _build_geometry():
    polar = coordinate_chart("Euclidean", "Polar", 2)
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    mapping = coordinate_map(polar, cart)
    r, theta = polar.symbols()
    scalar = ScalarField(polar, r**2 * sp.sin(theta) + r * sp.cos(theta))
    return polar, cart, mapping, scalar


def _cross_chart_tangent_tnf():
    polar, cart, _, _ = _build_geometry()
    return basis_transformation_matrix_tnf(tangent_basis(polar), tangent_basis(cart))


def _cross_chart_tangent_sympy():
    polar, cart, _, _ = _build_geometry()
    return basis_transformation_matrix(tangent_basis(polar), tangent_basis(cart))


def _cross_chart_orthonormal_tnf():
    polar, cart, _, _ = _build_geometry()
    return basis_transformation_matrix_tnf(orthonormal_tangent_basis(polar), tangent_basis(cart))


def _cross_chart_cotangent_tnf():
    polar, cart, _, _ = _build_geometry()
    return basis_transformation_matrix_tnf(cotangent_basis(polar), cotangent_basis(cart))


def _gradient_workflow():
    _, _, _, scalar = _build_geometry()
    return gradient(scalar)


def _cleanup_structural():
    x = sp.Symbol("x")
    return cleanup_phase((x**2 - 1) / (x - 1), CleanupPhase.STRUCTURAL)


def _cleanup_presentation():
    x = sp.Symbol("x")
    return cleanup_phase((x**2 - 1) / (x - 1), CleanupPhase.PRESENTATION)


def _matrix_equality():
    polar, cart, _, _ = _build_geometry()
    left = basis_transformation_matrix_tnf(tangent_basis(polar), tangent_basis(cart))
    right = basis_transformation_matrix(tangent_basis(polar), tangent_basis(cart))
    return matrix_equal(left, right)


def _scalar_equality():
    x = sp.Symbol("x")
    return scalar_equal((x**2 - 1) / (x - 1), x + 1)


if __name__ == "__main__":
    print_report(
        run_case("cross_chart_tangent_tnf", _cross_chart_tangent_tnf, repeat=200),
        run_case("cross_chart_tangent_sympy", _cross_chart_tangent_sympy, repeat=200),
        run_case("cross_chart_orthonormal_tnf", _cross_chart_orthonormal_tnf, repeat=200),
        run_case("cross_chart_cotangent_tnf", _cross_chart_cotangent_tnf, repeat=200),
        run_case("gradient_workflow", _gradient_workflow, repeat=100),
        run_case("cleanup_structural", _cleanup_structural, repeat=400),
        run_case("cleanup_presentation", _cleanup_presentation, repeat=200),
        run_case("matrix_equality", _matrix_equality, repeat=250),
        run_case("scalar_equality", _scalar_equality, repeat=500),
    )
