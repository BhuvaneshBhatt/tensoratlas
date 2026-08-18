from __future__ import annotations

import sympy as sp

from tensoratlas.examples.physical_tensors import physical_tensor_workflow, quadrupole_moment_disk_example, stress_strain_stiffness_example
from tensoratlas.examples.tensor_theory import (
    basis_change_example,
    dual_basis_example,
    metric_pullback_example,
    multilinear_metric_example,
    tensor_product_contraction_example,
    tensor_theory_workflow,
)


def test_dual_basis_gives_kronecker_delta():
    result = dual_basis_example()
    assert result["kronecker_delta"] == sp.eye(2)


def test_basis_change_preserves_pairing_and_linear_map_action():
    result = basis_change_example()
    assert result["pairing_difference"] == 0
    assert result["linear_map_consistency"] == sp.zeros(2, 1)


def test_metric_example_is_bilinear():
    result = multilinear_metric_example()
    assert result["linearity_residual"] == 0


def test_polar_metric_pullback():
    rho, theta = sp.symbols("rho theta", positive=True)
    result = metric_pullback_example()
    assert result["polar_metric"] == sp.Matrix([[1, 0], [0, rho**2]])


def test_tensor_product_contraction_examples():
    result = tensor_product_contraction_example()
    assert result["elementary_tensor"] == ((3, 4), (6, 8))
    assert result["trace"] == 5
    assert result["contracted_product"] == ((7, 10), (15, 22))


def test_quadrupole_disk_is_traceless():
    result = quadrupole_moment_disk_example()
    assert sp.simplify(result["trace"]) == 0
    assert result["quadrupole_tensor"].shape == (3, 3)


def test_stress_strain_reconstruction():
    result = stress_strain_stiffness_example()
    assert result["reconstruction_residual"] == sp.zeros(2, 2)


def test_workflow_collections_are_complete():
    tensor = tensor_theory_workflow()
    physical = physical_tensor_workflow()
    assert "basis_change" in tensor
    assert "quadrupole" in physical
