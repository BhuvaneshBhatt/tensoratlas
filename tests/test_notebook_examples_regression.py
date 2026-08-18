from __future__ import annotations

import sympy as sp

from tensoratlas.notebook_examples import (
    bianchi_identity_reduction_example,
    einstein_hilbert_variation_example,
    flrw_curvature_example,
    gauss_bonnet_low_dimension_example,
    maxwell_forms_curved_background_example,
    run_regression_benchmarks,
    schwarzschild_curvature_example,
    torsionful_toy_connection_example,
    two_sphere_curvature_example,
    weyl_decomposition_example,
)


def test_schwarzschild_vacuum_components_and_bianchi_zero():
    ex = schwarzschild_curvature_example()
    assert ex.known("Ricci_tt") == 0
    assert ex.known("ScalarCurvature") == 0
    assert ex.known("Einstein_tt") == 0
    assert ex.known("R_lower_trtr") != 0
    assert ex.zero_reductions["first_bianchi"].kind == "zero"
    assert ex.zero_reductions["second_bianchi"].kind == "zero"
    assert ex.canonical_forms["weyl_decomposition_n4"].metadata["dimension"] == 4


def test_flrw_known_einstein_components_are_symbolic_and_bianchi_zero():
    ex = flrw_curvature_example()
    t = ex.coordinates[0]
    k = ex.parameters[0]
    a = sp.Function("a")(t)
    expected_gtt = 3 * (sp.diff(a, t) ** 2 + k) / a**2
    assert sp.simplify(ex.known("G_tt") - expected_gtt) == 0
    assert sp.simplify(ex.known("Ricci_scalar") - 6 * (a * sp.diff(a, t, 2) + sp.diff(a, t) ** 2 + k) / a**2) == 0
    assert ex.zero_reductions["first_bianchi"].kind == "zero"


def test_two_sphere_known_curvature_and_low_dimension_weyl_zero():
    ex = two_sphere_curvature_example()
    theta = ex.coordinates[0]
    assert sp.simplify(ex.known("R_lower_thetaphi_thetaphi") - sp.sin(theta) ** 2) == 0
    assert sp.simplify(ex.known("Ricci_thetatheta") - 1) == 0
    assert sp.simplify(ex.known("ScalarCurvature") - 2) == 0
    assert ex.zero_reductions["weyl_n2"].kind == "zero"
    assert ex.canonical_forms["riemann_scalar_decomposition"].metadata["dimension"] == 2


def test_torsionful_toy_connection_known_components_and_form():
    ex = torsionful_toy_connection_example()
    tau = ex.parameters[0]
    assert ex.known("T_x_xy") == tau
    assert ex.known("T_x_yx") == -tau
    assert ex.canonical_forms["torsion_two_form"].kind == "torsion:two_form"


def test_maxwell_forms_on_curved_background_have_bianchi_zero_and_hodge_form():
    ex = maxwell_forms_curved_background_example()
    assert ex.zero_reductions["dF"].kind == "zero"
    assert ex.canonical_forms["F"].metadata["maxwell_role"] == "field_strength"
    assert "hodge_dual_of" in ex.canonical_forms["hodge_F"].metadata
    assert ex.canonical_forms["d_hodge_F"].metadata["maxwell_role"] == "source_equation"


def test_einstein_hilbert_variation_produces_euler_lagrange_and_boundary_zero_fixture():
    ex = einstein_hilbert_variation_example()
    assert ex.zero_reductions["boundary_removed"].kind == "zero"
    assert "euler_lagrange" in ex.canonical_forms
    assert ex.known("field_equation_tensor") == "Einstein"


def test_gauss_bonnet_low_dimension_identities():
    ex2 = gauss_bonnet_low_dimension_example(2)
    ex3 = gauss_bonnet_low_dimension_example(3)
    assert ex2.known("low_dimension_identity") == 0
    assert ex2.zero_reductions["weyl_or_quadratic_identity"].kind == "zero"
    assert ex3.zero_reductions["weyl_or_quadratic_identity"].kind == "zero"


def test_bianchi_identity_reductions_are_zero():
    ex = bianchi_identity_reduction_example()
    assert ex.zero_reductions["first_bianchi"].kind == "zero"
    assert ex.zero_reductions["second_bianchi"].kind == "zero"


def test_weyl_decomposition_in_n3_and_n4():
    n3 = weyl_decomposition_example(3)
    n4 = weyl_decomposition_example(4)
    assert n3.known("Weyl_vanishes") is True
    assert n3.zero_reductions["weyl_zero"].kind == "zero"
    assert n4.known("Weyl_vanishes") is False
    assert n4.canonical_forms["riemann_decomposition"].metadata["dimension"] == 4


def test_regression_benchmark_harness_runs_and_returns_canonical_keys():
    results = run_regression_benchmarks()
    assert {r.name for r in results} >= {"schwarzschild_bianchi", "sphere_decomposition", "eh_variation", "weyl_n3", "weyl_n4"}
    assert all(r.seconds >= 0 for r in results)
    assert all(r.canonical_key is not None for r in results)
