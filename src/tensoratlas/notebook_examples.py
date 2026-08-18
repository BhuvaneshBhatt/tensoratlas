"""Notebook-grade differential-geometry examples and regression fixtures.

The functions in this module are intentionally small, deterministic examples
that can be used both from notebooks and from regression tests.  They expose
known curvature components, zero reductions, TensorExpr canonical forms, and a
light-weight benchmark harness without requiring a full component-CAS runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import sympy as sp

from .connection_curvature import (
    canonicalize_geometry_ir,
    curvature_decomposition_ir,
    first_bianchi_identity_ir,
    second_bianchi_identity_ir,
    weyl_tensor_expr,
)
from .declarations import DeclarationRegistry, declaration_registry, standard_riemannian_registry
from .differential_forms_frame import (
    FrameCalculusPolicy,
    basis_one_form,
    canonicalize_wedge,
    curvature_two_form,
    exterior_covariant_derivative,
    hodge_star_form,
    torsion_two_form,
    wedge_forms,
)
from .semantic_ir import TensorExpr, canonical_ir_key, ir_node, scalar_ir, symbol_ir
from .tensor_expr_canonicalization import canonicalize_tensor_expr
from .variational_tensor_expr import einstein_hilbert_variation


@dataclass(frozen=True)
class GeometryExample:
    """Notebook/test fixture for a named geometry example."""

    name: str
    dimension: int
    coordinates: tuple[Any, ...]
    parameters: tuple[Any, ...] = ()
    metric: sp.Matrix | None = None
    known_components: Mapping[str, Any] = field(default_factory=dict)
    zero_reductions: Mapping[str, TensorExpr] = field(default_factory=dict)
    canonical_forms: Mapping[str, TensorExpr] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def known(self, key: str) -> Any:
        return self.known_components[key]

    def canonical_key(self, key: str) -> tuple[Any, ...]:
        return canonical_ir_key(self.canonical_forms[key])


@dataclass(frozen=True)
class RegressionBenchmarkResult:
    name: str
    seconds: float
    canonical_key: tuple[Any, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Declaration helpers


def _registry_for_dimension(dimension: int, *, signature: Sequence[int] | None = None) -> DeclarationRegistry:
    sig = tuple(signature) if signature is not None else tuple([-1] + [1] * (dimension - 1))
    reg = declaration_registry().declare_manifold("M", dimension, signature=sig)
    reg = reg.declare_bundle("TM", "M")
    reg = reg.declare_index_family("latin", "TM", symbols=tuple(chr(ord("a") + i) for i in range(max(4, dimension + 1))))
    reg = reg.declare_metric("g", "M", "TM", signature=sig)
    reg = reg.declare_connection("CD", "TM", metric="g", torsion_free=True, metric_compatible=True)
    return reg


# ---------------------------------------------------------------------------
# Component examples with known closed forms


def schwarzschild_curvature_example() -> GeometryExample:
    """Schwarzschild vacuum curvature fixture in coordinates (t, r, theta, phi).

    Known components use the common all-lowered convention for selected entries;
    Ricci, scalar curvature, and Einstein tensor vanish in vacuum outside r=0.
    """

    t, r, theta, phi, M = sp.symbols("t r theta phi M")
    f = 1 - 2 * M / r
    metric = sp.diag(-f, 1 / f, r**2, r**2 * sp.sin(theta) ** 2)
    reg = _registry_for_dimension(4)
    b1 = first_bianchi_identity_ir(reg, "CD", ("a", "b", "c", "d"))
    b2 = second_bianchi_identity_ir(reg, "CD", ("e", "a", "b", "c", "d"))
    return GeometryExample(
        name="Schwarzschild curvature",
        dimension=4,
        coordinates=(t, r, theta, phi),
        parameters=(M,),
        metric=metric,
        known_components={
            "R_lower_trtr": sp.simplify(2 * M / r**3),
            "R_lower_thetaphi_thetaphi": sp.simplify(2 * M * r * sp.sin(theta) ** 2),
            "Ricci_tt": sp.Integer(0),
            "Ricci_rr": sp.Integer(0),
            "ScalarCurvature": sp.Integer(0),
            "Einstein_tt": sp.Integer(0),
        },
        zero_reductions={
            "first_bianchi": canonicalize_geometry_ir(b1, reg).canonical,
            "second_bianchi": canonicalize_geometry_ir(b2, reg).canonical,
        },
        canonical_forms={
            "weyl_decomposition_n4": canonicalize_tensor_expr(curvature_decomposition_ir(reg, "CD", ("a", "b", "c", "d"))).canonical,
        },
        notes=("Vacuum Ricci-flat metric; nonzero Riemann/Weyl curvature.",),
    )


def flrw_curvature_example() -> GeometryExample:
    """FLRW metric fixture with covariant Einstein-tensor components."""

    t, r, theta, phi, k = sp.symbols("t r theta phi k")
    a = sp.Function("a")(t)
    adot = sp.diff(a, t)
    addot = sp.diff(a, t, 2)
    spatial = 1 - k * r**2
    metric = sp.diag(-1, a**2 / spatial, a**2 * r**2, a**2 * r**2 * sp.sin(theta) ** 2)
    pressure_factor = sp.simplify(-(2 * a * addot + adot**2 + k))
    reg = _registry_for_dimension(4)
    return GeometryExample(
        name="FLRW curvature and Einstein tensor",
        dimension=4,
        coordinates=(t, r, theta, phi),
        parameters=(k,),
        metric=metric,
        known_components={
            "G_tt": sp.simplify(3 * (adot**2 + k) / a**2),
            "G_rr": sp.simplify(pressure_factor / spatial),
            "G_thetatheta": sp.simplify(r**2 * pressure_factor),
            "G_phiphi": sp.simplify(r**2 * sp.sin(theta) ** 2 * pressure_factor),
            "Ricci_scalar": sp.simplify(6 * (a * addot + adot**2 + k) / a**2),
        },
        zero_reductions={
            "first_bianchi": canonicalize_geometry_ir(first_bianchi_identity_ir(reg, "CD", ("a", "b", "c", "d")), reg).canonical,
        },
        canonical_forms={
            "einstein_tensor_symbolic": ir_node("flrw:einstein_tensor", payload="G_ab", dimension=4),
        },
        notes=("Closed-form covariant Einstein-tensor components for signature (-,+,+,+).",),
    )


def two_sphere_curvature_example() -> GeometryExample:
    """Unit 2-sphere fixture."""

    theta, phi = sp.symbols("theta phi")
    metric = sp.diag(1, sp.sin(theta) ** 2)
    reg = _registry_for_dimension(2, signature=(1, 1))
    return GeometryExample(
        name="2-sphere curvature",
        dimension=2,
        coordinates=(theta, phi),
        metric=metric,
        known_components={
            "R_lower_thetaphi_thetaphi": sp.sin(theta) ** 2,
            "Ricci_thetatheta": sp.Integer(1),
            "Ricci_phiphi": sp.sin(theta) ** 2,
            "ScalarCurvature": sp.Integer(2),
            "GaussBonnet_density_identity": sp.Integer(0),
        },
        zero_reductions={
            "weyl_n2": canonicalize_geometry_ir(weyl_tensor_expr(reg, "CD", ("a", "b", "c", "d")), reg).canonical,
        },
        canonical_forms={
            "riemann_scalar_decomposition": canonicalize_tensor_expr(curvature_decomposition_ir(reg, "CD", ("a", "b", "c", "d"))).canonical,
        },
        notes=("Unit sphere has Gaussian curvature 1 and scalar curvature 2.",),
    )


def torsionful_toy_connection_example() -> GeometryExample:
    """Two-dimensional affine connection with T^x_{xy}=tau."""

    x, y, tau = sp.symbols("x y tau")
    reg = declaration_registry().declare_manifold("M", 2, signature=(1, 1))
    reg = reg.declare_bundle("TM", "M")
    reg = reg.declare_metric("g", "M", "TM", signature=(1, 1))
    reg = reg.declare_connection("D", "TM", metric="g", torsion_free=False, metric_compatible=True)
    torsion = torsion_two_form("D", "x")
    return GeometryExample(
        name="Torsionful toy connection",
        dimension=2,
        coordinates=(x, y),
        parameters=(tau,),
        known_components={
            "Gamma_x_xy": tau,
            "Gamma_x_yx": sp.Integer(0),
            "T_x_xy": tau,
            "T_x_yx": -tau,
        },
        zero_reductions={},
        canonical_forms={
            "torsion_two_form": canonicalize_tensor_expr(torsion).canonical,
        },
        notes=("Toy affine connection with antisymmetric torsion from Gamma^x_{xy}-Gamma^x_{yx}.",),
    )


# ---------------------------------------------------------------------------
# TensorExpr examples


def maxwell_forms_curved_background_example(dimension: int = 4) -> GeometryExample:
    """Maxwell forms on a curved background: F=dA, dF=0, d*F=J."""

    policy = FrameCalculusPolicy(dimension=dimension, signature=(-1,) + (1,) * (dimension - 1), orientation="positive")
    A = ir_node("form:potential", payload="A", degree=1, dimension=dimension)
    F = exterior_covariant_derivative(A, connection="d").with_metadata(name="F", maxwell_role="field_strength", degree=2)
    dF = exterior_covariant_derivative(F, connection="d").with_metadata(identity="maxwell_bianchi")
    starF = hodge_star_form(F, policy=policy)
    dstarF = exterior_covariant_derivative(starF, connection="d").with_metadata(equals="J", maxwell_role="source_equation")
    zero = ir_node("zero", reduced_from="d_d_A")
    return GeometryExample(
        name="Maxwell forms on curved background",
        dimension=dimension,
        coordinates=tuple(sp.symbols("x0:%d" % dimension)),
        zero_reductions={"dF": zero},
        canonical_forms={
            "F": canonicalize_tensor_expr(F).canonical,
            "hodge_F": canonicalize_tensor_expr(starF).canonical,
            "d_hodge_F": canonicalize_tensor_expr(dstarF).canonical,
        },
        notes=("Exterior calculus form: F=dA, dF=0, and d*F=J on a curved oriented background.",),
    )


def einstein_hilbert_variation_example() -> GeometryExample:
    reg = standard_riemannian_registry()
    report = einstein_hilbert_variation(reg)
    return GeometryExample(
        name="Einstein-Hilbert variation",
        dimension=4,
        coordinates=tuple(sp.symbols("x0:4")),
        zero_reductions={"boundary_removed": ir_node("zero", reduced_from="discarded_boundary_terms")},
        canonical_forms={
            "raw_variation": canonicalize_tensor_expr(report.raw_variation).canonical,
            "euler_lagrange": canonicalize_tensor_expr(report.euler_lagrange).canonical,
        },
        known_components={"field_equation_tensor": "Einstein"},
        notes=("Euler-Lagrange extraction returns the Einstein tensor density plus tracked boundary terms.",),
    )


def gauss_bonnet_low_dimension_example(dimension: int = 2) -> GeometryExample:
    if dimension not in (2, 3, 4):
        raise ValueError("Gauss-Bonnet fixture currently supports dimensions 2, 3, and 4")
    reg = _registry_for_dimension(dimension, signature=(1,) * dimension)
    if dimension == 2:
        identity_value = sp.Integer(0)
        canonical = ir_node("zero", identity="gauss_bonnet_2d_quadratic_reduction")
    elif dimension == 3:
        identity_value = "Weyl=0; Riemann determined by Ricci and scalar curvature"
        canonical = canonicalize_geometry_ir(weyl_tensor_expr(reg, "CD", ("a", "b", "c", "d")), reg).canonical
    else:
        identity_value = "Euler density is nontrivial and topological after integration"
        canonical = ir_node("gauss_bonnet:euler_density", dimension=4)
    return GeometryExample(
        name=f"Gauss-Bonnet identities in n={dimension}",
        dimension=dimension,
        coordinates=tuple(sp.symbols("x0:%d" % dimension)),
        known_components={"low_dimension_identity": identity_value},
        zero_reductions={"weyl_or_quadratic_identity": canonical if canonical.kind == "zero" else ir_node("zero", reduced_from="dimension_identity")},
        canonical_forms={"gauss_bonnet_form": canonical},
    )


def bianchi_identity_reduction_example(dimension: int = 4) -> GeometryExample:
    reg = _registry_for_dimension(dimension)
    first = canonicalize_geometry_ir(first_bianchi_identity_ir(reg, "CD", ("a", "b", "c", "d")), reg).canonical
    second = canonicalize_geometry_ir(second_bianchi_identity_ir(reg, "CD", ("e", "a", "b", "c", "d")), reg).canonical
    return GeometryExample(
        name="Bianchi identity reductions",
        dimension=dimension,
        coordinates=tuple(sp.symbols("x0:%d" % dimension)),
        zero_reductions={"first_bianchi": first, "second_bianchi": second},
        canonical_forms={"first_bianchi": first, "second_bianchi": second},
    )


def weyl_decomposition_example(dimension: int) -> GeometryExample:
    if dimension not in (3, 4):
        raise ValueError("Weyl decomposition fixture is intended for n=3 or n=4")
    reg = _registry_for_dimension(dimension)
    decomp = canonicalize_tensor_expr(curvature_decomposition_ir(reg, "CD", ("a", "b", "c", "d"))).canonical
    zero = canonicalize_geometry_ir(weyl_tensor_expr(reg, "CD", ("a", "b", "c", "d")), reg).canonical
    return GeometryExample(
        name=f"Weyl decomposition in n={dimension}",
        dimension=dimension,
        coordinates=tuple(sp.symbols("x0:%d" % dimension)),
        known_components={"Weyl_vanishes": dimension < 4},
        zero_reductions={"weyl_zero" if dimension < 4 else "no_zero_expected": zero},
        canonical_forms={"riemann_decomposition": decomp},
    )


# ---------------------------------------------------------------------------
# Regression benchmark harness


def run_regression_benchmarks(repeat: int = 1) -> tuple[RegressionBenchmarkResult, ...]:
    """Run light deterministic benchmarks used by notebooks and CI smoke tests."""

    tasks: tuple[tuple[str, Callable[[], TensorExpr]], ...] = (
        ("schwarzschild_bianchi", lambda: schwarzschild_curvature_example().zero_reductions["first_bianchi"]),
        ("sphere_decomposition", lambda: two_sphere_curvature_example().canonical_forms["riemann_scalar_decomposition"]),
        ("maxwell_wedge", lambda: maxwell_forms_curved_background_example().canonical_forms["F"]),
        ("eh_variation", lambda: einstein_hilbert_variation_example().canonical_forms["euler_lagrange"]),
        ("weyl_n3", lambda: weyl_decomposition_example(3).zero_reductions["weyl_zero"]),
        ("weyl_n4", lambda: weyl_decomposition_example(4).canonical_forms["riemann_decomposition"]),
    )
    results: list[RegressionBenchmarkResult] = []
    for name, fn in tasks:
        start = perf_counter()
        expr = None
        for _ in range(max(1, repeat)):
            expr = fn()
        elapsed = perf_counter() - start
        results.append(RegressionBenchmarkResult(name=name, seconds=elapsed, canonical_key=canonical_ir_key(expr) if expr is not None else None))
    return tuple(results)


def all_notebook_examples() -> tuple[GeometryExample, ...]:
    return (
        schwarzschild_curvature_example(),
        flrw_curvature_example(),
        two_sphere_curvature_example(),
        torsionful_toy_connection_example(),
        maxwell_forms_curved_background_example(),
        einstein_hilbert_variation_example(),
        gauss_bonnet_low_dimension_example(2),
        gauss_bonnet_low_dimension_example(3),
        bianchi_identity_reduction_example(),
        weyl_decomposition_example(3),
        weyl_decomposition_example(4),
    )

# ---------------------------------------------------------------------------
# Executable notebook workflow helpers backed by the public lightweight layers


def electromagnetic_tensor_valued_workflow() -> Mapping[str, Any]:
    """Small executable electromagnetic/gauge-form workflow for notebooks/tests."""

    from .tensor_valued_forms import TensorValuedForm, gauge_curvature

    A = TensorValuedForm(1, (1, -1), {("U1", "U1"): sp.Symbol("A")}, label="A")
    F = gauge_curvature(A)
    return {"potential": A, "curvature": F, "degree": F.degree, "variance": F.variance}


def two_sphere_relativity_workflow() -> Mapping[str, Any]:
    """Executable 2-sphere curvature workflow using relativity helpers."""

    from .relativity import ricci_tensor, scalar_curvature, two_sphere_metric

    model = two_sphere_metric()
    ric = ricci_tensor(model)
    scalar = scalar_curvature(model, ricci=ric)
    radius = model.parameters[0]
    return {"model": model, "ricci": ric, "scalar_curvature": scalar, "expected_scalar": 2 / radius**2}


def cartan_frame_workflow() -> Mapping[str, Any]:
    """Executable frame/vielbein Cartan-equation workflow."""

    from .tensor_valued_forms import cartan_first_equation, cartan_second_equation, connection_form, solder_form

    labels = (0, 1)
    theta = solder_form(labels)
    omega = connection_form(labels)
    torsion = cartan_first_equation(theta, omega)
    curvature = cartan_second_equation(omega)
    return {"theta": theta, "omega": omega, "torsion": torsion, "curvature": curvature}


def schwarzschild_relativity_workflow() -> Mapping[str, Any]:
    """Executable Schwarzschild workflow with selective Ricci/scalar checks."""

    from .relativity import nonzero_components, ricci_tensor, scalar_curvature, schwarzschild_metric

    model = schwarzschild_metric()
    ric = ricci_tensor(model)
    scalar = scalar_curvature(model, ricci=ric)
    return {"model": model, "ricci": ric, "scalar_curvature": scalar, "nonzero_ricci": nonzero_components(ric)}


def flrw_relativity_workflow() -> Mapping[str, Any]:
    """Executable FLRW workflow exposing a representative Christoffel component."""

    from .relativity import christoffel_component, flrw_metric

    model = flrw_metric()
    gamma_r_tr = christoffel_component(model, 1, 0, 1)
    return {"model": model, "Gamma^r_tr": gamma_r_tr}


def geometric_algebra_workflow() -> Mapping[str, Any]:
    """Executable geometric-algebra workflow for notebooks/tests."""

    from .geometric_algebra import GeometricAlgebra

    algebra = GeometricAlgebra.euclidean(3)
    e1, e2, e3 = algebra.basis_vectors()
    bivector = e1.wedge(e2)
    pseudoscalar = e1 * e2 * e3
    return {"algebra": algebra, "e1_squared": e1 * e1, "bivector": bivector, "pseudoscalar": pseudoscalar}


def executable_notebook_workflows() -> Mapping[str, Mapping[str, Any]]:
    """Return all executable workflow fragments used by the demo notebook."""

    return {
        "electromagnetism": electromagnetic_tensor_valued_workflow(),
        "two_sphere": two_sphere_relativity_workflow(),
        "cartan": cartan_frame_workflow(),
        "schwarzschild": schwarzschild_relativity_workflow(),
        "flrw": flrw_relativity_workflow(),
        "geometric_algebra": geometric_algebra_workflow(),
    }
