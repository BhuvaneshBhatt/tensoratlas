from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .abstract_tensor import (
    IndexType,
    Metric,
    TensorHead,
    connection,
    fully_antisymmetric_head,
    fully_symmetric_head,
    get_curvature_identity_library,
    index_type,
    metric,
    ricci_tensor_head,
    riemann_tensor_head,
    schouten_tensor_head,
    torsion,
    weyl_tensor_head,
)


@dataclass(frozen=True)
class StandardObjectSpec:
    name: str
    category: str
    object: Any
    description: str


@dataclass(frozen=True)
class StandardIdentitySpec:
    name: str
    library_name: str
    identities: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class StandardTensorLibrary:
    name: str
    objects: Mapping[str, StandardObjectSpec] = field(default_factory=dict)
    identities: Mapping[str, StandardIdentitySpec] = field(default_factory=dict)



def _riemannian_library() -> StandardTensorLibrary:
    tangent = index_type("M", dimension=None, metric_symmetry=1, metric_name="g")
    g = metric(tangent, name="g")
    eps = fully_antisymmetric_head("epsilon", [tangent.to_sympy()]*4)
    sym2 = fully_symmetric_head("S", [tangent.to_sympy()]*2)
    riem = riemann_tensor_head("Riemann", tangent.to_sympy())
    ric = ricci_tensor_head("Ricci", tangent.to_sympy())
    weyl = weyl_tensor_head("Weyl", tangent.to_sympy())
    sch = schouten_tensor_head("Schouten", tangent.to_sympy())
    lc = connection("LeviCivita", tangent, metric=g, metric_compatible=True)
    tors = torsion("Torsion", tangent)
    objects = {
        "index_type": StandardObjectSpec("index_type", "bundle", tangent, "Default tangent-bundle abstract index type."),
        "metric": StandardObjectSpec("metric", "metric", g, "Standard symmetric metric object."),
        "epsilon": StandardObjectSpec("epsilon", "tensor", eps, "Canonical antisymmetric epsilon tensor head."),
        "symmetric_rank2": StandardObjectSpec("symmetric_rank2", "tensor", sym2, "Canonical symmetric rank-2 head."),
        "riemann": StandardObjectSpec("riemann", "curvature", riem, "Standard Riemann curvature tensor head."),
        "ricci": StandardObjectSpec("ricci", "curvature", ric, "Standard Ricci tensor head."),
        "weyl": StandardObjectSpec("weyl", "curvature", weyl, "Standard Weyl tensor head."),
        "schouten": StandardObjectSpec("schouten", "curvature", sch, "Standard Schouten tensor head."),
        "levi_civita_connection": StandardObjectSpec("levi_civita_connection", "connection", lc, "Metric-compatible Levi-Civita-style connection object."),
        "torsion": StandardObjectSpec("torsion", "connection", tors, "Canonical torsion object."),
    }
    ids = {
        "core": StandardIdentitySpec("core", "core", tuple(get_curvature_identity_library("core").identities), "Core algebraic curvature identities."),
        "differential": StandardIdentitySpec("differential", "differential", tuple(get_curvature_identity_library("differential").identities), "Differential curvature identities."),
        "full": StandardIdentitySpec("full", "full", tuple(get_curvature_identity_library("full").identities), "Combined curvature identity library."),
    }
    return StandardTensorLibrary("riemannian_geometry", objects=objects, identities=ids)



def _lorentzian_library() -> StandardTensorLibrary:
    tangent = index_type("L", dimension=4, metric_symmetry=1, metric_name="eta")
    eta = metric(tangent, name="eta")
    riem = riemann_tensor_head("RiemannL", tangent.to_sympy())
    ric = ricci_tensor_head("RicciL", tangent.to_sympy())
    weyl = weyl_tensor_head("WeylL", tangent.to_sympy())
    sch = schouten_tensor_head("SchoutenL", tangent.to_sympy())
    conn = connection("LeviCivitaLorentz", tangent, metric=eta, metric_compatible=True)
    objects = {
        "index_type": StandardObjectSpec("index_type", "bundle", tangent, "Four-dimensional Lorentzian index type."),
        "metric": StandardObjectSpec("metric", "metric", eta, "Lorentzian metric object."),
        "riemann": StandardObjectSpec("riemann", "curvature", riem, "Lorentzian Riemann curvature head."),
        "ricci": StandardObjectSpec("ricci", "curvature", ric, "Lorentzian Ricci tensor head."),
        "weyl": StandardObjectSpec("weyl", "curvature", weyl, "Lorentzian Weyl tensor head."),
        "schouten": StandardObjectSpec("schouten", "curvature", sch, "Lorentzian Schouten tensor head."),
        "levi_civita_connection": StandardObjectSpec("levi_civita_connection", "connection", conn, "Lorentzian Levi-Civita connection object."),
    }
    ids = {
        "core": StandardIdentitySpec("core", "core", tuple(get_curvature_identity_library("core").identities), "Core algebraic curvature identities."),
        "full": StandardIdentitySpec("full", "full", tuple(get_curvature_identity_library("full").identities), "Combined Lorentzian curvature identities."),
    }
    return StandardTensorLibrary("lorentzian_geometry", objects=objects, identities=ids)


def _symplectic_library() -> StandardTensorLibrary:
    tangent = index_type("S", dimension=None, metric_symmetry=1, metric_name="omega")
    omega = metric(tangent, name="omega")
    eps = fully_antisymmetric_head("omega_form", [tangent.to_sympy()] * 2)
    sym2 = fully_symmetric_head("HamiltonianHessian", [tangent.to_sympy()] * 2)
    conn = connection("SymplecticConnection", tangent, metric=omega, metric_compatible=True)
    objects = {
        "index_type": StandardObjectSpec("index_type", "bundle", tangent, "Canonical symplectic bundle index type."),
        "form": StandardObjectSpec("form", "tensor", eps, "Canonical symplectic two-form head."),
        "metric": StandardObjectSpec("metric", "metric", omega, "Symplectic-form-backed metric placeholder."),
        "symmetric_rank2": StandardObjectSpec("symmetric_rank2", "tensor", sym2, "Symmetric Hamiltonian Hessian head."),
        "connection": StandardObjectSpec("connection", "connection", conn, "Symplectic connection object."),
    }
    ids = {
        "core": StandardIdentitySpec("core", "core", tuple(get_curvature_identity_library("core").identities), "Core tensor identities reusable in symplectic workflows."),
    }
    return StandardTensorLibrary("symplectic_geometry", objects=objects, identities=ids)

_STANDARD_LIBRARIES = {
    "riemannian_geometry": _riemannian_library(),
    "lorentzian_geometry": _lorentzian_library(),
    "symplectic_geometry": _symplectic_library(),
}


def list_standard_tensor_libraries() -> tuple[str, ...]:
    return tuple(sorted(_STANDARD_LIBRARIES))



def get_standard_tensor_library(name: str = "riemannian_geometry") -> StandardTensorLibrary:
    try:
        return _STANDARD_LIBRARIES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown standard tensor library: {name!r}") from exc



def standard_object(name: str, *, library: str = "riemannian_geometry") -> Any:
    lib = get_standard_tensor_library(library)
    try:
        return lib.objects[name].object
    except KeyError as exc:
        raise ValueError(f"Unknown standard object {name!r} in library {library!r}") from exc



def standard_identity_library(name: str = "full", *, library: str = "riemannian_geometry"):
    lib = get_standard_tensor_library(library)
    try:
        spec = lib.identities[name]
    except KeyError as exc:
        raise ValueError(f"Unknown standard identity library {name!r} in library {library!r}") from exc
    return get_curvature_identity_library(spec.library_name)


__all__ = [
    "StandardObjectSpec",
    "StandardIdentitySpec",
    "StandardTensorLibrary",
    "list_standard_tensor_libraries",
    "get_standard_tensor_library",
    "standard_object",
    "standard_identity_library",
]
