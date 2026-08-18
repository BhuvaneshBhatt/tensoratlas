"""Richer coordinate-transform data and transition-graph utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Mapping, Sequence

from .coordinate_tools import (
    CoordinateMap,
    CoordinateSingularity,
    coordinate_domain_assumptions,
    coordinate_map_between,
    infer_coordinate_singularities,
    list_standard_coordinates,
    standard_coordinate_entry,
    standard_coordinate_map_to_cartesian,
)
from .manifolds import Manifold, TensorKernelError

Scalar = Any


def _require_sympy():
    try:
        import sympy as sp  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise TensorKernelError("Coordinate-transform data operations require SymPy.") from exc
    return sp


@dataclass(frozen=True, slots=True)
class InverseBranch:
    """One local inverse branch for a symbolic coordinate map."""

    expressions: tuple[Scalar, ...]
    conditions: tuple[Scalar, ...] = ()
    name: str = "principal"
    notes: str = ""


@dataclass(frozen=True, slots=True)
class CoordinateMapData:
    """Computed properties for a coordinate map."""

    coordinate_map: CoordinateMap
    properties: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def property(self, name: str) -> Any:
        if name in self.properties:
            return self.properties[name]
        raise TensorKernelError(f"Unknown coordinate-transform property {name!r}.")

    def as_dict(self) -> dict[str, Any]:
        return dict(self.properties)


def solve_inverse_branches(cmap: CoordinateMap, *, max_branches: int = 8) -> tuple[InverseBranch, ...]:
    """Attempt symbolic inverse solving for a coordinate map.

    Catalogued inverse branches are returned first.  When none is present, this
    uses SymPy's solver conservatively and records each solution as a local
    branch without pretending global uniqueness.
    """
    sp = _require_sympy()
    if cmap.inverse is not None:
        return (InverseBranch(cmap.inverse, (cmap.is_locally_invertible_condition(),), "catalogued"),)
    src = cmap.source_symbols
    tgt = cmap.target_symbols
    equations = [sp.Eq(fwd, target) for fwd, target in zip(cmap.forward, tgt)]
    try:
        solutions = sp.solve(equations, src, dict=True)
    except Exception:
        return tuple()
    branches: list[InverseBranch] = []
    for pos, soln in enumerate(solutions[:max_branches]):
        if all(symbol in soln for symbol in src):
            branches.append(InverseBranch(tuple(soln[symbol] for symbol in src), (cmap.is_locally_invertible_condition(),), f"solved_{pos}"))
    return tuple(branches)


def coordinate_map_data(
    cmap: CoordinateMap,
    *,
    include_inverse_jacobian: bool = True,
    include_inverse_branches: bool = True,
) -> CoordinateMapData:
    """Return a rich property bundle for a coordinate map.

    Expensive inverse-Jacobian and inverse-branch computations can be disabled
    for lightweight summaries.
    """
    sp = _require_sympy()
    jac = cmap.jacobian()
    inv_jac = cmap.inverse_jacobian() if include_inverse_jacobian else None
    if cmap.source.dimension == cmap.target.dimension:
        raw_det = sp.Matrix(jac).det()
        if raw_det.has(sp.atan2) or raw_det.has(sp.Abs):
            try:
                jac_det = sp.factor(raw_det)
            except Exception:
                jac_det = raw_det
        else:
            try:
                factored = sp.factor(raw_det)
                if hasattr(factored, "count_ops") and factored.count_ops() <= 64:
                    jac_det = sp.factor(sp.trigsimp(factored))
                else:
                    jac_det = factored
            except Exception:
                jac_det = raw_det
    else:
        jac_det = None
    inverse_branches = solve_inverse_branches(cmap) if include_inverse_branches else tuple()
    singularities = infer_coordinate_singularities(jacobian_det=jac_det, known=cmap.singularities)
    props = {
        "source": cmap.source.name,
        "target": cmap.target.name,
        "source_coordinates": cmap.source.coordinate_names,
        "target_coordinates": cmap.target.coordinate_names,
        "forward": cmap.forward,
        "inverse": cmap.inverse,
        "inverse_branches": inverse_branches,
        "jacobian": jac,
        "inverse_jacobian": inv_jac,
        "jacobian_determinant": jac_det,
        "local_invertibility_condition": sp.Ne(jac_det, 0) if jac_det is not None else None,
        "orientation": sp.sign(jac_det) if jac_det is not None else None,
        "singularities": singularities,
        "domain_notes": dict(cmap.domain_notes),
    }
    return CoordinateMapData(cmap, props)


def standard_coordinate_system_data(
    name: str,
    *,
    include_inverse_metric: bool = True,
    include_transform_properties: bool = True,
) -> dict[str, Any]:
    """Return richer data properties for a catalogued coordinate system.

    Expensive inverse metric and transform-property computations can be
    disabled for lightweight metadata inspection.
    """
    sp = _require_sympy()
    entry = standard_coordinate_entry(name)
    coords = entry.symbols()
    metric = entry.metric(coords)
    det = sp.factor(sp.Matrix(metric).det())
    mapped = None
    transform_props = None
    try:
        mapped = entry.map_to_cartesian()
        if include_transform_properties:
            transform_props = coordinate_map_data(mapped).as_dict()
    except Exception:
        transform_props = None
    return {
        "name": entry.name,
        "dimension": entry.dimension,
        "coordinates": coords,
        "domains": dict(entry.domains),
        "assumptions": coordinate_domain_assumptions(entry, coords),
        "metric": metric,
        "inverse_metric": sp.Matrix(metric).inv() if include_inverse_metric and det != 0 else None,
        "metric_determinant": det,
        "sqrt_abs_metric_determinant": sp.sqrt(sp.Abs(det)),
        "singularities": infer_coordinate_singularities(metric=metric, coordinates=coords, known=entry.singularities(coords)),
        "to_cartesian": mapped,
        "transform_properties": transform_props,
        "description": entry.description,
    }


def coordinate_transform_graph() -> dict[str, tuple[str, ...]]:
    """Return a conservative graph of catalogued transition-map availability."""
    names = list_standard_coordinates()
    graph: dict[str, set[str]] = {name: set() for name in names}
    for name in names:
        entry = standard_coordinate_entry(name)
        if entry.to_cartesian_builder is not None:
            cart_name = f"cartesian{entry.dimension}"
            if cart_name in graph and cart_name != name:
                graph[name].add(cart_name)
                if entry.from_cartesian_builder is not None:
                    graph[cart_name].add(name)
    for a, b in (("spherical", "cylindrical"), ("cylindrical", "spherical")):
        if a in graph and b in graph:
            graph[a].add(b)
    return {name: tuple(sorted(targets)) for name, targets in graph.items()}


def catalog_transition_map(source: str, target: str, *, manifold: Manifold | None = None) -> CoordinateMap:
    """Return a transition map using the catalog transform graph."""
    return coordinate_map_between(source, target, manifold=manifold)
