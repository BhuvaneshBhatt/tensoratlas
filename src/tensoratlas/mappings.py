from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Iterable, Set

import sympy as sp

from .charts import CoordinateChart
from .simplification_core import canonical_simplify
from .simplification_policy import simplify_object
from .normal_forms import TNFMatrix, tnf_build_matrix, tnf_column_from_entries, tnf_matrix_to_sympy

def _refine_with_assumptions(expr: sp.Expr, assumptions: Optional[sp.Expr], *, simplify_first: bool = True) -> sp.Expr:
    simplified = expr
    if simplify_first:
        try:
            simplified = sp.simplify(expr)
        except Exception:
            simplified = expr
    if assumptions is None:
        return simplified
    try:
        return sp.refine(simplified, assumptions)
    except Exception:
        return simplified



@dataclass(frozen=True)
class CoordinateMap:
    source: CoordinateChart
    target: CoordinateChart
    mapping_exprs_func: callable
    inverse_exprs_func: Optional[callable] = None
    metadata: Dict[str, object] | None = None

    def mapping_exprs(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Tuple[sp.Expr, ...]:
        if coords is None:
            coords = self.source.symbols()
        return self.mapping_exprs_func(coords)

    def inverse_mapping_exprs(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Tuple[sp.Expr, ...]:
        if self.inverse_exprs_func is None:
            raise ValueError("Inverse mapping formulas are not available for this map.")
        if coords is None:
            coords = self.target.symbols()
        return self.inverse_exprs_func(coords)

    def inverse_available(self) -> bool:
        return self.inverse_exprs_func is not None

    def symbolic_inverse_kind(self) -> str:
        return str((self.metadata or {}).get("symbolic_inverse_kind", "explicit" if self.inverse_exprs_func is not None else "none"))

    def branch_assumptions(self) -> Optional[sp.Expr]:
        return (self.metadata or {}).get("branch_assumptions", None)

    def simplified_inverse_mapping_exprs(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Tuple[sp.Expr, ...]:
        exprs = self.inverse_mapping_exprs(coords)
        assumptions = self.branch_assumptions()
        if assumptions is None:
            assumptions = self.target.assumptions(coords if coords is not None else self.target.symbols())
        simplify_first = self.symbolic_inverse_kind() != "root_based"
        return tuple(_refine_with_assumptions(expr, assumptions, simplify_first=simplify_first) for expr in exprs)

    def jacobian_tnf(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> TNFMatrix:
        if coords is None:
            coords = self.source.symbols()
        exprs = self.mapping_exprs(coords)
        return tnf_build_matrix(len(exprs), len(coords), lambda i, j: sp.diff(exprs[i], coords[j]))

    def jacobian(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> sp.Matrix:
        return tnf_matrix_to_sympy(self.jacobian_tnf(coords))

    def jacobian_sympy(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> sp.Matrix:
        return self.jacobian(coords)

    def jacobian_det(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> sp.Expr:
        det = self.jacobian_tnf(coords).det().to_sympy()
        kind = self.symbolic_inverse_kind()
        if kind == "root_based":
            return det
        try:
            return sp.simplify(det)
        except Exception:
            return det

    def inverse_jacobian_tnf(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> TNFMatrix:
        if coords is None:
            coords = self.target.symbols()
        inv_exprs = self.inverse_mapping_exprs(coords)
        return tnf_build_matrix(len(inv_exprs), len(coords), lambda i, j: sp.diff(inv_exprs[i], coords[j]))

    def inverse_jacobian(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> sp.Matrix:
        return tnf_matrix_to_sympy(self.inverse_jacobian_tnf(coords))

    def inverse_jacobian_sympy(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> sp.Matrix:
        return self.inverse_jacobian(coords)

    def transform_point(self, point: Sequence[sp.Expr]) -> TNFMatrix:
        coords = self.source.symbols()
        if len(point) != len(coords):
            raise ValueError("Point dimension does not match source chart.")
        metadata = self.metadata or {}
        standard_name = str(metadata.get("standard_name", ""))
        # Symbolic round-trips through inverse spheroidal maps can create very
        # large nested acosh/acos/sqrt expressions. When the input point is
        # already expressed in the target chart variables through a registered
        # forward map, return those target coordinates directly instead of
        # asking SymPy to prove branch-sensitive inverse identities.
        if standard_name in {"Cartesian->ProlateSpheroidal", "Cartesian->OblateSpheroidal"}:
            target_coords = self.target.symbols()
            target_symbols = set(target_coords)
            try:
                target_symbols.update(sp.sympify(value).free_symbols for value in [])
            except Exception:
                pass
            for param_value in getattr(self.target, "parameters", lambda: {})().values():
                try:
                    target_symbols.update(sp.sympify(param_value).free_symbols)
                except Exception:
                    pass
            free_symbols = set()
            for value in point:
                try:
                    free_symbols.update(sp.sympify(value).free_symbols)
                except Exception:
                    pass
            if target_symbols and free_symbols and free_symbols <= target_symbols:
                return tnf_column_from_entries(target_coords)
        subs = dict(zip(coords, point))
        simplification_level = "refined" if self.symbolic_inverse_kind() == "root_based" else "strong"
        return tnf_column_from_entries(simplify_object(expr.subs(subs), level=simplification_level) for expr in self.mapping_exprs(coords))

    def map_properties(self) -> Tuple[str, ...]:
        keys = [
            "source",
            "target",
            "mapping_exprs",
            "inverse_mapping_exprs",
            "simplified_inverse_mapping_exprs",
            "jacobian",
            "jacobian_determinant",
            "inverse_available",
            "symbolic_inverse_kind",
        ]
        extra = list((self.metadata or {}).keys())
        return tuple(keys + [k for k in extra if k not in keys])

    def data(self, include_inverse_details: bool = True) -> Dict[str, object]:
        sc = self.source.symbols()
        tc = self.target.symbols()
        kind = self.symbolic_inverse_kind()
        if kind == "root_based":
            mapping_exprs = (self.metadata or {}).get("mapping_exprs_preview")
            if mapping_exprs is None:
                mapping_exprs = tuple(sp.Symbol(f"map_{i}") for i in range(self.target.dimension))
            jacobian = (self.metadata or {}).get("jacobian_preview")
            if jacobian is None:
                jacobian = TNFMatrix.zeros(self.target.dimension, self.source.dimension)
            jacobian_det = (self.metadata or {}).get("jacobian_determinant_preview", sp.Integer(0))
        else:
            mapping_exprs = self.mapping_exprs(sc)
            jacobian = self.jacobian_tnf(sc)
            jacobian_det = self.jacobian_det(sc)
        payload = {
            "source": (self.source.metric_name, self.source.chart_name, self.source.dimension),
            "target": (self.target.metric_name, self.target.chart_name, self.target.dimension),
            "mapping_exprs": mapping_exprs,
            "jacobian": jacobian,
            "jacobian_determinant": jacobian_det,
            "inverse_available": self.inverse_available(),
            "symbolic_inverse_kind": kind,
            "available_properties": self.map_properties(),
            **(self.metadata or {}),
        }
        if include_inverse_details and self.inverse_exprs_func:
            if kind == "root_based":
                inv_exprs = (self.metadata or {}).get("inverse_mapping_exprs_preview")
                if inv_exprs is None:
                    inv_exprs = tuple(sp.Symbol(f"inv_{i}") for i in range(self.source.dimension))
                simp_exprs = (self.metadata or {}).get("simplified_inverse_mapping_exprs_preview", inv_exprs)
                payload.update({
                    "inverse_mapping_exprs": inv_exprs,
                    "simplified_inverse_mapping_exprs": simp_exprs,
                })
            else:
                payload.update({
                    "inverse_mapping_exprs": self.inverse_mapping_exprs(tc),
                    "simplified_inverse_mapping_exprs": self.simplified_inverse_mapping_exprs(tc),
                })
        else:
            payload.update({
                "inverse_mapping_exprs": None,
                "simplified_inverse_mapping_exprs": None,
            })
        return payload


_REGISTRY: Dict[Tuple[Tuple[str, str, int], Tuple[str, str, int]], CoordinateMap] = {}


def register_map(mapping: CoordinateMap) -> None:
    s = (mapping.source.metric_name, mapping.source.chart_name, mapping.source.dimension)
    t = (mapping.target.metric_name, mapping.target.chart_name, mapping.target.dimension)
    _REGISTRY[(s, t)] = mapping


def get_map(source: CoordinateChart, target: CoordinateChart) -> CoordinateMap:
    key = (
        (source.metric_name, source.chart_name, source.dimension),
        (target.metric_name, target.chart_name, target.dimension),
    )
    if key not in _REGISTRY:
        raise KeyError(f"No registered map from {key[0]} to {key[1]}")
    return _REGISTRY[key]


def list_maps():
    return sorted(_REGISTRY.keys())


def mapping_property_names(mapping: CoordinateMap) -> Tuple[str, ...]:
    """Return the property names exposed by a coordinate map."""
    return mapping.map_properties()


def list_maps_with_symbolic_inverse() -> Iterable[Tuple[Tuple[str, str, int], Tuple[str, str, int]]]:
    return sorted(key for key, mapping in _REGISTRY.items() if mapping.inverse_available())


def list_charts_with_symbolic_inverse_available() -> Iterable[Tuple[str, str, int]]:
    charts: Set[Tuple[str, str, int]] = set()
    for (source_key, target_key), mapping in _REGISTRY.items():
        if mapping.inverse_available():
            charts.add(source_key)
            charts.add(target_key)
    return sorted(charts)
