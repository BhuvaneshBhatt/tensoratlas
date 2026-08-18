from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PublicReturnPolicy:
    name: str
    returns: str
    paired_with: Optional[str] = None
    notes: str = ""


_PUBLIC_RETURN_POLICIES: dict[str, PublicReturnPolicy] = {
    "CoordinateChart.metric": PublicReturnPolicy("CoordinateChart.metric", "sympy", "CoordinateChart.metric_tnf", "Public chart metric accessor returns a SymPy matrix."),
    "CoordinateChart.metric_tnf": PublicReturnPolicy("CoordinateChart.metric_tnf", "tnf", "CoordinateChart.metric", "Internal/native chart metric accessor returns TNFMatrix."),
    "CoordinateChart.inverse_metric": PublicReturnPolicy("CoordinateChart.inverse_metric", "sympy", "CoordinateChart.inverse_metric_tnf"),
    "CoordinateChart.inverse_metric_tnf": PublicReturnPolicy("CoordinateChart.inverse_metric_tnf", "tnf", "CoordinateChart.inverse_metric"),
    "CoordinateMap.jacobian": PublicReturnPolicy("CoordinateMap.jacobian", "sympy", "CoordinateMap.jacobian_tnf"),
    "CoordinateMap.jacobian_tnf": PublicReturnPolicy("CoordinateMap.jacobian_tnf", "tnf", "CoordinateMap.jacobian"),
    "CoordinateMap.inverse_jacobian": PublicReturnPolicy("CoordinateMap.inverse_jacobian", "sympy", "CoordinateMap.inverse_jacobian_tnf"),
    "CoordinateMap.inverse_jacobian_tnf": PublicReturnPolicy("CoordinateMap.inverse_jacobian_tnf", "tnf", "CoordinateMap.inverse_jacobian"),
    "basis_transformation_matrix": PublicReturnPolicy("basis_transformation_matrix", "sympy", "basis_transformation_matrix_tnf"),
    "basis_transformation_matrix_tnf": PublicReturnPolicy("basis_transformation_matrix_tnf", "tnf", "basis_transformation_matrix"),
    "frame_to_chart_matrix": PublicReturnPolicy("frame_to_chart_matrix", "sympy", "frame_to_chart_matrix_tnf"),
    "frame_to_chart_matrix_tnf": PublicReturnPolicy("frame_to_chart_matrix_tnf", "tnf", "frame_to_chart_matrix"),
    "chart_to_frame_matrix": PublicReturnPolicy("chart_to_frame_matrix", "sympy", "chart_to_frame_matrix_tnf"),
    "chart_to_frame_matrix_tnf": PublicReturnPolicy("chart_to_frame_matrix_tnf", "tnf", "chart_to_frame_matrix"),
    "frame_metric": PublicReturnPolicy("frame_metric", "sympy", "frame_metric_tnf"),
    "frame_metric_tnf": PublicReturnPolicy("frame_metric_tnf", "tnf", "frame_metric"),
    "gram_schmidt_frame": PublicReturnPolicy("gram_schmidt_frame", "sympy", "gram_schmidt_frame_tnf"),
    "gram_schmidt_frame_tnf": PublicReturnPolicy("gram_schmidt_frame_tnf", "tnf", "gram_schmidt_frame"),
    "transform_coordinates": PublicReturnPolicy("transform_coordinates", "sympy", None, "Point-transform convenience API returns a SymPy column matrix."),
}


def public_return_policy(name: str) -> PublicReturnPolicy:
    if name not in _PUBLIC_RETURN_POLICIES:
        raise KeyError(f"No public return-type policy registered for {name!r}.")
    return _PUBLIC_RETURN_POLICIES[name]


def list_public_return_policies() -> tuple[PublicReturnPolicy, ...]:
    return tuple(_PUBLIC_RETURN_POLICIES[name] for name in sorted(_PUBLIC_RETURN_POLICIES))


def paired_public_api(name: str) -> Optional[str]:
    return public_return_policy(name).paired_with


def returns_tnormal_forms(name: str) -> bool:
    return public_return_policy(name).returns == "tnf"


def returns_sympy(name: str) -> bool:
    return public_return_policy(name).returns == "sympy"
