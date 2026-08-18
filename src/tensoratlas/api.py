from __future__ import annotations

import json
from importlib import resources
from typing import Sequence

import sympy as sp

from .charts import CoordinateChart, get_chart
from .fields import ScalarField, TensorField, VectorField
from .mappings import CoordinateMap, get_map


def load_coordinate_catalog() -> dict:
    """Load the packaged coordinate chart and transform catalog shipped with tensoratlas."""
    catalog_path = resources.files("tensoratlas").joinpath("coordinate_catalog.json")
    return json.loads(catalog_path.read_text())


def coordinate_chart(metric_name: str, chart_name: str, dimension: int = 3) -> CoordinateChart:
    """Return a registered coordinate chart by metric family, chart name, and dimension."""
    return get_chart(metric_name, chart_name, dimension)


def coordinate_map(source: CoordinateChart, target: CoordinateChart) -> CoordinateMap:
    """Return the registered mapping between two charts."""
    return get_map(source, target)


def transform_coordinates(
    source: CoordinateChart,
    target: CoordinateChart,
    point: Sequence[sp.Expr],
    mapping: CoordinateMap | None = None,
) -> sp.Matrix:
    """Map one point from the source chart into the target chart."""
    active_map = get_map(source, target) if mapping is None else mapping
    return active_map.transform_point(point)


def transform_field(
    field_obj: ScalarField | VectorField | TensorField,
    source: CoordinateChart,
    target: CoordinateChart,
    mapping: CoordinateMap | None = None,
    source_convention: str = "coordinate_basis",
    target_convention: str = "coordinate_basis",
):
    """Transform a scalar, vector, or tensor field between charts."""
    active_map = get_map(source, target) if mapping is None else mapping
    if not isinstance(field_obj, (ScalarField, VectorField, TensorField)):
        raise TypeError("field_obj must be a ScalarField, VectorField, or TensorField")
    if field_obj.chart != source:
        raise ValueError("source chart does not match field_obj.chart")
    if isinstance(field_obj, VectorField):
        return field_obj.transform(active_map, source_convention=source_convention, target_convention=target_convention)
    return field_obj.transform(active_map)


from .builtin_maps import register_builtin_maps

register_builtin_maps()
