"""Electromagnetism examples using formal tensor-valued forms."""

from __future__ import annotations

from tensoratlas.tensor_valued_forms import TensorValuedForm, exterior_derivative_tvform, gauge_curvature


def electromagnetic_workflow() -> dict[str, object]:
    """Create a formal potential, field strength, and Bianchi expression."""
    potential = TensorValuedForm(1, (), {(): "A"})
    field_strength = exterior_derivative_tvform(potential)
    bianchi = exterior_derivative_tvform(field_strength)
    gauge = TensorValuedForm(1, (+1, -1), {(0, 0): "omega"})
    curvature = gauge_curvature(gauge)
    return {
        "potential": potential,
        "field_strength": field_strength,
        "bianchi": bianchi,
        "gauge_curvature": curvature,
    }
