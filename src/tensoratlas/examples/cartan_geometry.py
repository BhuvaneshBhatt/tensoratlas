"""Cartan calculus examples with tensor-valued forms."""

from __future__ import annotations

from tensoratlas.tensor_valued_forms import cartan_first_equation, cartan_second_equation, connection_form, solder_form


def cartan_structure_workflow() -> dict[str, object]:
    """Return torsion and curvature expressions for a symbolic two-frame."""
    theta = solder_form({0: "theta0", 1: "theta1"})
    omega = connection_form({(0, 1): "omega01", (1, 0): "omega10"})
    torsion = cartan_first_equation(theta, omega)
    curvature = cartan_second_equation(omega)
    return {
        "solder": theta,
        "connection": omega,
        "solder_form": theta,
        "connection_form": omega,
        "torsion": torsion,
        "curvature": curvature,
    }
