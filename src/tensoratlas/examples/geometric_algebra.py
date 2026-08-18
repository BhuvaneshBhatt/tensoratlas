"""Geometric-algebra tutorial examples."""

from __future__ import annotations

from tensoratlas.geometric_algebra import GeometricAlgebra, reflect, rotate, rotor


def geometric_algebra_workflow() -> dict[str, object]:
    """Demonstrate products, blades, reflection, and rotation in Euclidean 2D."""
    algebra = GeometricAlgebra.euclidean(2)
    e1 = algebra.vector(0)
    e2 = algebra.vector(1)
    bivector = e1.exterior(e2)
    import sympy as sp
    alpha = sp.Symbol("alpha")
    rotor_value = rotor(alpha, bivector)
    vector = e1 + 2 * e2
    return {
        "e1_squared": e1 * e1,
        "anticommutator": e1 * e2 + e2 * e1,
        "bivector": bivector,
        "rotor": rotor_value,
        "rotated_vector": rotate(vector, rotor_value),
        "reflected_vector": reflect(vector, e1),
    }
