"""Common multivector transformations."""

from __future__ import annotations

from typing import Any

import sympy as sp

from tensoratlas.errors import UnsupportedGeometryError

from .multivector import Multivector


def rotor(angle: Any, plane: Multivector) -> Multivector:
    """Return the elementary rotor ``cos(angle/2) - B sin(angle/2)``."""
    algebra = plane.algebra
    return algebra.scalar(sp.cos(angle / 2)) - plane * sp.sin(angle / 2)


def rotate(vector: Multivector, rotor_value: Multivector) -> Multivector:
    """Rotate a vector by ``R v R^{-1}``.

    For unit rotors this is equivalent to ``R v reverse(R)``.  Using the
    inverse makes the function safer for symbolic or not-yet-normalized rotors.
    """
    return rotor_value * vector * rotor_value.inverse()


def reflect(vector: Multivector, normal: Multivector) -> Multivector:
    """Reflect ``vector`` in the hyperplane normal to ``normal``."""
    return -(normal * vector * normal.inverse())


def project_vector_onto_vector(vector: Multivector, direction: Multivector) -> Multivector:
    """Project one grade-1 vector onto another non-null grade-1 vector."""
    if vector.grades() - {1} or direction.grades() - {1}:
        raise UnsupportedGeometryError("project_vector_onto_vector expects grade-1 vector arguments")
    denominator = sp.simplify(direction.norm_squared(require_scalar=True))
    if denominator == 0:
        raise ZeroDivisionError("cannot project onto a zero or null direction")
    return direction * (direction.inner(vector).scalar_part() / denominator)


def project(vector: Multivector, direction: Multivector) -> Multivector:
    """Compatibility alias for vector-on-vector projection."""
    return project_vector_onto_vector(vector, direction)
