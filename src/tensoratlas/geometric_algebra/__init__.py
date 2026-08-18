"""Orthogonal-metric multivector geometric algebra."""

from .algebra import Blade, GeometricAlgebra, canonical_exterior_blade
from .multivector import Multivector
from .operations import project, project_vector_onto_vector, reflect, rotate, rotor

__all__ = [
    "Blade",
    "GeometricAlgebra",
    "Multivector",
    "canonical_exterior_blade",
    "rotor",
    "rotate",
    "reflect",
    "project",
    "project_vector_onto_vector",
]
