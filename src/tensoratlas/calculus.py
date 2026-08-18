from __future__ import annotations

from typing import Any

import sympy as sp

from .fields import ScalarField, TensorField, VectorField


def gradient(field: ScalarField) -> VectorField:
    """Return the contravariant gradient of a scalar field."""
    if not isinstance(field, ScalarField):
        raise TypeError("gradient expects a ScalarField.")
    return field.gradient()


def divergence(field: VectorField | TensorField) -> sp.Expr | TensorField:
    """Return the divergence of a vector field or general tensor field."""
    if isinstance(field, VectorField):
        return field.divergence()
    if isinstance(field, TensorField):
        return field.divergence()
    raise TypeError("divergence expects a VectorField or TensorField.")


def curl(field: VectorField | TensorField) -> VectorField:
    """Return the curl of a vector field or rank-1 tensor field."""
    if isinstance(field, VectorField):
        return field.curl()
    if isinstance(field, TensorField):
        return field.curl()
    raise TypeError("curl expects a VectorField or rank-1 TensorField.")


def laplacian(field: ScalarField | VectorField | TensorField) -> sp.Expr | VectorField | TensorField:
    """Return the scalar or connection Laplacian, depending on input type."""
    if isinstance(field, ScalarField):
        return field.laplacian()
    if isinstance(field, (VectorField, TensorField)):
        return field.connection_laplacian()
    raise TypeError("laplacian expects a ScalarField, VectorField, or TensorField.")


def connection_laplacian(field: VectorField | TensorField) -> VectorField | TensorField:
    """Return the rough / connection Laplacian of a vector or tensor field."""
    if not isinstance(field, (VectorField, TensorField)):
        raise TypeError("connection_laplacian expects a VectorField or TensorField.")
    return field.connection_laplacian()


def covariant_derivative(field: ScalarField | VectorField | TensorField) -> VectorField | TensorField:
    """Public dispatch wrapper for covariant differentiation."""
    if not isinstance(field, (ScalarField, VectorField, TensorField)):
        raise TypeError("covariant_derivative expects a ScalarField, VectorField, or TensorField.")
    return field.covariant_derivative()


def hessian(field: ScalarField) -> TensorField:
    """Return the covariant Hessian of a scalar field."""
    if not isinstance(field, ScalarField):
        raise TypeError("hessian expects a ScalarField.")
    return field.hessian()


def lie_derivative(field: VectorField | TensorField, vector: VectorField) -> VectorField | TensorField:
    """Return the Lie derivative of a vector or tensor field along a vector field."""
    if not isinstance(field, (VectorField, TensorField)):
        raise TypeError("lie_derivative expects a VectorField or TensorField as its first argument.")
    if not isinstance(vector, VectorField):
        raise TypeError("lie_derivative expects a VectorField as its second argument.")
    return field.lie_derivative(vector)


def exterior_derivative(field: ScalarField | TensorField) -> TensorField:
    """Return the exterior derivative of a scalar field or form-like tensor field."""
    if not isinstance(field, (ScalarField, TensorField)):
        raise TypeError("exterior_derivative expects a ScalarField or TensorField.")
    return field.exterior_derivative()
