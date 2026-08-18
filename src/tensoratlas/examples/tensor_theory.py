"""Tensor-theory tutorial examples.

These examples provide the linear-algebra bridge used in the main notebook:
vectors and covectors, dual bases, multilinear maps, metrics, tensor
products, contractions, and basic transformation rules.
"""

from __future__ import annotations

import sympy as sp

from tensoratlas.core import tensor_contract, tensor_product


def notation_table() -> list[dict[str, str]]:
    """Return a compact notation guide for common tensor types."""
    return [
        {"type": "(1, 0)", "index": "v^i", "meaning": "vector / tangent vector"},
        {"type": "(0, 1)", "index": "alpha_i", "meaning": "covector / one-form"},
        {"type": "(1, 1)", "index": "A^i_j", "meaning": "linear map"},
        {"type": "(0, 2)", "index": "g_ij", "meaning": "bilinear form / metric"},
        {"type": "(2, 0)", "index": "P^ij", "meaning": "contravariant rank-two tensor"},
        {"type": "(2, 2)", "index": "R^ij_kl", "meaning": "mixed tensor such as a curvature-like object"},
    ]


def vector_covector_pairing_example() -> dict[str, object]:
    """Show that a covector is a linear functional on vectors."""
    vector = sp.Matrix([2, -1])
    covector = sp.Matrix([[3, 5]])
    return {"vector": vector, "covector": covector, "pairing": (covector * vector)[0]}


def covariant_contravariant_scaling_example() -> dict[str, object]:
    """Compare contravariant length components and covariant rate components."""
    length_km = sp.Integer(1)
    length_m = 1000 * length_km
    length_mm = 1000000 * length_km
    rate_per_km = sp.Integer(60)
    rate_per_m = sp.Rational(60, 1000)
    rate_per_mm = sp.Rational(60, 1000000)
    return {
        "length_components": (length_km, length_m, length_mm),
        "rate_components": (rate_per_km, rate_per_m, rate_per_mm),
    }


def dual_basis_example() -> dict[str, object]:
    """Compute the dual basis rows for a non-orthogonal basis."""
    basis_matrix = sp.Matrix([[1, 1], [0, 2]])
    dual_rows = basis_matrix.inv()
    kronecker = dual_rows * basis_matrix
    return {"basis_matrix": basis_matrix, "dual_basis_rows": dual_rows, "kronecker_delta": kronecker}


def basis_change_example() -> dict[str, object]:
    """Show vector, covector, and linear-map behavior under a basis change."""
    old_to_new_basis = sp.Matrix([[1, 1], [0, 2]])
    new_components_from_old = old_to_new_basis.inv()
    vector_old = sp.Matrix([5, 7])
    covector_old = sp.Matrix([[3, 4]])
    linear_map_old = sp.Matrix([[1, 2], [3, 4]])

    vector_new = new_components_from_old * vector_old
    covector_new = covector_old * old_to_new_basis
    linear_map_new = new_components_from_old * linear_map_old * old_to_new_basis

    vector_consistency = sp.simplify(new_components_from_old * (linear_map_old * vector_old) - linear_map_new * vector_new)
    pairing_old = (covector_old * vector_old)[0]
    pairing_new = (covector_new * vector_new)[0]
    return {
        "basis_matrix": old_to_new_basis,
        "inverse_basis_matrix": new_components_from_old,
        "vector_old": vector_old,
        "vector_new": vector_new,
        "covector_old": covector_old,
        "covector_new": covector_new,
        "linear_map_old": linear_map_old,
        "linear_map_new": linear_map_new,
        "pairing_difference": sp.simplify(pairing_old - pairing_new),
        "linear_map_consistency": vector_consistency,
    }


def multilinear_metric_example() -> dict[str, object]:
    """Verify that a metric is bilinear in its vector arguments."""
    x1, x2, y1, y2, u1, u2, a, b = sp.symbols("x1 x2 y1 y2 u1 u2 a b")
    x = sp.Matrix([x1, x2])
    y = sp.Matrix([y1, y2])
    u = sp.Matrix([u1, u2])
    metric = sp.Matrix([[1, 0], [0, 4]])
    value = (x.T * metric * y)[0]
    left = ((a * x + b * u).T * metric * y)[0]
    right = a * (x.T * metric * y)[0] + b * (u.T * metric * y)[0]
    return {"metric": metric, "value": value, "linearity_residual": sp.simplify(left - right)}


def metric_raising_lowering_example() -> dict[str, object]:
    """Lower and raise an index with a polar-coordinate metric."""
    radius, v_r, v_theta = sp.symbols("r v_r v_theta", positive=True)
    metric = sp.Matrix([[1, 0], [0, radius**2]])
    vector = sp.Matrix([v_r, v_theta])
    lowered = metric * vector
    raised = metric.inv() * lowered
    return {"metric": metric, "vector": vector, "lowered": lowered, "raise_lower_residual": sp.simplify(raised - vector)}


def metric_pullback_example() -> dict[str, object]:
    """Derive the polar metric as a pullback of the Cartesian Euclidean metric."""
    radius, angle = sp.symbols("rho theta", positive=True)
    x = radius * sp.cos(angle)
    y = radius * sp.sin(angle)
    jacobian = sp.Matrix([[sp.diff(x, radius), sp.diff(x, angle)], [sp.diff(y, radius), sp.diff(y, angle)]])
    metric_cartesian = sp.eye(2)
    metric_polar = jacobian.T * metric_cartesian * jacobian
    return {"jacobian": jacobian, "polar_metric": sp.simplify(metric_polar)}


def tensor_product_contraction_example() -> dict[str, object]:
    """Build elementary tensors and show contraction as index removal."""
    vector = (1, 2)
    covector = (3, 4)
    elementary = tensor_product(vector, covector)
    matrix = ((1, 2), (3, 4))
    trace = tensor_contract(matrix, (0, 1))
    product = tensor_product(matrix, matrix)
    matrix_product_like = tensor_contract(product, (1, 2))
    return {
        "elementary_tensor": elementary.components,
        "trace": trace.components,
        "contracted_product": matrix_product_like.components,
    }


def tensor_theory_workflow() -> dict[str, object]:
    """Return the tensor-theory examples used by the main tutorial."""
    return {
        "notation": notation_table(),
        "pairing": vector_covector_pairing_example(),
        "scaling": covariant_contravariant_scaling_example(),
        "dual_basis": dual_basis_example(),
        "basis_change": basis_change_example(),
        "metric_bilinearity": multilinear_metric_example(),
        "raising_lowering": metric_raising_lowering_example(),
        "metric_pullback": metric_pullback_example(),
        "tensor_products": tensor_product_contraction_example(),
    }
