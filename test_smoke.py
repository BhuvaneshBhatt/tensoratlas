from __future__ import annotations

import sympy as sp

from tensoratlas.core import (
    catalog_transition_map,
    tensor_contract,
    tensor_product,
    transform_scalar_field,
)
from tensoratlas.differential_forms_frame import (
    basis_one_form,
    wedge_forms,
)
from tensoratlas.geometric_algebra import GeometricAlgebra
from tensoratlas.relativity import (
    christoffel_component,
    scalar_curvature,
    two_sphere_metric,
)


def main() -> None:
    # ------------------------------------------------------------
    # 1. Coordinate transformation
    #
    # In polar coordinates:
    #     r^2 = x^2 + y^2
    #
    # Use the coordinate symbols owned by the CoordinateMap rather
    # than independently constructing symbols with the same names.
    # ------------------------------------------------------------
    polar_to_cartesian = catalog_transition_map(
        "polar",
        "cartesian2",
    )

    r, theta = polar_to_cartesian.source_symbols
    x, y = polar_to_cartesian.target_symbols

    transformed_scalar = transform_scalar_field(
        r**2,
        polar_to_cartesian,
    )

    assert sp.simplify(
        transformed_scalar - (x**2 + y**2)
    ) == 0

    print("PASS: coordinate transformation")
    print("      r^2 ->", transformed_scalar)


    # ------------------------------------------------------------
    # 2. Tensor product and contraction
    #
    # These expected components come directly from TensorAtlas's
    # tensor-operation regression tests.
    # ------------------------------------------------------------
    first = (
        (1, 2),
        (3, 4),
    )
    second = (
        (0, 5),
        (6, 7),
    )

    product = tensor_product(first, second)
    contracted = tensor_contract(product, (1, 2))

    expected_contraction = (
        (12, 19),
        (24, 43),
    )

    assert contracted.dimensions == (2, 2)
    assert contracted.components == expected_contraction

    print("PASS: tensor product + contraction")
    print("      components =", contracted.components)


    # ------------------------------------------------------------
    # 3. Differential-form wedge product
    #
    # theta1 ^ theta0 = - theta0 ^ theta1
    #
    # TensorAtlas canonicalizes the basis ordering and records the
    # antisymmetric sign in metadata.
    # ------------------------------------------------------------
    theta1 = basis_one_form("theta1")
    theta0 = basis_one_form("theta0")

    two_form = wedge_forms(theta1, theta0)

    assert two_form.kind == "form:wedge"
    assert two_form.metadata["degree"] == 2
    assert two_form.metadata["basis_labels"] == (
        "theta0",
        "theta1",
    )
    assert two_form.metadata["coefficient"] == -1

    # Exterior product of a one-form with itself must vanish.
    repeated = wedge_forms(theta0, theta0)
    assert repeated.kind == "zero"

    print("PASS: differential-form wedge algebra")
    print(
        "      theta1 ^ theta0 ->",
        two_form.metadata["coefficient"],
        two_form.metadata["basis_labels"],
    )


    # ------------------------------------------------------------
    # 4. Curvature of a two-sphere
    #
    # For a 2-sphere of radius R:
    #
    #     scalar curvature = 2 / R^2
    #
    # and
    #
    #     Gamma^theta_{phi phi}
    #       = -sin(theta) cos(theta)
    # ------------------------------------------------------------
    sphere = two_sphere_metric()

    radius = sphere.parameters[0]
    sphere_theta, sphere_phi = sphere.coordinates

    curvature = scalar_curvature(sphere)

    assert sp.simplify(
        curvature - 2 / radius**2
    ) == 0

    gamma_theta_phiphi = christoffel_component(
        sphere,
        0,
        1,
        1,
    )

    expected_gamma = (
        -sp.sin(sphere_theta) * sp.cos(sphere_theta)
    )

    assert sp.simplify(
        gamma_theta_phiphi - expected_gamma
    ) == 0

    print("PASS: two-sphere curvature")
    print("      scalar curvature =", curvature)
    print(
        "      Gamma^theta_{phi phi} =",
        gamma_theta_phiphi,
    )


    # ------------------------------------------------------------
    # 5. Euclidean geometric algebra Cl(2, 0)
    #
    # For orthonormal Euclidean basis vectors:
    #
    #     e1^2 = 1
    #     e2^2 = 1
    #     e1 e2 + e2 e1 = 0
    #     e1 ^ e2 has unit bivector coefficient
    # ------------------------------------------------------------
    algebra = GeometricAlgebra.euclidean(2)
    e1, e2 = algebra.basis_vectors()

    e1_squared = e1 * e1
    e2_squared = e2 * e2
    anticommutator = e1 * e2 + e2 * e1
    bivector = e1.wedge(e2)

    assert e1_squared.coeffs == {(): sp.Integer(1)}
    assert e2_squared.coeffs == {(): sp.Integer(1)}
    assert anticommutator.is_zero()
    assert bivector.coeffs == {
        (0, 1): sp.Integer(1),
    }

    print("PASS: geometric algebra")
    print("      e1^2 =", e1_squared.coeffs)
    print("      e2^2 =", e2_squared.coeffs)
    print(
        "      e1*e2 + e2*e1 is zero:",
        anticommutator.is_zero(),
    )
    print("      e1 ^ e2 =", bivector.coeffs)


    # ------------------------------------------------------------
    # All selected subsystems behaved as expected.
    # ------------------------------------------------------------
    print()
    print("TensorAtlas smoke test PASSED.")


if __name__ == "__main__":
    main()
