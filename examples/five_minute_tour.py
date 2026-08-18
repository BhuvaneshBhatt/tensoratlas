"""Five-minute TensorAtlas tour.

Run from the repository root with:

    python examples/five_minute_tour.py
"""

from __future__ import annotations

import sympy as sp

from tensoratlas.core import (
    catalog_transition_map,
    coordinate_curl,
    coordinate_divergence,
    coordinate_gradient,
    tensor_contract,
    tensor_product,
    transform_scalar_field,
)
from tensoratlas.differential_forms_frame import basis_one_form, wedge_forms
from tensoratlas.geometric_algebra import GeometricAlgebra
from tensoratlas.relativity import christoffel_component, scalar_curvature, two_sphere_metric


def main() -> None:
    x, y = sp.symbols("x y", real=True)
    r, theta = sp.symbols("r theta", positive=True)

    print("# Coordinate fields")
    cart_to_polar = catalog_transition_map("cartesian2", "polar")
    print("map:", cart_to_polar.summary()["name"])
    print("scalar pullback:", transform_scalar_field(x**2 + y**2, cart_to_polar))
    polar_metric = ((1, 0), (0, r**2))
    print("gradient:", coordinate_gradient(r**2, (r, theta), metric=polar_metric))
    print("divergence:", coordinate_divergence((r, 0), (r, theta), metric=polar_metric))
    print("planar curl:", coordinate_curl((-y, x), (x, y)))

    print("\n# Tensor arrays")
    product = tensor_product(((1, 2), (3, 4)), ((0, 5), (6, 7)))
    print("contracted tensor:", tensor_contract(product, (1, 2)).components)

    print("\n# Differential forms")
    dx = basis_one_form("dx")
    dy = basis_one_form("dy")
    print("area form:", wedge_forms(dx, dy))

    print("\n# Relativity / curvature")
    sphere = two_sphere_metric()
    print("scalar curvature:", scalar_curvature(sphere))
    print("Gamma^theta_{phi phi}:", christoffel_component(sphere, 0, 1, 1))

    print("\n# Geometric algebra")
    ga = GeometricAlgebra.euclidean(2)
    e1, e2 = ga.basis_vectors()
    print("e1^2:", e1 * e1)
    print("e1 wedge e2:", e1.wedge(e2))


if __name__ == "__main__":
    main()
