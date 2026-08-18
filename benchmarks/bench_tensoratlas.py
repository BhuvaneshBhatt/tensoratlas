from __future__ import annotations

import time

import sympy as sp

from tensoratlas import ScalarField, VectorField, coordinate_chart, coordinate_map, transform_coordinates


def main() -> None:
    cart = coordinate_chart("Euclidean", "Cartesian", 3)
    spherical = coordinate_chart("Euclidean", "Spherical", 3)
    mapping = coordinate_map(cart, spherical)
    x, y, z = cart.symbols()
    scalar = ScalarField(cart, (x + y + z) ** 3)
    vector = VectorField(cart, sp.Matrix([[y], [-x], [z]]))

    start = time.perf_counter()
    for _ in range(25):
        transform_coordinates(cart, spherical, sp.Matrix([2, 3, 4]), mapping)
    elapsed_roundtrip = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(10):
        scalar.gradient()
        scalar.laplacian()
        vector.curl()
        vector.divergence()
    elapsed_calculus = time.perf_counter() - start

    print({
        "coordinate_roundtrips_s": round(elapsed_roundtrip, 4),
        "calculus_ops_s": round(elapsed_calculus, 4),
    })


if __name__ == "__main__":
    main()
