from __future__ import annotations

import time
import sympy as sp

from tensoratlas import ScalarField, VectorField, coordinate_chart


def main() -> None:
    sph = coordinate_chart("Euclidean", "Spherical", 3)
    r, theta, phi = sph.symbols()
    scalar = ScalarField(sph, r**2 * sp.cos(theta) + sp.sin(phi))
    vector = VectorField(sph, sp.Matrix([[r], [theta], [phi]]))

    start = time.perf_counter()
    for _ in range(10):
        scalar.gradient()
        scalar.laplacian()
        vector.divergence()
    elapsed = time.perf_counter() - start
    print({"geometry_ops_s": round(elapsed, 4)})


if __name__ == "__main__":
    main()
