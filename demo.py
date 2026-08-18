"""Small command-line demonstration for TensorAtlas."""

import sympy as sp

from tensoratlas.core import (
    catalog_transition_map,
    coordinate_gradient,
    coordinate_map_data,
    standard_coordinate_system_data,
    transform_scalar_field,
)


def main() -> None:
    r, theta = sp.symbols("r theta", positive=True)
    polar = standard_coordinate_system_data("polar")
    print("Polar metric:")
    print(polar["metric"])

    cmap = catalog_transition_map("polar", "cartesian2")
    props = coordinate_map_data(cmap).as_dict()
    print("\nPolar to Cartesian Jacobian determinant:")
    print(props["jacobian_determinant"])

    scalar = r**2
    print("\nGradient of r^2 in polar coordinates:")
    print(coordinate_gradient(scalar, (r, theta), polar["metric"]).components)

    print("\nScalar field transported to Cartesian coordinates:")
    print(transform_scalar_field(scalar, cmap))


if __name__ == "__main__":
    main()
