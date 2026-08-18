import sympy as sp

from tensoratlas.core import (
    catalog_transition_map,
    coordinate_gradient,
    tensor_contract,
    tensor_product,
    transform_scalar_field,
)
from tensoratlas.differential_forms_frame import basis_one_form, wedge_forms
from tensoratlas.geometric_algebra import GeometricAlgebra
from tensoratlas.relativity import scalar_curvature, two_sphere_metric


# 1. Coordinate transformation
x, y = sp.symbols("x y", real=True)

cart_to_polar = catalog_transition_map("cartesian2", "polar")
transformed = transform_scalar_field(x**2 + y**2, cart_to_polar)

print("coordinate transform:")
print(transformed)


# 2. Vector calculus in curvilinear coordinates
r, theta = sp.symbols("r theta", positive=True)
polar_metric = ((1, 0), (0, r**2))

gradient = coordinate_gradient(
    r**2,
    (r, theta),
    metric=polar_metric,
)

print("\ngradient:")
print(gradient)


# 3. Tensor product + contraction
A = ((1, 2), (3, 4))
B = ((0, 5), (6, 7))

product = tensor_product(A, B)
contracted = tensor_contract(product, (1, 2))

print("\ntensor contraction:")
print(contracted.components)


# 4. Differential forms
dx = basis_one_form("dx")
dy = basis_one_form("dy")
area_form = wedge_forms(dx, dy)

print("\ndifferential form:")
print(area_form)


# 5. Curvature
sphere = two_sphere_metric()
R = scalar_curvature(sphere)

print("\ntwo-sphere scalar curvature:")
print(R)


# 6. Geometric algebra
ga = GeometricAlgebra.euclidean(2)
e1, e2 = ga.basis_vectors()

print("\ngeometric algebra:")
print("e1^2 =", e1 * e1)
print("e1 wedge e2 =", e1.wedge(e2))
