# Fields and metric-aware geometry

## Field objects

The public field layer includes:

- `ScalarField`
- `VectorField`
- `TensorField`

Each field is attached to a chart and stores symbolic component data appropriate to its type.

## Basic scalar operators

For a scalar field `f`, typical operations include:

- gradient
- covariant derivative
- Hessian
- Laplacian
- exterior derivative in the form layer

Example:

```python
r, theta, phi = spherical.symbols()
f = ScalarField(spherical, r**2 * sp.cos(theta))

f.gradient()
f.laplacian()
f.hessian()
```

## Vector and tensor operators

For vector and tensor fields, the package provides symbolic versions of operations such as:

- divergence
- raising and lowering indices
- covariant derivative
- Lie derivative
- tensor contraction
- rough / connection Laplacian
- selected Ricci / Lichnerowicz-style Laplacian helpers

Example:

```python
V = VectorField(spherical, sp.Matrix([[1], [0], [0]]), "contravariant")
V.lower_index()
V.divergence()
V.covariant_derivative()
```

## Christoffel symbols and curvature

At the chart level, `tensoratlas` supports standard curvature data:

- Christoffel symbols of the first kind
- Christoffel symbols of the second kind
- Riemann tensor
- Ricci tensor
- scalar curvature
- Einstein tensor
- Schouten tensor
- Weyl tensor

This is especially convenient when checking identities or generating symbolic formulas in standard orthogonal coordinates.

## Geodesics

The chart layer includes geodesic-equation helpers. Conceptually, these are built from the Levi-Civita connection:

$$
\ddot{x}^i + \Gamma^i_{jk} \dot{x}^j \dot{x}^k = 0.
$$

The package provides symbolic geodesic equations and related helpers such as geodesic Lagrangians and cyclic-coordinate first integrals where implemented.

## Curl, frames, and orthonormal viewpoints

In 3D Riemannian coordinate charts, the package includes a coordinate-basis curl operator. It also supports orthonormal-frame conversions for orthogonal charts, which is often useful when one wants physically natural components rather than raw coordinate components.

## Scope

These operators are implemented in a practical way for the chart families supported by the repository. They are intended for symbolic work and exploration rather than high-performance numerical geometry.

## Related notebook sections

- [Section 8: Covectors and gradients](../notebooks/tensoratlas_demo.ipynb#8-covectors-and-gradients)
- [Section 9: Metrics, raising/lowering, and scale factors](../notebooks/tensoratlas_demo.ipynb#9-metrics-raisinglowering-and-scale-factors)
- [Section 13: Coordinate calculus in curvilinear coordinates](../notebooks/tensoratlas_demo.ipynb#13-coordinate-calculus-in-curvilinear-coordinates)
- [Section 43: Basic examples: gradient, divergence, and Laplacian in one simple chart](../notebooks/tensoratlas_demo.ipynb#43-basic-examples-gradient-divergence-and-laplacian-in-one-simple-chart)
- [Section 44: Basic examples: raising/lowering one index, building a rank-2 tensor, and contracting it](../notebooks/tensoratlas_demo.ipynb#44-basic-examples-raisinglowering-one-index-building-a-rank-2-tensor-and-contracting-it)
