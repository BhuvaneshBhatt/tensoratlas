# Semantic Exterior Calculus and Spin Geometry

This layer deepens TensorAtlas in two directions:

- full exterior-calculus utilities over `ExteriorFormNF`
- a more operational spin / Dirac subsystem over orthonormal frames

## Exterior calculus

New APIs:

- `hodge_star_nf(form, ...)`
- `codifferential_nf(form, coordinates, ...)`
- `interior_product_nf(vector_components, form)`
- `lie_derivative_nf(vector_components, form, coordinates)`
- `hodge_laplacian_nf(form, coordinates, ...)`

These operate on canonicalized exterior forms and keep computations structural rather than printer-driven.

## Spin geometry

New APIs:

- `spin_connection(frame, ...)`
- `gamma_frame_generators(frame, clifford=None)`
- `antisymmetrized_gamma_product(indices, clifford)`
- `gamma_string_simplify(expr, clifford)`
- `gamma_trace(expr, clifford)`
- `spin_covariant_derivative(spinor, spin_conn, clifford, coordinates)`
- `dirac_operator(spinor, spin_conn, clifford, coordinates)`

The current implementation is aimed at orthonormal-frame workflows and symbolic gamma-string reduction.
