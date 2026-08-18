# Differential forms, frames, and Cartan-style tools

## Differential forms

`tensoratlas` includes a differential-form layer that complements the tensor-field and indexed-tensor systems.

Typical operations include:

- antisymmetrization
- wedge product
- interior product with vector fields
- exterior derivative
- Hodge star
- codifferential

These operations are especially useful in orthogonal-coordinate and moving-frame calculations.

## Frames and coframes

The basis module exports helpers for working with frames and coframes:

- `frame_basis(...)`
- `coframe_basis(...)`
- `frame_to_chart_matrix(...)`
- `chart_to_frame_matrix(...)`
- `frame_metric(...)`
- `frame_structure_coefficients(...)`
- `frame_connection_coefficients(...)`
- `connection_one_forms(...)`

## Cartan-style structure checks

The package includes helpers related to Cartan's structure equations, including residual checks based on computed forms and frame data. This is useful for validating frame constructions or symbolic geometry formulas.

## Curvature and torsion forms

Additional helpers include:

- `curvature_two_forms(...)`
- `torsion_two_forms(...)`
- `coframe_connection_one_forms(...)`
- `exterior_derivative_coframe_1form(...)`
- `first_structure_equation_residuals(...)`
- `second_structure_equation_residuals(...)`

This part of the package is valuable when one wants a moving-frame viewpoint rather than staying entirely in coordinate components.

## Related notebook sections

- [Section 14: Differential forms](../notebooks/tensoratlas_demo.ipynb#14-differential-forms)
- [Section 28: Hodge star, codifferential, and the de Rham Laplacian](../notebooks/tensoratlas_demo.ipynb#28-hodge-star-codifferential-and-the-de-rham-laplacian)
- [Section 30: Exterior-calculus identities and Cartan's formula](../notebooks/tensoratlas_demo.ipynb#30-exterior-calculus-identities-and-cartans-formula)
- [Section 45: Basic examples: a tiny wedge-product and Hodge-star computation](../notebooks/tensoratlas_demo.ipynb#45-basic-examples-a-tiny-wedge-product-and-hodge-star-computation)
