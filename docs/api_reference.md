# API reference overview

This is a compact map of the preferred public APIs. Import from the domain modules shown here rather than relying on implementation modules.

## Coordinates and fields (`tensoratlas.core`)

- `list_standard_coordinates()`
- `standard_coordinate_system_data(name)`
- `standard_coordinate_entry(name)`
- `standard_metric(name)`
- `catalog_transition_map(source, target)`
- `coordinate_map_data(cmap)`
- `transform_scalar_field(expr, cmap)`
- `transform_vector_field(components, cmap)`
- `transform_covector_field(components, cmap)`
- `transform_tensor_field(components, cmap, variance)`
- `transform_tensor_density(components, cmap, variance, density_weight)`

Coordinate charts and maps expose `summary()`, `validate()`, and `domain_conditions()` helpers.

## Vector calculus (`tensoratlas.core`)

- `coordinate_gradient(expr, coordinates, metric=...)`
- `coordinate_divergence(vector, coordinates, metric=...)`
- `coordinate_curl(vector, coordinates, metric=...)`
- `coordinate_hessian(expr, coordinates, metric=..., convention=...)`
- `coordinate_laplacian(expr, coordinates, metric=..., convention=...)`
- `coordinate_curl_result(...)`
- `coordinate_hessian_result(...)`
- `coordinate_laplacian_result(...)`
- `vector_laplacian(...)`
- `tensor_laplacian(...)`

## Symbolic tensor arrays (`tensoratlas.core`)

- `TensorArray`
- `tensor_product(...)`
- `tensor_contract(...)`
- `tensor_transpose(...)`
- `tensor_dimensions(...)`
- `tensor_properties(...)`

`TensorArray` also provides `summary()`, `validate()`, `contract(...)`, `tensor_product(...)`, `to_sympy_array()`, and `from_sympy_array(...)`.

## Relativity (`tensoratlas.relativity`)

- `MetricModel`
- `minkowski_metric()`
- `two_sphere_metric()`
- `schwarzschild_metric()`
- `flrw_metric()`
- `christoffel_symbols(model, simplify=...)`
- `christoffel_component(model, a, b, c, simplify=...)`
- `riemann_tensor(model, simplify=...)`
- `riemann_component(model, a, b, c, d, simplify=...)`
- `ricci_tensor(model, simplify=...)`
- `ricci_component(model, a, b, simplify=...)`
- `scalar_curvature(model, simplify=...)`
- `einstein_tensor(model, simplify=...)`
- `einstein_component(model, a, b, simplify=...)`
- `geodesic_equations(model, simplify=...)`
- `geodesic_equation(model, a, simplify=...)`
- `nonzero_christoffel(...)`, `nonzero_riemann(...)`, `nonzero_ricci(...)`, `nonzero_einstein(...)`

`MetricModel` exposes method wrappers such as `.christoffel()`, `.ricci()`, `.scalar_curvature()`, `.einstein()`, `.geodesic_equations()`, `.summary()`, and `.validate()`.

## Tensor-valued forms (`tensoratlas.tensor_valued_forms`)

- `TensorValuedForm`
- `solder_form(...)`
- `connection_form(...)`
- `torsion_form(...)`
- `curvature_form(...)`
- `cartan_first_equation(...)`
- `cartan_second_equation(...)`
- `gauge_curvature(...)`

`TensorValuedForm` exposes `.summary()`, `.validate()`, `.wedge(...)`, and `.exterior_derivative()`.

## Geometric algebra (`tensoratlas.geometric_algebra`)

- `GeometricAlgebra`
- `Multivector`

The public multivector layer supports orthogonal metrics and operations including geometric product, wedge product, inner product, left contraction, grade projection, reverse, grade involution, Clifford conjugation, duals, inverses, commutators, rotors, rotations, reflections, and projections.

## Display and usability (`tensoratlas.display`, `tensoratlas.errors`)

- `to_latex(obj)`
- `display_components(obj)`
- `display_nonzero_components(mapping)`
- `TensorAtlasError` and domain-specific subclasses for coordinates, contractions, metrics, forms, and unsupported geometry.


## Validation helpers

- `ValidationReport`: structured result returned by `validation_report()` methods.
- `check_indices(context, dimension, *indices)`: shared index-bound checker used by selected-component APIs.

Most public objects still provide `validate()` methods that raise on invalid state. Use `validation_report()` when you need diagnostics without exception handling.


## Dense inspection helpers

`nonzero_christoffel`, `nonzero_riemann`, `nonzero_ricci`, and `nonzero_einstein` compute dense tensors and then filter nonzero entries. They are intended for small metrics and tutorials. Prefer selected-component APIs for larger symbolic metrics.
