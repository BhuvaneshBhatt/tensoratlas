# Charts, mappings, and coordinate-fields

The coordinate layer provides standard coordinate systems, metadata-rich coordinate maps, transition maps, and symbolic field transformations between charts.

## Coordinate catalog

Use `list_standard_coordinates()` and `standard_coordinate_system_data(name)` to inspect built-in systems.  Catalog entries include coordinate symbols, domain metadata, metric matrices, determinant data, known singularities, and transform-to-Cartesian data when available.

```python
from tensoratlas.core import list_standard_coordinates, standard_coordinate_system_data

print(list_standard_coordinates())
print(standard_coordinate_system_data("polar")["metric"])
```

## Transform data

`coordinate_map_data(cmap)` returns a property bundle containing:

- source and target coordinate names;
- forward map;
- inverse branches;
- Jacobian and inverse Jacobian;
- Jacobian determinant;
- local-invertibility condition;
- orientation expression;
- known and inferred singularities;
- domain notes.

```python
from tensoratlas.core import catalog_transition_map, coordinate_map_data

cmap = catalog_transition_map("cartesian2", "polar")
props = coordinate_map_data(cmap).as_dict()
print(props["jacobian"])
print(props["inverse_branches"])
```

## Field transformations

Use:

- `transform_scalar_field(expr, cmap)`;
- `transform_vector_field(components, cmap)`;
- `transform_covector_field(components, cmap)`;
- `transform_tensor_field(components, cmap, variance)`;
- `transform_tensor_density(components, cmap, variance, density_weight=...)`;
- `transform_field(...)` for dispatch by field type.

Tensor-density transformations carry convention metadata.  The convention currently used is that target components are multiplied by `|det(d source / d target)|**weight`.

## Coordinate versus physical components

Vector-calculus helpers distinguish coordinate components from physical/orthonormal components where scale factors are available:

```python
from tensoratlas.core import coordinate_to_physical_vector, physical_to_coordinate_vector, scale_factors
```

For diagonal metrics, scale factors are `sqrt(|g_ii|)`.  Physical components are usually more intuitive in orthogonal curvilinear coordinates, while coordinate components are the natural input for tensor transformation laws.
