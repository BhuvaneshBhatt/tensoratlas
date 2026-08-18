# Getting started

TensorAtlas is authored by Bhuvanesh Bhatt and distributed under the GNU General Public License v3.0 only (GPL-3.0-only).

Install the project in editable mode from the repository root:

```bash
python -m pip install -e ".[dev,plot,docs]"
python -m pytest
python tools/release_audit.py
```

## List catalogued coordinate systems

```python
from tensoratlas.core import list_standard_coordinates

print(list_standard_coordinates())
```

## Inspect coordinate metadata

```python
from tensoratlas.core import standard_coordinate_system_data

data = standard_coordinate_system_data("spherical")
print(data["coordinates"])
print(data["metric"])
print(data["singularities"])
```

## Transform a scalar field

```python
import sympy as sp
from tensoratlas.core import catalog_transition_map, transform_scalar_field

x, y = sp.symbols("x y", real=True)
cart_to_polar = catalog_transition_map("cartesian2", "polar")
print(cart_to_polar.summary())
print(transform_scalar_field(x**2 + y**2, cart_to_polar))
```

## Compute a scalar Laplacian

```python
import sympy as sp
from tensoratlas.core import coordinate_laplacian_result

r, theta = sp.symbols("r theta", positive=True)
metric = ((1, 0), (0, r**2))
result = coordinate_laplacian_result(r**2, (r, theta), metric=metric)
print(result.components)
print(result.convention_metadata)
```

## Compute selected curvature components

```python
from tensoratlas.relativity import christoffel_component, scalar_curvature, two_sphere_metric

sphere = two_sphere_metric()
print(christoffel_component(sphere, 0, 1, 0))
print(scalar_curvature(sphere))
```

## Canonicalize a symmetric tensor term

```python
from fractions import Fraction
from tensoratlas.core import Manifold, TensorFactor, TensorHead, TensorTerm, canonicalize_tensor

M = Manifold("M", 4)
T = M.index_type("T")
a, b = T.indices("a b", variance="up")
S = TensorHead("S", (T, T), symmetry="symmetric", variance=(None, None))
term = TensorTerm(Fraction(1), (TensorFactor(S, (b, a)),))

print(canonicalize_tensor(term, explain=True).expression)
```

## Use summaries and validation

```python
from tensoratlas.core import catalog_transition_map

cart_to_polar = catalog_transition_map("cartesian2", "polar")
print(cart_to_polar.summary())
print(cart_to_polar.validate())
```

## Run the five-minute tour

```bash
python examples/five_minute_tour.py
```

The tour is the shortest end-to-end example for new users. It covers coordinate transformations, vector calculus, tensor-array operations, differential forms, curvature, and geometric algebra.

## Choosing a starting layer

Start with `tensoratlas.core` for explicit component and coordinate calculations. Use `tensoratlas.relativity` for metric-derived curvature and geodesic workflows. Use differential forms and tensor-valued forms for exterior-calculus or Cartan-style calculations. Use the abstract/indexed tensor APIs when expression equivalence, dummy-index handling, and symbolic identities are the main goal.
