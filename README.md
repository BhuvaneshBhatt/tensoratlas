# TensorAtlas

TensorAtlas is a SymPy-backed Python package for symbolic tensor algebra, coordinate geometry, differential forms, curvature calculations, tensor-valued forms, and orthogonal-metric geometric algebra. It is designed for inspectable mathematical workflows: conventions are explicit, objects expose summaries and validation helpers, and expensive symbolic simplification can be controlled.

**Author:** Bhuvanesh Bhatt (bhuvaneshbhatt@gmail.com)
**License:** GNU General Public License v3.0 only (GPL-3.0-only)

## Highlights

- Coordinate charts, coordinate maps, scalar/vector/covector/tensor fields, and coordinate-field transformations.
- Vector-calculus helpers for gradients, divergence, curl, Hessians, and Laplacians in coordinate bases.
- Symbolic tensor arrays with tensor products, contractions, transposes, summaries, validation, and SymPy array bridges.
- Differential forms, frames, coframes, Hodge-oriented workflows, and tensor-valued forms for Cartan-style geometry.
- Relativity utilities for metric catalogs, Christoffel symbols, Riemann/Ricci/scalar/Einstein curvature, geodesic equations, selected components, and nonzero-component inspection.
- Abstract and indexed tensor canonicalization with dummy-index normalization and identity-oriented reduction utilities.
- Orthogonal-metric multivector geometric algebra with geometric, exterior, inner, and contraction products, involutions, duals, reflections, and rotor helpers.
- A large tutorial notebook plus tested example modules and runnable scripts.

## Installation for development

From the repository root:

```bash
python -m pip install -e ".[dev,plot,docs]"
python -m pytest
python tools/release_audit.py
python examples/five_minute_tour.py
```

The base runtime dependency is SymPy. Plotting examples use the optional `plot` dependencies.

## Quick examples

### Curvature of the two-sphere

```python
from tensoratlas.relativity import scalar_curvature, two_sphere_metric

sphere = two_sphere_metric()
print(scalar_curvature(sphere))
```

Expected output:

```text
2/R**2
```

### Coordinate-field transformation

```python
import sympy as sp
from tensoratlas.core import catalog_transition_map, transform_scalar_field

x, y = sp.symbols("x y", real=True)
cart_to_polar = catalog_transition_map("cartesian2", "polar")
print(cart_to_polar.summary())
print(transform_scalar_field(x**2 + y**2, cart_to_polar))
```

### Tensor products and contractions

```python
from tensoratlas.core import tensor_contract, tensor_product

A = ((1, 2), (3, 4))
B = ((0, 5), (6, 7))
product = tensor_product(A, B)
contracted = tensor_contract(product, (1, 2))
print(contracted.components)
```

### Orthogonal geometric algebra

```python
from tensoratlas.geometric_algebra import GeometricAlgebra

alg = GeometricAlgebra(3)
e1 = alg.vector("e1")
e2 = alg.vector("e2")
print(e1 * e1)
print(e1.wedge(e2))
```

## Tutorial and examples

The main tutorial is:

```text
notebooks/tensoratlas_demo.ipynb
```

It includes tensor theory background, coordinate workflows, vector calculus, differential forms, electromagnetic forms, tensor-valued forms and Cartan equations, curvature on the two-sphere, Schwarzschild and FLRW examples, abstract tensor canonicalization, geometric algebra, debugging, validation, and performance notes.

Plain Python examples are in `examples/`, and tested reusable example workflows are in `src/tensoratlas/examples/`.

## Mathematical conventions

Relativity helpers use mostly-plus Lorentzian signature for built-in spacetime metrics. The Riemann convention is

```text
R^a{}_{bcd} = ∂_c Γ^a{}_{bd} - ∂_d Γ^a{}_{bc}
              + Γ^a{}_{ce} Γ^e{}_{bd} - Γ^a{}_{de} Γ^e{}_{bc}
```

with Ricci contraction `R_bd = R^a{}_{bad}` and scalar curvature `R = g^{ab} R_ab`. Tensor-valued Cartan helpers use

```text
T^a = dθ^a + ω^a{}_b ∧ θ^b
Ω^a{}_b = dω^a{}_b + ω^a{}_c ∧ ω^c{}_b
```

See `docs/conventions.md` for the full convention reference.

## Scope and limitations

TensorAtlas focuses on symbolic, inspectable workflows. It is not a numerical relativity framework, a plotting library, or a complete replacement for specialized tensor-canonicalization and geometric-algebra systems. The geometric algebra layer currently supports diagonal/orthogonal metrics; non-diagonal Clifford metrics are rejected deliberately rather than simplified incorrectly.

## Release checks

Before publishing a distribution, run:

```bash
python tools/release_audit.py
python -m pytest
python -m build
python -m twine check dist/*
```


## Usability and performance notes

Geometric algebra examples usually unpack basis vectors directly:

```python
from tensoratlas.geometric_algebra import GeometricAlgebra

ga = GeometricAlgebra.euclidean(3)
e1, e2, e3 = ga.basis_vectors()
rotor_input = e1.wedge(e2)
```

Dense helpers such as `nonzero_riemann` compute a full tensor before filtering. For larger metrics, prefer selected components such as `christoffel_component`, `riemann_component`, `ricci_component`, and `einstein_component`.

Small benchmark scripts are available under `benchmarks/`:

```bash
python benchmarks/benchmark_import_time.py
python benchmarks/benchmark_relativity.py
python benchmarks/benchmark_geometric_algebra.py
python benchmarks/benchmark_tensor_canonicalization.py
```


### Optional visualization examples

Install the optional plotting dependencies and run:

```bash
python examples/visualization_workflow.py
```

The visualization examples illustrate basis changes, covectors, metrics, tensor products, contractions, forms, pullbacks, curvature, geodesics, continuum-mechanics tensors, quadrupole moments, geometric-algebra rotors, and canonicalization diagrams.

## Five-minute tour

A compact public example is available at:

```bash
python examples/five_minute_tour.py
```

It demonstrates coordinate maps, vector-calculus helpers, tensor products and contractions, differential forms, selected curvature calculations, and a small geometric-algebra rotor workflow.

## Stability policy

For the 0.1.x series, the most stable public interfaces are the documented top-level exports, `tensoratlas.core`, `tensoratlas.relativity`, `tensoratlas.geometric_algebra`, `tensoratlas.examples`, and the APIs described in `docs/api_reference.md`. Lower-level canonicalization, semantic-rewrite, and internal normal-form modules are available for experimentation, but may change before the 0.2.x series as the public design settles.

## Choosing the right layer

Use `tensoratlas.core` when you have explicit tensor components, coordinate charts, coordinate maps, scalar fields, vector fields, covector fields, or tensor fields.

Use `tensoratlas.relativity` when you want metric-derived quantities such as Christoffel symbols, Riemann tensors, Ricci tensors, scalar curvature, Einstein tensors, selected components, geodesics, or built-in spacetime metric examples.

Use differential-form and tensor-valued-form helpers when your calculation is naturally written with wedge products, exterior derivatives, frames, coframes, Cartan equations, curvature forms, or electromagnetic forms.

Use the abstract/indexed tensor APIs when you care about symbolic identities, dummy indices, canonicalization, slot symmetries, normal forms, or expression-equivalence workflows.

Use `tensoratlas.geometric_algebra` for orthogonal-metric Clifford/geometric algebra calculations. The current public layer deliberately supports diagonal/orthogonal metrics; non-diagonal Clifford metrics are rejected rather than simplified incorrectly.

## Example limitations

The public examples are designed to be small and inspectable. Dense curvature helpers such as full nonzero-component scans compute large intermediate arrays before filtering, so selected-component APIs are preferable for larger metrics. Visualization examples return Matplotlib figures and the runnable script closes them immediately so release checks do not accumulate open figures.
