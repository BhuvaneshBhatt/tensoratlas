# TensorAtlas conventions

TensorAtlas keeps mathematical conventions explicit because tensor calculations are otherwise easy to misread.  This page summarizes the conventions used by the public examples, relativity utilities, tensor-valued forms, and geometric-algebra layer.

## Coordinate and component conventions

Coordinate-basis components are the default for scalar, vector, covector, tensor, and density workflows.  A vector component tuple therefore means components in the ordered coordinate basis attached to the chart, not normalized physical components.

For orthogonal coordinate systems, helper functions may convert between coordinate-basis components and normalized frame components.  These conversions are deliberate and explicit because scale factors matter in curvilinear coordinates.

## Index and variance conventions

TensorAtlas uses explicit variance metadata where supported.  In tensor-valued forms, variance entries use `+1` for contravariant bundle slots and `-1` for covariant bundle slots.  Component keys must have the same length as the variance tuple.  For example, a vector-valued form has variance `(+1,)`, while an endomorphism-valued form has variance `(+1, -1)`.

## Curvature conventions

The relativity module returns Christoffel symbols as $\Gamma^a{}_{bc}$ with lower-index symmetry for Levi-Civita connections.  The Riemann tensor convention is

$$
R^a{}_{bcd}
= \partial_c \Gamma^a{}_{bd}
- \partial_d \Gamma^a{}_{bc}
+ \Gamma^a{}_{ce}\Gamma^e{}_{bd}
- \Gamma^a{}_{de}\Gamma^e{}_{bc}.
$$

The Ricci tensor is contracted as

$$
R_{bd} = R^a{}_{bad},
$$

and scalar curvature is

$$
R = g^{ab}R_{ab}.
$$

The Einstein tensor is

$$
G_{ab} = R_{ab} - \frac{1}{2}g_{ab}R.
$$

The standard Lorentzian examples use mostly-plus signature $(-,+,+,+)$.

## Differential forms and Hodge operations

Wedge products are graded-antisymmetric.  Repeated basis one-forms annihilate a wedge product.  The exterior derivative increases degree by one.  Some tensor-valued form examples are formal symbolic workflows; coordinate-level exterior derivatives should be used when component derivatives are required.

## Tensor-valued forms and Cartan equations

A tensor-valued $p$-form stores a form degree and a finite mapping from tensor-component keys to form expressions.  The wedge product uses the usual graded form sign and combines tensor keys by concatenation unless a specialized composition operation is requested.

Cartan workflows use the following conventions:

$$
T^a = d\theta^a + \omega^a{}_b \wedge \theta^b,
$$

$$
\Omega^a{}_b = d\omega^a{}_b + \omega^a{}_c \wedge \omega^c{}_b.
$$

Here $\theta^a$ is the solder form, $\omega^a{}_b$ is a connection one-form, $T^a$ is torsion, and $\Omega^a{}_b$ is curvature.

## Geometric algebra conventions

The native multivector layer implements geometric algebra for orthogonal metric signatures.  The geometric product satisfies

$$
e_i e_i = g_{ii}, \qquad e_i e_j = -e_j e_i \quad (i \ne j)
$$

for the diagonal metric entries supplied to the algebra.  Non-diagonal bilinear forms are intentionally rejected rather than simplified incorrectly.

The exterior product is the grade-raising antisymmetric part of the geometric product.  Reversion, grade involution, Clifford conjugation, duals, reflections, and rotors follow the orthogonal-basis conventions documented in the geometric-algebra API.

## Simplification policy

Most public symbolic routines accept a `simplify` option.  `simplify=True` uses SymPy simplification on selected components, `simplify=False` leaves expressions closer to their construction form, and a callable may be supplied for custom cleanup.  For large curvature calculations, selected-component APIs are often preferable to dense tensor construction.
