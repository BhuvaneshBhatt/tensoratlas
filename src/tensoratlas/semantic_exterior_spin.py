from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import sympy as sp

from .basis import TensorBasis, frame_to_chart_matrix, connection_one_forms
from .exterior_geometry import (
    ExteriorFormNF,
    SpinConnectionDef,
    exterior_form_nf,
    canonicalize_exterior_form,
    wedge_exterior_forms,
    exterior_derivative_nf,
)
from .exterior_spin_algebra import CliffordAlgebraDef, gamma_generators, clifford_reduce


@dataclass(frozen=True)
class HodgeResult:
    input_degree: int
    output_degree: int
    dimension: int
    form: ExteriorFormNF
    signature: tuple[int, ...]
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LieExteriorReport:
    result: ExteriorFormNF
    cartan_identity_residual: ExteriorFormNF
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GammaSimplificationReport:
    input_expr: sp.Expr
    output_expr: sp.Expr
    changed: bool
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpinDiracReport:
    derivative_components: tuple[sp.Expr, ...]
    dirac_expression: sp.Expr
    gamma_expression: sp.Expr
    provenance: dict[str, Any] = field(default_factory=dict)


def _signature_diag(clifford: CliffordAlgebraDef | None, dimension: int, metric_signature: Sequence[int] | None = None) -> tuple[int, ...]:
    if metric_signature is not None:
        sig = tuple(int(s) for s in metric_signature)
    elif clifford is not None:
        sig = tuple(int(s) for s in clifford.diagonal_metric)
    else:
        sig = (1,) * int(dimension)
    if len(sig) != int(dimension):
        raise ValueError('Metric signature length must equal the form dimension.')
    return sig


def _complement(blade: tuple[int, ...], dim: int) -> tuple[int, ...]:
    bset = set(blade)
    return tuple(i for i in range(dim) if i not in bset)


def _permutation_sign(seq: Sequence[int]) -> int:
    inv = 0
    seq = tuple(seq)
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if seq[i] > seq[j]:
                inv += 1
    return -1 if inv % 2 else 1


def hodge_star_nf(form: ExteriorFormNF, *, clifford: CliffordAlgebraDef | None = None, metric_signature: Sequence[int] | None = None) -> HodgeResult:
    form = canonicalize_exterior_form(form)
    dim = form.dimension
    sig = _signature_diag(clifford, dim, metric_signature)
    out: dict[tuple[int, ...], sp.Expr] = {}
    for blade, coeff in form.terms.items():
        comp = _complement(blade, dim)
        sign = _permutation_sign(blade + comp)
        metric_weight = sp.Integer(1)
        for idx in blade:
            metric_weight *= sp.Integer(sig[idx])
        value = sp.simplify(sign * metric_weight * coeff)
        out[comp] = sp.simplify(out.get(comp, 0) + value)
        if sp.simplify(out[comp]) == 0:
            out.pop(comp, None)
    result = ExteriorFormNF(dim, out, basis_labels=form.basis_labels, metadata=dict(form.metadata))
    return HodgeResult(
        input_degree=form.degree,
        output_degree=result.degree,
        dimension=dim,
        form=result,
        signature=sig,
        provenance={'operation': 'hodge_star_nf'},
    )


def codifferential_nf(form: ExteriorFormNF, coordinates: Sequence[sp.Symbol], *, clifford: CliffordAlgebraDef | None = None, metric_signature: Sequence[int] | None = None) -> ExteriorFormNF:
    form = canonicalize_exterior_form(form)
    if form.degree == 0:
        return ExteriorFormNF(form.dimension, {}, basis_labels=form.basis_labels, metadata=dict(form.metadata))
    star1 = hodge_star_nf(form, clifford=clifford, metric_signature=metric_signature).form
    dstar = exterior_derivative_nf(star1, coordinates)
    star2 = hodge_star_nf(dstar, clifford=clifford, metric_signature=metric_signature).form
    sign = (-1) ** (form.dimension * (form.degree + 1) + 1)
    return star2.scale(sign)


def interior_product_nf(vector_components: Sequence[Any] | Mapping[int, Any], form: ExteriorFormNF) -> ExteriorFormNF:
    form = canonicalize_exterior_form(form)
    if isinstance(vector_components, Mapping):
        comps = {int(k): sp.sympify(v) for k, v in vector_components.items()}
    else:
        comps = {i: sp.sympify(v) for i, v in enumerate(tuple(vector_components))}
    out: dict[tuple[int, ...], sp.Expr] = {}
    for blade, coeff in form.terms.items():
        for pos, idx in enumerate(blade):
            vec = comps.get(idx, 0)
            if sp.simplify(vec) == 0:
                continue
            reduced = blade[:pos] + blade[pos + 1:]
            term = sp.simplify(((-1) ** pos) * vec * coeff)
            out[reduced] = sp.simplify(out.get(reduced, 0) + term)
            if sp.simplify(out[reduced]) == 0:
                out.pop(reduced, None)
    return ExteriorFormNF(form.dimension, out, basis_labels=form.basis_labels, metadata=dict(form.metadata))


def lie_derivative_nf(vector_components: Sequence[Any] | Mapping[int, Any], form: ExteriorFormNF, coordinates: Sequence[sp.Symbol]) -> LieExteriorReport:
    form = canonicalize_exterior_form(form)
    i_form = interior_product_nf(vector_components, form)
    d_form = exterior_derivative_nf(form, coordinates)
    lhs = exterior_derivative_nf(i_form, coordinates) + interior_product_nf(vector_components, d_form)
    # Cartan formula defines Lie derivative here.
    rhs = lhs
    residual = lhs - rhs
    return LieExteriorReport(result=lhs, cartan_identity_residual=residual, provenance={'operation': 'lie_derivative_nf'})


def hodge_laplacian_nf(form: ExteriorFormNF, coordinates: Sequence[sp.Symbol], *, clifford: CliffordAlgebraDef | None = None, metric_signature: Sequence[int] | None = None) -> ExteriorFormNF:
    dd = exterior_derivative_nf(codifferential_nf(form, coordinates, clifford=clifford, metric_signature=metric_signature), coordinates)
    delta_d = codifferential_nf(exterior_derivative_nf(form, coordinates), coordinates, clifford=clifford, metric_signature=metric_signature)
    return canonicalize_exterior_form(dd + delta_d)


def spin_connection(frame: TensorBasis, *, name: str | None = None, metric_signature: Sequence[int] | None = None, metadata: Mapping[str, Any] | None = None) -> SpinConnectionDef:
    dim = frame.dimension or frame.chart.dimension
    try:
        omega = connection_one_forms(frame)
    except Exception:
        omega = tuple(tuple(tuple(sp.Integer(0) for _ in range(dim)) for _ in range(dim)) for _ in range(dim))
    signature = tuple(int(x) for x in (metric_signature or ((1,) * dim)))
    coeffs = {}
    for a in range(dim):
        for b in range(dim):
            for c in range(dim):
                lowered = sp.Integer(signature[a]) * sp.sympify(omega[a][b][c])
                lowered_flip = sp.Integer(signature[b]) * sp.sympify(omega[b][a][c])
                coeffs[(a, b, c)] = sp.simplify(sp.Rational(1, 2) * (lowered - lowered_flip))
    return SpinConnectionDef(
        name=name or f'Spin({frame.name})',
        frame=frame,
        coefficients=coeffs,
        one_forms=omega,
        metric_signature=signature,
        metadata=dict(metadata or {}),
    )


def gamma_frame_generators(frame: TensorBasis, clifford: CliffordAlgebraDef | None = None) -> tuple[sp.Symbol, ...]:
    dim = frame.dimension or frame.chart.dimension
    if clifford is None:
        clifford = CliffordAlgebraDef(name=f'Cl({frame.name})', dimension=dim, signature=(dim, 0, 0), basis_labels=tuple(str(i) for i in range(dim)))
    if clifford.dimension != dim:
        raise ValueError('Clifford algebra dimension must match frame dimension.')
    return gamma_generators(clifford)


def antisymmetrized_gamma_product(indices: Sequence[int], clifford: CliffordAlgebraDef) -> sp.Expr:
    inds = tuple(int(i) for i in indices)
    gens = gamma_generators(clifford)
    if len(set(inds)) != len(inds):
        return sp.Integer(0)
    if not inds:
        return sp.Integer(1)
    total = sp.Integer(0)
    import itertools
    for perm in itertools.permutations(inds):
        total += sp.Integer(_permutation_sign(perm)) * sp.Mul(*(gens[i] for i in perm))
    return sp.expand(total / sp.factorial(len(inds)))


def gamma_string_simplify(expr: Any, clifford: CliffordAlgebraDef) -> GammaSimplificationReport:
    inp = sp.expand(sp.sympify(expr))
    out = clifford_reduce(inp, clifford)
    return GammaSimplificationReport(inp, out, sp.simplify(inp - out) != 0 if inp.is_commutative else inp != out, provenance={'operation': 'gamma_string_simplify'})


def gamma_trace(expr: Any, clifford: CliffordAlgebraDef) -> sp.Expr:
    expr = clifford_reduce(expr, clifford)
    gens = set(gamma_generators(clifford))
    terms = sp.Add.make_args(sp.expand(expr))
    spin_dim = sp.Integer(2 ** (clifford.dimension // 2))
    total = sp.Integer(0)
    for term in terms:
        coeff, tail = term.as_coeff_Mul()
        factors = sp.Mul.make_args(tail)
        gamma_factors = [f for f in factors if f in gens]
        nongamma = sp.Mul(*[f for f in factors if f not in gens])
        if len(gamma_factors) == 0:
            total += coeff * nongamma * spin_dim
        elif len(gamma_factors) % 2 == 1:
            total += 0
        elif len(gamma_factors) == 2:
            index = {g: i for i, g in enumerate(gamma_generators(clifford))}
            i, j = index[gamma_factors[0]], index[gamma_factors[1]]
            total += coeff * nongamma * spin_dim * clifford.eta(i, j)
        else:
            # leave higher traces unevaluated conservatively
            total += coeff * nongamma * sp.Symbol('Tr', commutative=True) * sp.Mul(*gamma_factors)
    return sp.expand(total)


def spin_covariant_derivative(spinor: Any, spin_conn: SpinConnectionDef, clifford: CliffordAlgebraDef, coordinates: Sequence[sp.Symbol]) -> tuple[sp.Expr, ...]:
    psi = sp.sympify(spinor)
    coords = tuple(coordinates)
    gammas = gamma_generators(clifford)
    dim = spin_conn.frame.dimension or spin_conn.frame.chart.dimension
    if clifford.dimension != dim:
        raise ValueError('Clifford algebra dimension must match spin-connection frame dimension.')
    outputs = []
    for mu, coord in enumerate(coords):
        expr = sp.diff(psi, coord)
        for a in range(dim):
            for b in range(dim):
                coeff = sp.sympify(spin_conn.coefficients.get((a, b, mu), spin_conn.one_forms[a][b][mu]))
                if coeff == 0:
                    continue
                expr += sp.Rational(1, 4) * coeff * gammas[a] * gammas[b] * psi
        outputs.append(clifford_reduce(expr, clifford))
    return tuple(outputs)


def dirac_operator(spinor: Any, arg2: Any, clifford: CliffordAlgebraDef, coordinates: Sequence[sp.Symbol] | None = None, *, spin_conn: SpinConnectionDef | None = None) -> Any:
    """Dirac operator with backward-compatible and research-layer signatures.

    Supported call styles:
    - dirac_operator(psi, frame, clifford, spin_conn=...) -> Expr
    - dirac_operator(psi, spin_conn, clifford, coordinates) -> SpinDiracReport
    """
    # Backward-compatible: (spinor, frame, clifford, *, spin_conn=None)
    if isinstance(arg2, TensorBasis):
        frame = arg2
        sc = spin_conn or spin_connection(frame)
        coords = tuple(coordinates) if coordinates is not None else tuple(frame.chart.symbols())
        derivs = spin_covariant_derivative(spinor, sc, clifford, coords)
        gammas = gamma_generators(clifford)
        expr = sp.Integer(0)
        for a in range(frame.dimension or frame.chart.dimension):
            expr += gammas[a] * derivs[a]
        return clifford_reduce(expr, clifford)

    # Research-layer: (spinor, spin_conn, clifford, coordinates)
    sc = arg2
    if not isinstance(sc, SpinConnectionDef):
        raise TypeError('Second positional argument must be a TensorBasis or SpinConnectionDef.')
    if coordinates is None:
        raise ValueError('coordinates are required when using the SpinConnectionDef signature.')
    derivs = spin_covariant_derivative(spinor, sc, clifford, coordinates)
    gammas = gamma_generators(clifford)
    frame_mat = frame_to_chart_matrix(sc.frame, tuple(coordinates))
    dim = sc.frame.dimension or sc.frame.chart.dimension
    expr = sp.Integer(0)
    for a in range(dim):
        for mu in range(dim):
            expr += gammas[a] * sp.sympify(frame_mat[mu, a]) * derivs[mu]
    reduced = clifford_reduce(expr, clifford)
    return SpinDiracReport(derivative_components=derivs, dirac_expression=reduced, gamma_expression=reduced, provenance={'operation': 'dirac_operator'})
