from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import sympy as sp

from .basis import (
    TensorBasis,
    TensorFrame,
    IndexBundle,
    frame_connection_coefficients,
    connection_one_forms,
    coframe_basis,
)
from .exterior_spin_algebra import CliffordAlgebraDef, gamma_generators, clifford_reduce


@dataclass(frozen=True)
class SpinConnectionDef:
    name: str
    frame: TensorBasis
    coefficients: Any
    one_forms: tuple[tuple[tuple[sp.Expr, ...], ...], ...]
    metric_signature: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExteriorFormNF:
    dimension: int
    terms: dict[tuple[int, ...], sp.Expr]
    basis_labels: tuple[str, ...] = tuple()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def degree(self) -> int:
        if not self.terms:
            return 0
        return len(next(iter(self.terms)))

    def __add__(self, other: 'ExteriorFormNF') -> 'ExteriorFormNF':
        if self.dimension != other.dimension:
            raise ValueError('Exterior forms must have the same dimension.')
        labels = self.basis_labels or other.basis_labels
        out: dict[tuple[int, ...], sp.Expr] = dict(self.terms)
        for key, value in other.terms.items():
            out[key] = sp.simplify(out.get(key, 0) + value)
            if sp.simplify(out[key]) == 0:
                out.pop(key, None)
        return ExteriorFormNF(self.dimension, out, basis_labels=labels)

    def __neg__(self) -> 'ExteriorFormNF':
        return ExteriorFormNF(self.dimension, {k: -v for k, v in self.terms.items()}, basis_labels=self.basis_labels, metadata=dict(self.metadata))

    def __sub__(self, other: 'ExteriorFormNF') -> 'ExteriorFormNF':
        return self + (-other)

    def scale(self, scalar: Any) -> 'ExteriorFormNF':
        scalar = sp.sympify(scalar)
        return ExteriorFormNF(self.dimension, {k: sp.simplify(scalar * v) for k, v in self.terms.items()}, basis_labels=self.basis_labels, metadata=dict(self.metadata))

    def wedge(self, other: 'ExteriorFormNF') -> 'ExteriorFormNF':
        return wedge_exterior_forms(self, other)


@dataclass(frozen=True)
class ExteriorIdentityReport:
    d_squared_zero: bool
    graded_leibniz_holds: bool
    associativity_holds: bool
    details: dict[str, Any] = field(default_factory=dict)


def _inversion_parity(seq: Sequence[int]) -> int:
    inv = 0
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if seq[i] > seq[j]:
                inv += 1
    return -1 if inv % 2 else 1


def _canonical_blade(indices: Sequence[int]) -> tuple[int, tuple[int, ...]]:
    indices = tuple(int(i) for i in indices)
    if len(set(indices)) != len(indices):
        return 0, tuple()
    return _inversion_parity(indices), tuple(sorted(indices))


def exterior_form_nf(terms: Mapping[Sequence[int], Any] | ExteriorFormNF, *, dimension: int | None = None, basis_labels: Sequence[str] | None = None, metadata: Mapping[str, Any] | None = None) -> ExteriorFormNF:
    if isinstance(terms, ExteriorFormNF):
        return terms
    if dimension is None:
        flat = [int(i) for blade in terms for i in blade]
        dimension = max(flat) + 1 if flat else 0
    out: dict[tuple[int, ...], sp.Expr] = {}
    for blade, coeff in terms.items():
        sign, canon = _canonical_blade(tuple(blade))
        if sign == 0:
            continue
        value = sp.simplify(sign * sp.sympify(coeff))
        out[canon] = sp.simplify(out.get(canon, 0) + value)
        if sp.simplify(out[canon]) == 0:
            out.pop(canon, None)
    labels = tuple(basis_labels or tuple(f'theta{i}' for i in range(int(dimension))))
    return ExteriorFormNF(int(dimension), out, basis_labels=labels, metadata=dict(metadata or {}))


def canonicalize_exterior_form(form: Mapping[Sequence[int], Any] | ExteriorFormNF, *, dimension: int | None = None, basis_labels: Sequence[str] | None = None) -> ExteriorFormNF:
    return exterior_form_nf(form, dimension=dimension, basis_labels=basis_labels)


def wedge_exterior_forms(left: ExteriorFormNF, right: ExteriorFormNF) -> ExteriorFormNF:
    if left.dimension != right.dimension:
        raise ValueError('Exterior forms must have the same dimension.')
    labels = left.basis_labels or right.basis_labels
    out: dict[tuple[int, ...], sp.Expr] = {}
    for lb, lc in left.terms.items():
        for rb, rc in right.terms.items():
            sign, canon = _canonical_blade(lb + rb)
            if sign == 0:
                continue
            value = sp.simplify(lc * rc * sign)
            out[canon] = sp.simplify(out.get(canon, 0) + value)
            if sp.simplify(out[canon]) == 0:
                out.pop(canon, None)
    return ExteriorFormNF(left.dimension, out, basis_labels=labels)


def exterior_derivative_nf(form: ExteriorFormNF, coordinates: Sequence[sp.Symbol]) -> ExteriorFormNF:
    coords = tuple(coordinates)
    dim = form.dimension
    out: dict[tuple[int, ...], sp.Expr] = {}
    for blade, coeff in form.terms.items():
        for i, coord in enumerate(coords):
            deriv = sp.simplify(sp.diff(coeff, coord))
            if deriv == 0:
                continue
            sign, canon = _canonical_blade((i,) + blade)
            if sign == 0:
                continue
            out[canon] = sp.simplify(out.get(canon, 0) + sign * deriv)
            if sp.simplify(out[canon]) == 0:
                out.pop(canon, None)
    return ExteriorFormNF(dim, out, basis_labels=form.basis_labels)


def exterior_identity_report(alpha: ExteriorFormNF, beta: ExteriorFormNF, coordinates: Sequence[sp.Symbol]) -> ExteriorIdentityReport:
    d_alpha = exterior_derivative_nf(alpha, coordinates)
    d_beta = exterior_derivative_nf(beta, coordinates)
    lhs_sq = exterior_derivative_nf(d_alpha, coordinates)
    d_sq_zero = all(sp.simplify(v) == 0 for v in lhs_sq.terms.values())

    lhs_leibniz = exterior_derivative_nf(alpha.wedge(beta), coordinates)
    rhs_leibniz = d_alpha.wedge(beta) + alpha.wedge(d_beta).scale((-1) ** alpha.degree)
    graded = lhs_leibniz.terms == rhs_leibniz.terms

    gamma = exterior_form_nf({(0,): sp.Symbol('a')}, dimension=max(alpha.dimension, beta.dimension, 1), basis_labels=alpha.basis_labels or beta.basis_labels or ('theta0',))
    assoc = alpha.wedge(beta).wedge(gamma).terms == alpha.wedge(beta.wedge(gamma)).terms

    return ExteriorIdentityReport(
        d_squared_zero=d_sq_zero,
        graded_leibniz_holds=graded,
        associativity_holds=assoc,
        details={
            'd_squared_terms': lhs_sq.terms,
            'lhs_leibniz': lhs_leibniz.terms,
            'rhs_leibniz': rhs_leibniz.terms,
        },
    )


def _ensure_frame_transform(frame: TensorBasis) -> TensorBasis:
    if frame.metadata.get('transform_to_chart') is not None or frame.chart is None:
        return frame
    chart = frame.chart
    coords = chart.symbols()
    dim = frame.dimension or chart.dimension
    if frame.kind in {'tangent', 'cotangent'}:
        def _identity(actual_coords, dim=dim):
            return sp.eye(dim)
        md = dict(frame.metadata)
        md['transform_to_chart'] = _identity
        return TensorBasis(frame.name, frame.kind, chart, frame.dimension, frame.dual_name, md)
    if frame.kind == 'orthonormal_tangent' and chart.is_orthogonal(coords):
        scales = chart.scale_factors(coords)
        def _orth_frame(actual_coords, chart=chart):
            local_scales = chart.scale_factors(tuple(actual_coords))
            return sp.diag(*[sp.simplify(1 / s) for s in local_scales])
        md = dict(frame.metadata)
        md['transform_to_chart'] = _orth_frame
        return TensorBasis(frame.name, frame.kind, chart, frame.dimension, frame.dual_name, md)
    if frame.kind == 'orthonormal_cotangent' and chart.is_orthogonal(coords):
        def _orth_coframe(actual_coords, chart=chart):
            local_scales = chart.scale_factors(tuple(actual_coords))
            return sp.diag(*[sp.simplify(s) for s in local_scales])
        md = dict(frame.metadata)
        md['transform_to_chart'] = _orth_coframe
        return TensorBasis(frame.name, frame.kind, chart, frame.dimension, frame.dual_name, md)
    return frame


def _signature_diagonal(frame: TensorBasis) -> tuple[int, ...]:
    metric = sp.Matrix(frame.chart.metric(frame.chart.symbols())) if frame.chart is not None and frame.chart.metric(frame.chart.symbols()) is not None else None
    if getattr(frame, 'kind', '').startswith('orthonormal'):
        if metric is not None:
            diag = []
            for i in range(metric.rows):
                entry = sp.simplify(metric[i, i])
                diag.append(-1 if entry.could_extract_minus_sign() else 1)
            return tuple(diag)
        return tuple(1 for _ in range(frame.dimension or 0))
    if metric is not None and metric.is_diagonal():
        diag = []
        for i in range(metric.rows):
            entry = sp.simplify(metric[i, i])
            diag.append(-1 if entry.could_extract_minus_sign() else 1)
        return tuple(diag)
    return tuple(1 for _ in range(frame.dimension or frame.chart.dimension))


def spin_connection(frame: TensorBasis, *, name: str = 'omega_spin', metric_signature: Sequence[int] | None = None, metadata: Mapping[str, Any] | None = None) -> SpinConnectionDef:
    frame = _ensure_frame_transform(frame)
    gamma = frame_connection_coefficients(frame)
    one_forms = connection_one_forms(frame)
    signature = tuple(int(s) for s in (metric_signature or _signature_diagonal(frame)))
    dim = frame.dimension or frame.chart.dimension
    # antisymmetrize the lowered first index for orthonormal-frame usage
    coeffs = {}
    for a in range(dim):
        for b in range(dim):
            for c in range(dim):
                lowered = signature[a] * gamma[a, b, c]
                lowered_flip = signature[b] * gamma[b, a, c]
                coeffs[(a, b, c)] = sp.simplify(sp.Rational(1, 2) * (lowered - lowered_flip))
    return SpinConnectionDef(name=name, frame=frame, coefficients=coeffs, one_forms=one_forms, metric_signature=signature, metadata=dict(metadata or {}))


def gamma_frame_generators(frame: TensorBasis, clifford: CliffordAlgebraDef) -> tuple[sp.Symbol, ...]:
    gens = gamma_generators(clifford)
    dim = frame.dimension or frame.chart.dimension
    if clifford.dimension != dim:
        raise ValueError('Clifford algebra dimension must match the frame dimension.')
    return gens


def spin_covariant_components(spinor: Any, frame: TensorBasis, clifford: CliffordAlgebraDef, *, spin_conn: SpinConnectionDef | None = None) -> tuple[sp.Expr, ...]:
    psi = sp.sympify(spinor)
    frame = _ensure_frame_transform(frame)
    sc = spin_conn or spin_connection(frame)
    coords = frame.chart.symbols()
    gens = gamma_frame_generators(frame, clifford)
    dim = frame.dimension or frame.chart.dimension
    out = []
    for c in range(dim):
        base = sp.diff(psi, coords[c])
        correction = sp.Integer(0)
        for a in range(dim):
            for b in range(dim):
                coeff = sc.coefficients[(a, b, c)]
                if coeff == 0:
                    continue
                correction += sp.Rational(1, 4) * coeff * gens[a] * gens[b] * psi
        out.append(clifford_reduce(base + correction, clifford))
    return tuple(out)


def dirac_operator(spinor: Any, frame: TensorBasis, clifford: CliffordAlgebraDef, *, spin_conn: SpinConnectionDef | None = None) -> sp.Expr:
    frame = _ensure_frame_transform(frame)
    gens = gamma_frame_generators(frame, clifford)
    comps = spin_covariant_components(spinor, frame, clifford, spin_conn=spin_conn)
    return clifford_reduce(sum((gens[i] * comps[i] for i in range(len(comps))), sp.Integer(0)), clifford)


def serialize_basis(basis: TensorBasis) -> dict[str, Any]:
    payload = {
        'type': 'TensorBasis',
        'name': basis.name,
        'kind': basis.kind,
        'dimension': basis.dimension,
        'dual_name': basis.dual_name,
        'chart_name': None if basis.chart is None else basis.chart.chart_name,
        'metric_name': None if basis.chart is None else basis.chart.metric_name,
        'metadata': {k: v for k, v in basis.metadata.items() if k not in {'transform_to_chart', 'bundle'}},
    }
    bundle = basis.metadata.get('bundle') if hasattr(basis, 'metadata') else None
    if isinstance(bundle, IndexBundle):
        payload['bundle'] = {'name': bundle.name, 'dimension': bundle.dimension}
    tf = basis.metadata.get('transform_to_chart') if hasattr(basis, 'metadata') else None
    if tf is not None and basis.chart is not None:
        coords = basis.chart.symbols()
        matrix = sp.Matrix(tf(coords))
        payload['transform_matrix_entries'] = [[sp.srepr(sp.sympify(matrix[i, j])) for j in range(matrix.cols)] for i in range(matrix.rows)]
        payload['coordinate_names'] = [str(c) for c in coords]
    return payload


def deserialize_basis(payload: Mapping[str, Any], chart: Any) -> TensorBasis:
    metadata = dict(payload.get('metadata', {}))
    bundle_payload = payload.get('bundle')
    if bundle_payload is not None:
        metadata['bundle'] = IndexBundle(bundle_payload['name'], bundle_payload.get('dimension'))
    entries = payload.get('transform_matrix_entries')
    if entries is not None:
        source_coord_names = tuple(payload.get('coordinate_names', [str(c) for c in chart.symbols()]))
        source_coords = sp.symbols(' '.join(source_coord_names)) if source_coord_names else tuple()
        if not isinstance(source_coords, tuple):
            source_coords = (source_coords,)
        frozen = sp.Matrix([[sp.sympify(cell) for cell in row] for row in entries])
        def _transform(actual_coords, frozen=frozen, source_coords=source_coords):
            subs = dict(zip(source_coords, tuple(actual_coords)))
            return frozen.subs(subs)
        metadata['transform_to_chart'] = _transform
    return TensorBasis(
        name=payload['name'],
        kind=payload['kind'],
        chart=chart,
        dimension=payload.get('dimension'),
        dual_name=payload.get('dual_name'),
        metadata=metadata,
    )


def serialize_frame(frame: TensorFrame) -> dict[str, Any]:
    return {
        'type': 'TensorFrame',
        'name': frame.name,
        'kind': frame.kind,
        'dimension': frame.dimension,
        'dual_name': frame.dual_name,
        'metadata': dict(frame.metadata),
        'bundle': None if frame.bundle is None else {'name': frame.bundle.name, 'dimension': frame.bundle.dimension},
        'basis': serialize_basis(frame.as_basis()),
    }


def deserialize_frame(payload: Mapping[str, Any], chart: Any) -> TensorFrame:
    basis = deserialize_basis(payload['basis'], chart)
    bundle_payload = payload.get('bundle')
    bundle = None if bundle_payload is None else IndexBundle(bundle_payload['name'], bundle_payload.get('dimension'))
    tf = basis.metadata.get('transform_to_chart')
    if tf is None:
        raise ValueError('Serialized frame is missing transform_to_chart data.')
    return TensorFrame(
        name=payload['name'],
        kind=payload['kind'],
        chart=chart,
        dimension=int(payload.get('dimension', chart.dimension)),
        transform_to_chart=tf,
        dual_name=payload.get('dual_name'),
        metadata=dict(payload.get('metadata', {})),
        bundle=bundle,
    )


__all__ = [
    'SpinConnectionDef',
    'ExteriorFormNF',
    'ExteriorIdentityReport',
    'spin_connection',
    'gamma_frame_generators',
    'spin_covariant_components',
    'dirac_operator',
    'exterior_form_nf',
    'canonicalize_exterior_form',
    'wedge_exterior_forms',
    'exterior_derivative_nf',
    'exterior_identity_report',
    'serialize_basis',
    'deserialize_basis',
    'serialize_frame',
    'deserialize_frame',
]
