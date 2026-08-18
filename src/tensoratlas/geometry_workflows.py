from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import sympy as sp

from .basis import TensorBasis, frame_to_chart_matrix, connection_one_forms
from .charts import CoordinateChart
from .exterior_geometry import ExteriorFormNF, canonicalize_exterior_form
from .exterior_spin_algebra import CliffordAlgebraDef, clifford_algebra
from .semantic_exterior_spin import (
    hodge_star_nf,
    codifferential_nf,
    interior_product_nf,
    lie_derivative_nf,
    hodge_laplacian_nf,
    spin_connection,
    dirac_operator,
)


@dataclass(frozen=True)
class ExteriorExecutionReport:
    hodge: ExteriorFormNF
    codifferential: ExteriorFormNF
    interior: ExteriorFormNF
    lie: ExteriorFormNF
    laplacian: ExteriorFormNF
    signature: tuple[int, ...]
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpinExecutionReport:
    spin_connection_name: str
    dirac_expression: sp.Expr
    gamma_count: int
    signature: tuple[int, ...]
    provenance: dict[str, Any] = field(default_factory=dict)



def metric_signature_from_chart(chart: CoordinateChart, coords: Sequence[sp.Symbol] | None = None) -> tuple[int, ...]:
    actual = tuple(coords) if coords is not None else chart.symbols()
    metric = chart.metric(actual)
    if metric is None:
        return tuple(1 for _ in range(chart.dimension))
    out = []
    for i in range(metric.rows):
        entry = sp.simplify(metric[i, i])
        if entry.is_number:
            out.append(1 if entry > 0 else -1 if entry < 0 else 0)
        else:
            pos = sp.ask(sp.Q.positive(entry))
            neg = sp.ask(sp.Q.negative(entry))
            out.append(1 if pos else -1 if neg else 1)
    return tuple(int(v) for v in out)



def exterior_execution_pipeline(form: ExteriorFormNF, *, chart: CoordinateChart | None = None, vector_components: Sequence[Any] | Mapping[int, Any] | None = None, coordinates: Sequence[sp.Symbol] | None = None, clifford: CliffordAlgebraDef | None = None, metric_signature: Sequence[int] | None = None) -> ExteriorExecutionReport:
    form = canonicalize_exterior_form(form)
    coords = tuple(coordinates) if coordinates is not None else tuple(chart.symbols() if chart is not None else tuple(sp.Symbol(f'x{i}') for i in range(form.dimension)))
    sig = tuple(metric_signature) if metric_signature is not None else (metric_signature_from_chart(chart, coords) if chart is not None else tuple(1 for _ in range(form.dimension)))
    cl = clifford or clifford_algebra(form.dimension, (sum(1 for s in sig if s > 0), sum(1 for s in sig if s < 0), sum(1 for s in sig if s == 0)), basis_labels=form.basis_labels or tuple(str(i) for i in range(form.dimension)))
    vec = vector_components or tuple(0 for _ in range(form.dimension))
    h = hodge_star_nf(form, clifford=cl, metric_signature=sig).form
    c = codifferential_nf(form, coords, clifford=cl, metric_signature=sig)
    i = interior_product_nf(vec, form)
    l = lie_derivative_nf(vec, form, coords).result
    lap = hodge_laplacian_nf(form, coords, clifford=cl, metric_signature=sig)
    return ExteriorExecutionReport(hodge=h, codifferential=c, interior=i, lie=l, laplacian=lap, signature=tuple(int(s) for s in sig), provenance={'operation': 'exterior_execution_pipeline'})



def spin_execution_pipeline(spinor: Any, frame: TensorBasis, *, clifford: CliffordAlgebraDef | None = None, coordinates: Sequence[sp.Symbol] | None = None, metric_signature: Sequence[int] | None = None) -> SpinExecutionReport:
    coords = tuple(coordinates) if coordinates is not None else tuple(frame.chart.symbols())
    sig = tuple(metric_signature) if metric_signature is not None else metric_signature_from_chart(frame.chart, coords)
    cl = clifford or clifford_algebra(frame.dimension or len(sig), (sum(1 for s in sig if s > 0), sum(1 for s in sig if s < 0), sum(1 for s in sig if s == 0)), basis_labels=tuple(str(i) for i in range(frame.dimension or len(sig))))
    sc = spin_connection(frame, metric_signature=sig)
    expr = dirac_operator(spinor, frame, cl, coordinates=coords, spin_conn=sc)
    return SpinExecutionReport(spin_connection_name=sc.name, dirac_expression=sp.expand(expr), gamma_count=len([f for f in sp.Mul.make_args(sp.expand(expr)) if getattr(f, 'is_commutative', True) is False]), signature=tuple(int(s) for s in sig), provenance={'operation': 'spin_execution_pipeline'})


__all__ = [
    'ExteriorExecutionReport',
    'SpinExecutionReport',
    'metric_signature_from_chart',
    'exterior_execution_pipeline',
    'spin_execution_pipeline',
]
