from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import sympy as sp

from .exterior_geometry import ExteriorFormNF, canonicalize_exterior_form
from .exterior_spin_algebra import CliffordAlgebraDef


@dataclass(frozen=True)
class HodgeExpr:
    form: ExteriorFormNF
    clifford: CliffordAlgebraDef | None = None
    metric_signature: tuple[int, ...] = tuple()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CodifferentialExpr:
    form: ExteriorFormNF
    coordinates: tuple[sp.Symbol, ...]
    clifford: CliffordAlgebraDef | None = None
    metric_signature: tuple[int, ...] = tuple()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InteriorExpr:
    vector_components: tuple[sp.Expr, ...]
    form: ExteriorFormNF
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LieExpr:
    vector_components: tuple[sp.Expr, ...]
    form: ExteriorFormNF
    coordinates: tuple[sp.Symbol, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GammaStringExpr:
    clifford: CliffordAlgebraDef
    factors: tuple[int, ...]
    scalar: sp.Expr = sp.Integer(1)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def degree(self) -> int:
        return len(self.factors)



def hodge_expr(form: ExteriorFormNF, *, clifford: CliffordAlgebraDef | None = None, metric_signature: Sequence[int] | None = None, metadata: Mapping[str, Any] | None = None) -> HodgeExpr:
    return HodgeExpr(
        form=canonicalize_exterior_form(form),
        clifford=clifford,
        metric_signature=tuple(int(x) for x in (metric_signature or tuple())),
        metadata=dict(metadata or {}),
    )



def codifferential_expr(form: ExteriorFormNF, coordinates: Sequence[sp.Symbol], *, clifford: CliffordAlgebraDef | None = None, metric_signature: Sequence[int] | None = None, metadata: Mapping[str, Any] | None = None) -> CodifferentialExpr:
    return CodifferentialExpr(
        form=canonicalize_exterior_form(form),
        coordinates=tuple(coordinates),
        clifford=clifford,
        metric_signature=tuple(int(x) for x in (metric_signature or tuple())),
        metadata=dict(metadata or {}),
    )



def interior_expr(vector_components: Sequence[Any] | Mapping[int, Any], form: ExteriorFormNF, *, metadata: Mapping[str, Any] | None = None) -> InteriorExpr:
    if isinstance(vector_components, Mapping):
        max_idx = max((int(k) for k in vector_components.keys()), default=-1)
        comps = tuple(sp.sympify(vector_components.get(i, 0)) for i in range(max_idx + 1))
    else:
        comps = tuple(sp.sympify(v) for v in vector_components)
    return InteriorExpr(vector_components=comps, form=canonicalize_exterior_form(form), metadata=dict(metadata or {}))



def lie_expr(vector_components: Sequence[Any] | Mapping[int, Any], form: ExteriorFormNF, coordinates: Sequence[sp.Symbol], *, metadata: Mapping[str, Any] | None = None) -> LieExpr:
    if isinstance(vector_components, Mapping):
        max_idx = max((int(k) for k in vector_components.keys()), default=-1)
        comps = tuple(sp.sympify(vector_components.get(i, 0)) for i in range(max_idx + 1))
    else:
        comps = tuple(sp.sympify(v) for v in vector_components)
    return LieExpr(vector_components=comps, form=canonicalize_exterior_form(form), coordinates=tuple(coordinates), metadata=dict(metadata or {}))



def gamma_string(clifford: CliffordAlgebraDef, factors: Sequence[int], *, scalar: Any = 1, metadata: Mapping[str, Any] | None = None) -> GammaStringExpr:
    return GammaStringExpr(clifford=clifford, factors=tuple(int(i) for i in factors), scalar=sp.sympify(scalar), metadata=dict(metadata or {}))



def gamma_string_to_sympy(expr: GammaStringExpr) -> sp.Expr:
    from .exterior_spin_algebra import gamma_generators

    gens = gamma_generators(expr.clifford)
    if not expr.factors:
        return sp.sympify(expr.scalar)
    return sp.expand(sp.sympify(expr.scalar) * sp.Mul(*(gens[i] for i in expr.factors)))



def evaluate_semantic_operator(obj: Any) -> Any:
    from .semantic_exterior_spin import (
        hodge_star_nf,
        codifferential_nf,
        interior_product_nf,
        lie_derivative_nf,
        gamma_string_simplify,
    )

    if isinstance(obj, HodgeExpr):
        return hodge_star_nf(obj.form, clifford=obj.clifford, metric_signature=obj.metric_signature or None).form
    if isinstance(obj, CodifferentialExpr):
        return codifferential_nf(obj.form, obj.coordinates, clifford=obj.clifford, metric_signature=obj.metric_signature or None)
    if isinstance(obj, InteriorExpr):
        return interior_product_nf(obj.vector_components, obj.form)
    if isinstance(obj, LieExpr):
        return lie_derivative_nf(obj.vector_components, obj.form, obj.coordinates).result
    if isinstance(obj, GammaStringExpr):
        return gamma_string_simplify(gamma_string_to_sympy(obj), obj.clifford).output_expr
    return obj


__all__ = [
    'HodgeExpr',
    'CodifferentialExpr',
    'InteriorExpr',
    'LieExpr',
    'GammaStringExpr',
    'hodge_expr',
    'codifferential_expr',
    'interior_expr',
    'lie_expr',
    'gamma_string',
    'gamma_string_to_sympy',
    'evaluate_semantic_operator',
]
