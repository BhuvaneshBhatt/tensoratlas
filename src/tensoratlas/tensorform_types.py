from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import sympy as sp

@dataclass(frozen=True)
class TensorFormTerm:
    scalar: sp.Expr
    factors: tuple
    free_signature: tuple
    bundle_signature: tuple

    def semantic_key(self) -> tuple[Any, ...]:
        return (tuple(self.factors), tuple(self.free_signature), tuple(self.bundle_signature))

    def with_scalar(self, scalar: sp.Expr) -> "TensorFormTerm":
        return TensorFormTerm(scalar, self.factors, self.free_signature, self.bundle_signature)

    @property
    def is_scalar_only(self) -> bool:
        return not self.factors

    @property
    def factor_count(self) -> int:
        return len(self.factors)

@dataclass(frozen=True)
class IndexedTensorForm:
    terms: tuple

    def semantic_terms(self) -> tuple[tuple[Any, ...], ...]:
        return tuple(t.semantic_key() if hasattr(t, "semantic_key") else (tuple(getattr(t, "factors", tuple())), tuple(getattr(t, "free_signature", tuple())), tuple(getattr(t, "bundle_signature", tuple()))) for t in self.terms)

    @property
    def is_zero(self) -> bool:
        return not self.terms

@dataclass(frozen=True)
class TensorOptimizationReport:
    original_kind: str
    optimized_kind: str
    removed_zero_terms: int
    removed_identity_terms: int
    scalar_factor_extracted: object
    used_component_expansion: bool

@dataclass(frozen=True)
class TensorSpace:
    name: str
    dimension: int
    parent: object | None = None

@dataclass(frozen=True)
class ContractionPlan:
    ordered_factors: tuple
    priorities: tuple
    estimated_cost: int

@dataclass(frozen=True)
class NormalizationDiagnostics:
    used_cache: bool
    used_optimizer_prepass: bool
    used_component_expansion: bool
    passes: int
    tier: int
    contraction_plan_cost: int
    removed_zero_terms: int
    removed_identity_terms: int

@dataclass(frozen=True)
class AbstractIndexedExpr:
    expr: object

@dataclass(frozen=True)
class ComponentIndexedExpr:
    expr: object

def wrap_abstract(obj: Any) -> AbstractIndexedExpr:
    if isinstance(obj, (AbstractIndexedExpr, ComponentIndexedExpr)):
        return obj
    return AbstractIndexedExpr(obj)

def unwrap_layer(obj: Any):
    if isinstance(obj, (AbstractIndexedExpr, ComponentIndexedExpr)):
        return obj.expr
    return obj

def abstract_layer(obj: Any) -> AbstractIndexedExpr:
    return AbstractIndexedExpr(unwrap_layer(obj))

def component_layer(obj: Any) -> ComponentIndexedExpr:
    return ComponentIndexedExpr(unwrap_layer(obj))

def is_abstract_layer(obj: Any) -> bool:
    return isinstance(obj, AbstractIndexedExpr)

def is_component_layer(obj: Any) -> bool:
    return isinstance(obj, ComponentIndexedExpr)
