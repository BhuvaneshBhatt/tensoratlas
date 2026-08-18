from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import sympy as sp

from .abstract_tensor import (
    AbstractNormalForm,
    BridgeConversionReport,
    abstract_normal_form,
    canonical_tensor_expression,
    canonical_tensor_normal_form,
    component_to_abstract,
    bridge_tensor_expression,
)
from .tensor_algebra import IndexedReductionReport, tensor_reduce_indexed_staged
from .rewrite_families import RewritePolicy, execute_rewrite_policy, select_rewrite_policy
from .semantic_rewrite import SemanticRewriteRule, semantic_rewrite
from .semantic_core import CanonicalSemanticForm, canonical_semantic_form, semantic_execute, semantic_ir, semantic_ir_for_object, semantic_layer_of


@dataclass(frozen=True)
class ProofTraceStep:
    name: str
    before: Any
    after: Any
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReductionTrace:
    layer: str
    original: Any
    final: Any
    steps: tuple[ProofTraceStep, ...]


@dataclass(frozen=True)
class UnifiedTensorNormalForm:
    layer: str
    expr: Any
    key: tuple[Any, ...]
    semantic_form: CanonicalSemanticForm | None = None
    abstract: AbstractNormalForm | None = None
    indexed_report: IndexedReductionReport | None = None


def _is_exterior_form(obj: Any) -> bool:
    return type(obj).__name__ == "ExteriorFormNF"


def _is_semantic_research_object(obj: Any) -> bool:
    return type(obj).__name__ in {"ExteriorFormNF", "SpinConnectionDef", "CliffordAlgebraDef", "TensorBasis", "TensorFrame", "HodgeExpr", "CodifferentialExpr", "InteriorExpr", "LieExpr", "GammaStringExpr"}


def _is_indexed(obj: Any) -> bool:
    try:
        from .tensor_indices import IndexedTensor, IndexedTensorExpr
    except Exception:
        return False
    return isinstance(obj, (IndexedTensor, IndexedTensorExpr))


def unified_tensor_normal_form(obj: Any, *, dimension: int | sp.Expr | None = None, policy: RewritePolicy | None = None, semantic_rules: tuple[SemanticRewriteRule, ...] = tuple()) -> UnifiedTensorNormalForm:
    if semantic_rules:
        obj, _ = semantic_rewrite(obj, semantic_rules, dimension=dimension)
    if _is_exterior_form(obj):
        from .exterior_geometry import canonicalize_exterior_form
        canon = canonicalize_exterior_form(obj)
        sem = semantic_execute(canon, layer="exterior", dimension=dimension, metadata={"normalization": "exterior_nf"})
        return UnifiedTensorNormalForm("exterior", canon, sem.key, semantic_form=sem)
    if _is_semantic_research_object(obj):
        sem = semantic_execute(obj, layer=semantic_layer_of(obj, default="abstract"), dimension=dimension)
        return UnifiedTensorNormalForm(sem.ir.layer, obj, sem.key, semantic_form=sem)
    if _is_indexed(obj):
        # Prefer the central TensorExpr canonicalizer for indexed expressions.
        # The older staged indexed reducer can be expensive for simple product
        # comparisons and was responsible for nontermination in broad suites.
        from .tensor_expr_canonicalization import canonicalize_tensor_expr
        report_expr = canonicalize_tensor_expr(obj)
        return UnifiedTensorNormalForm('indexed', report_expr.canonical, report_expr.canonical_key, semantic_form=None, indexed_report=None)
    chosen = policy if policy is not None else select_rewrite_policy(layer='abstract')
    reduced, trace = execute_rewrite_policy(obj, chosen, with_trace=True, dimension=dimension)
    abs_nf = canonical_tensor_normal_form(reduced, dimension=dimension)
    sem = semantic_execute(abs_nf.expr, layer='abstract', dimension=dimension, metadata={'policy': chosen.name})
    return UnifiedTensorNormalForm('abstract', abs_nf.expr, sem.key, semantic_form=sem, abstract=abs_nf)


def _abstract_contracted_product_signature(expr: Any) -> tuple[Any, ...] | None:
    """Small alpha-renaming signature for scalar products of SymPy tensors."""
    try:
        from itertools import permutations
        from sympy.tensor.tensor import Tensor, TensMul
    except Exception:
        return None
    factors = tuple(expr.args) if isinstance(expr, TensMul) else (expr,)
    if not factors or not all(isinstance(f, Tensor) for f in factors):
        return None

    def index_name(index: Any) -> str:
        return str(index).lstrip('-')

    def index_variance(index: Any) -> str:
        return 'l' if str(index).startswith('-') else 'u'

    signatures = []
    for order in permutations(range(len(factors))):
        labels: dict[str, str] = {}
        next_id = 0
        factor_sigs = []
        for pos in order:
            factor = factors[pos]
            slots = []
            for ind in factor.get_indices():
                raw = index_name(ind)
                if raw not in labels:
                    labels[raw] = f'd{next_id}'
                    next_id += 1
                slots.append((labels[raw], index_variance(ind)))
            factor_sigs.append((str(factor.component), tuple(slots)))
        signatures.append(tuple(factor_sigs))
    return min(signatures, key=repr)


def compare_unified_normal_forms(left: Any, right: Any, *, dimension: int | sp.Expr | None = None, policy: RewritePolicy | None = None, semantic_rules: tuple[SemanticRewriteRule, ...] = tuple()) -> bool:
    if _is_semantic_research_object(left) or _is_semantic_research_object(right):
        return unified_tensor_normal_form(left, dimension=dimension, policy=policy, semantic_rules=semantic_rules).key == unified_tensor_normal_form(right, dimension=dimension, policy=policy, semantic_rules=semantic_rules).key
    if not _is_indexed(left) and not _is_indexed(right):
        left_can = canonical_tensor_expression(left, dimension=dimension).expr
        right_can = canonical_tensor_expression(right, dimension=dimension).expr
        try:
            from .abstract_tensor import _free_indices  # type: ignore
            if not _free_indices(left_can) and not _free_indices(right_can):
                if left_can == right_can:
                    return True
                left_sig = _abstract_contracted_product_signature(left_can)
                right_sig = _abstract_contracted_product_signature(right_can)
                if left_sig is not None and left_sig == right_sig:
                    return True
        except Exception:
            pass
        left_nf = unified_tensor_normal_form(left_can, dimension=dimension, policy=policy, semantic_rules=semantic_rules)
        right_nf = unified_tensor_normal_form(right_can, dimension=dimension, policy=policy, semantic_rules=semantic_rules)
        return left_nf.key == right_nf.key
    return unified_tensor_normal_form(left, dimension=dimension, policy=policy, semantic_rules=semantic_rules).key == unified_tensor_normal_form(right, dimension=dimension, policy=policy, semantic_rules=semantic_rules).key


def unified_reduce_with_trace(obj: Any, *, dimension: int | sp.Expr | None = None, policy: RewritePolicy | None = None, semantic_rules: tuple[SemanticRewriteRule, ...] = tuple()) -> tuple[Any, ReductionTrace]:
    steps: list[ProofTraceStep] = []
    if semantic_rules:
        rewritten, semantic_report = semantic_rewrite(obj, semantic_rules, dimension=dimension)
        if semantic_report.steps:
            steps.append(ProofTraceStep("semantic_rewrite", obj, rewritten, {"passes": semantic_report.passes, "rules": tuple(step.rule for step in semantic_report.steps)}))
        obj = rewritten
    if _is_indexed(obj):
        chosen = policy if policy is not None else select_rewrite_policy(layer="indexed")
        reduced_by_policy, policy_trace = execute_rewrite_policy(obj, chosen, with_trace=True)
        reduced, report = tensor_reduce_indexed_staged(reduced_by_policy, with_report=True)
        steps.append(ProofTraceStep("rewrite_policy", obj, reduced_by_policy, {"policy": chosen.name, "families": chosen.families, "rewrite_steps": len(policy_trace.steps)}))
        steps.extend(ProofTraceStep(step.name, step.before, step.after, {"changed": step.changed}) for step in report.executed_steps)
        return reduced, ReductionTrace("indexed", obj, reduced, tuple(steps))
    chosen = policy if policy is not None else select_rewrite_policy(layer="abstract")
    reduced_by_policy, policy_trace = execute_rewrite_policy(obj, chosen, with_trace=True, dimension=dimension)
    steps.append(ProofTraceStep("rewrite_policy", obj, reduced_by_policy, {"policy": chosen.name, "families": chosen.families, "rewrite_steps": len(policy_trace.steps)}))
    nf = canonical_tensor_normal_form(reduced_by_policy, dimension=dimension)
    steps.append(ProofTraceStep("normal_form", reduced_by_policy, nf.expr, {"free_indices": nf.free_indices, "dummy_indices": nf.dummy_indices, "contraction_pairs": nf.contraction_pairs}))
    return nf.expr, ReductionTrace("abstract", obj, nf.expr, tuple(steps))


def bridge_and_normalize(obj: Any, *, tensor_registry: Mapping[str, object] | None = None, bundle_name: str | None = None, dimension: int | sp.Expr | None = None, policy: RewritePolicy | None = None, semantic_rules: tuple[SemanticRewriteRule, ...] = tuple()):
    abstract_obj, bridge_report = bridge_tensor_expression(obj, target='abstract', bundle_name=bundle_name, with_report=True)
    nf = unified_tensor_normal_form(abstract_obj.expr, dimension=dimension, policy=policy, semantic_rules=semantic_rules)
    return nf, bridge_report
