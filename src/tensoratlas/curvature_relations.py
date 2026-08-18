
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
import sympy as sp

from .conflict_priority_geometry_engine import PriorityRewriteRule, conflict_aware_priority_reduce
from .semantic_ir import TensorExpr, ir_node, normalize_tensor_expr, curvature_ir, canonical_ir_key


@dataclass(frozen=True)
class CurvatureSymbol:
    family: str
    rank: int
    dimension: int
    name: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CurvatureExpr:
    op: str
    args: tuple[Any, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CurvatureOrientationPolicy:
    name: str
    description: str
    preferred_targets: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutableCurvatureRewriteReport:
    original: Any
    reduced_terms: tuple[tuple[sp.Expr, Any], ...]
    applied_rules: tuple[str, ...]
    orientation_policy: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


DEFAULT_CURVATURE_ORIENTATION_POLICY = CurvatureOrientationPolicy(
    name="toward_lower_rank_and_decomposed_forms",
    description="Prefer rewrites from higher-rank curvature objects toward Ricci/scalar/Einstein/Weyl decomposed targets to improve termination.",
    preferred_targets=("Ricci", "ScalarCurvature", "Einstein", "WeylDecomposed"),
)


def Riemann(dimension: int, name: str = "Riemann") -> CurvatureSymbol:
    return CurvatureSymbol("Riemann", 4, dimension, name)


def Ricci(dimension: int, name: str = "Ricci") -> CurvatureSymbol:
    return CurvatureSymbol("Ricci", 2, dimension, name)


def ScalarCurvature(dimension: int, name: str = "ScalarCurvature") -> CurvatureSymbol:
    return CurvatureSymbol("ScalarCurvature", 0, dimension, name)


def Weyl(dimension: int, name: str = "Weyl") -> CurvatureSymbol:
    return CurvatureSymbol("Weyl", 4, dimension, name)


def Einstein(dimension: int, name: str = "Einstein") -> CurvatureSymbol:
    return CurvatureSymbol("Einstein", 2, dimension, name)


def Metric(dimension: int, name: str = "g") -> CurvatureSymbol:
    return CurvatureSymbol("Metric", 2, dimension, name)


def _coeff(x: Any) -> sp.Expr:
    return sp.sympify(x)


def _weighted_terms(expr_or_terms: Any):
    if isinstance(expr_or_terms, Sequence) and not isinstance(expr_or_terms, (str, bytes)):
        out = []
        for item in expr_or_terms:
            if isinstance(item, tuple) and len(item) == 2:
                out.append((_coeff(item[0]), item[1]))
            else:
                out.append((sp.Integer(1), item))
        return out
    return [(sp.Integer(1), expr_or_terms)]


def _term_key(term: Any):
    if isinstance(term, CurvatureSymbol):
        return ("CurvatureSymbol", term.family, term.rank, term.dimension, term.name)
    if isinstance(term, CurvatureExpr):
        return ("CurvatureExpr", term.op, tuple(_term_key(a) for a in term.args))
    try:
        return ("sympy", sp.srepr(sp.sympify(term)))
    except Exception:
        return ("repr", repr(term))


def _normalize(weighted_terms):
    groups = {}
    for c, t in weighted_terms:
        groups.setdefault(_term_key(t), []).append((_coeff(c), t))
    out = []
    for items in groups.values():
        coeff = sp.simplify(sum(c for c, _ in items))
        if coeff != 0:
            out.append((coeff, items[0][1]))
    out.sort(key=lambda x: (repr(_term_key(x[1])), sp.srepr(_coeff(x[0]))))
    return out


def curvature_contraction(source: CurvatureSymbol, target_family: str) -> CurvatureExpr:
    return CurvatureExpr("contract", (source, target_family), {})


def curvature_decomposition(source: CurvatureSymbol, target_family: str) -> CurvatureExpr:
    return CurvatureExpr("decompose", (source, target_family), {})


def curvature_linear_combo(*args: Any) -> CurvatureExpr:
    return CurvatureExpr("linear_combo", tuple(args), {})


def _expand_riemann_to_ricci_scalar_weyl(term: Any):
    if not isinstance(term, CurvatureSymbol) or term.family != "Riemann":
        return None
    dim = term.dimension
    return curvature_linear_combo(
        Weyl(dim, "Weyl"),
        curvature_decomposition(term, "RicciPart"),
        curvature_decomposition(term, "ScalarPart"),
    )


def _expand_ricci_to_einstein_scalar(term: Any):
    if not isinstance(term, CurvatureSymbol) or term.family != "Ricci":
        return None
    dim = term.dimension
    return curvature_linear_combo(
        Einstein(dim, "Einstein"),
        curvature_decomposition(term, "MetricScalarPart"),
    )


def _contract_riemann_to_ricci(term: Any):
    if not isinstance(term, CurvatureSymbol) or term.family != "Riemann":
        return None
    return curvature_contraction(term, "Ricci")


def _contract_ricci_to_scalar(term: Any):
    if not isinstance(term, CurvatureSymbol) or term.family != "Ricci":
        return None
    return curvature_contraction(term, "ScalarCurvature")


def _rewrite_terms(weighted_terms, matcher, replacement_builder):
    out = []
    changed = False
    for c, t in weighted_terms:
        if matcher(t):
            repl = replacement_builder(t)
            if repl is not None:
                out.append((c, repl))
                changed = True
            else:
                out.append((c, t))
        else:
            out.append((c, t))
    return _normalize(out), changed


def _match_riemann(term: Any) -> bool:
    return isinstance(term, CurvatureSymbol) and term.family == "Riemann"


def _match_ricci(term: Any) -> bool:
    return isinstance(term, CurvatureSymbol) and term.family == "Ricci"


def _apply_riemann_decomposition(terms):
    return _rewrite_terms(terms, _match_riemann, _expand_riemann_to_ricci_scalar_weyl)


def _apply_riemann_contraction_to_ricci(terms):
    return _rewrite_terms(terms, _match_riemann, _contract_riemann_to_ricci)


def _apply_ricci_contraction_to_scalar(terms):
    return _rewrite_terms(terms, _match_ricci, _contract_ricci_to_scalar)


def _apply_ricci_to_einstein(terms):
    return _rewrite_terms(terms, _match_ricci, _expand_ricci_to_einstein_scalar)


def _build_curvature_executable_rules() -> tuple[PriorityRewriteRule, ...]:
    return (
        PriorityRewriteRule(
            "rewrite_riemann_to_ricci_contraction",
            "curvature_executable",
            140,
            ("curvature", "contract_riemann", 1),
            _apply_riemann_contraction_to_ricci,
            {"orientation": "lower_rank", "terminating": True},
        ),
        PriorityRewriteRule(
            "rewrite_ricci_to_scalar_contraction",
            "curvature_executable",
            130,
            ("curvature", "contract_ricci", 2),
            _apply_ricci_contraction_to_scalar,
            {"orientation": "lower_rank", "terminating": True},
        ),
        PriorityRewriteRule(
            "rewrite_ricci_to_einstein_decomposition",
            "curvature_executable",
            120,
            ("curvature", "einstein", 3),
            _apply_ricci_to_einstein,
            {"orientation": "decompose", "terminating": True},
        ),
        PriorityRewriteRule(
            "rewrite_riemann_to_weyl_ricci_scalar_decomposition",
            "curvature_executable",
            110,
            ("curvature", "weyl_decompose", 4),
            _apply_riemann_decomposition,
            {"orientation": "decompose", "terminating": True},
        ),
    )


def get_curvature_executable_rules() -> tuple[PriorityRewriteRule, ...]:
    cached = globals().get("_CURVATURE_EXECUTABLE_RULES_CACHE")
    if cached is None:
        cached = _build_curvature_executable_rules()
        globals()["_CURVATURE_EXECUTABLE_RULES_CACHE"] = cached
    return cached


def __getattr__(name: str) -> Any:
    if name == "CURVATURE_EXECUTABLE_RULES":
        return get_curvature_executable_rules()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def executable_curvature_reduce(expr_or_terms: Any, *, policy: CurvatureOrientationPolicy = DEFAULT_CURVATURE_ORIENTATION_POLICY) -> ExecutableCurvatureRewriteReport:
    weighted = _normalize(_weighted_terms(expr_or_terms))
    report = conflict_aware_priority_reduce(weighted, rules=get_curvature_executable_rules())
    return ExecutableCurvatureRewriteReport(
        original=expr_or_terms,
        reduced_terms=report.reduced_terms,
        applied_rules=report.applied_rules,
        orientation_policy=policy.name,
        metadata={
            "blocked_rules": report.blocked_rules,
            "iterations": report.iterations,
            "preferred_targets": policy.preferred_targets,
        },
    )


def curvature_object_to_ir(obj: Any) -> TensorExpr:
    if isinstance(obj, TensorExpr):
        return obj
    if isinstance(obj, CurvatureSymbol):
        return ir_node(
            "curvature_symbol",
            payload=obj.name,
            family=obj.family,
            rank=obj.rank,
            dimension=obj.dimension,
            tensor_expr_kind="curvature",
            **dict(obj.metadata),
        )
    if isinstance(obj, CurvatureExpr):
        if obj.op == "contract":
            source, target = obj.args
            return ir_node(
                "contract",
                curvature_object_to_ir(source),
                target_family=target,
                family="CurvatureContraction",
                **dict(obj.metadata),
            )
        if obj.op == "decompose":
            source, target = obj.args
            return ir_node(
                "curvature_decomposition",
                curvature_object_to_ir(source),
                target_family=target,
                family="CurvatureDecomposition",
                **dict(obj.metadata),
            )
        if obj.op == "linear_combo":
            return ir_node(
                "curvature_linear_combo",
                *(curvature_object_to_ir(arg) for arg in obj.args),
                **dict(obj.metadata),
            )
        return ir_node(
            f"curvature_expr:{obj.op}",
            *(curvature_object_to_ir(arg) for arg in obj.args),
            **dict(obj.metadata),
        )
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        terms = []
        for item in obj:
            if isinstance(item, tuple) and len(item) == 2:
                terms.append(ir_node("weighted_term", curvature_object_to_ir(item[1]), coefficient=_coeff(item[0])))
            else:
                terms.append(curvature_object_to_ir(item))
        return normalize_tensor_expr(ir_node("curvature_linear_combo", *terms))
    return ir_node("scalar", payload=sp.sympify(obj))


def curvature_reduce_to_ir(expr_or_terms: Any, *, policy: CurvatureOrientationPolicy = DEFAULT_CURVATURE_ORIENTATION_POLICY) -> TensorExpr:
    report = executable_curvature_reduce(expr_or_terms, policy=policy)
    return curvature_object_to_ir(tuple(report.reduced_terms))
