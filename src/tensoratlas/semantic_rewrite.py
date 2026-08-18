from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .semantic_core import (
    SemanticNode,
    compile_semantic_node,
    materialize_semantic_node,
    normalize_semantic_node,
    semantic_node_fingerprint,
)
from .semantic_matching import semantic_equivalent_objects, semantic_match_key, indexed_tensor_orbit_specs, indexed_expression_orbit_nodes, indexed_tree_dummy_normalize, indexed_graph_equivalent


Predicate = Callable[[Any], bool]
Condition = Callable[[Mapping[str, Any]], bool]
Replacement = Callable[[Mapping[str, Any]], Any] | Any


@dataclass(frozen=True)
class SemanticVar:
    name: str
    kind: str | None = None
    predicate: Predicate | None = None


@dataclass(frozen=True)
class SemanticPattern:
    kind: str | None = None
    value: Any = None
    children: tuple[Any, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    bind: str | None = None


@dataclass(frozen=True)
class SemanticRewriteRule:
    name: str
    pattern: Any
    replacement: Replacement
    condition: Condition | None = None
    normalize_after: bool = True


@dataclass(frozen=True)
class SemanticRewriteStep:
    rule: str
    before: Any
    after: Any
    bindings: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticRewriteReport:
    original: Any
    final: Any
    steps: tuple[SemanticRewriteStep, ...] = tuple()
    passes: int = 0
    fixed_point: bool = False


@dataclass(frozen=True)
class SemanticRewriteContext:
    layer: str | None = None
    dimension: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def svar(name: str, *, kind: str | None = None, predicate: Predicate | None = None) -> SemanticVar:
    return SemanticVar(name=name, kind=kind, predicate=predicate)


def spat(kind: str | None = None, *children: Any, value: Any = None, bind: str | None = None, metadata: Mapping[str, Any] | None = None) -> SemanticPattern:
    return SemanticPattern(kind=kind, value=value, children=tuple(children), bind=bind, metadata=dict(metadata or {}))



def _semantic_node_to_pattern(node: SemanticNode) -> SemanticPattern:
    if str(getattr(node, "kind", "")).startswith("indexed"):
        node = indexed_tree_dummy_normalize(node)
    meta = {k: v for k, v in dict(node.metadata).items() if not str(k).startswith("_")}
    return SemanticPattern(
        kind=node.kind,
        value=node.value,
        children=tuple(_semantic_node_to_pattern(child) for child in node.children),
        metadata=meta,
        bind=None,
    )

def _metadata_matches(node: SemanticNode, wanted: Mapping[str, Any]) -> bool:
    for key, value in wanted.items():
        if node.metadata.get(key) != value:
            return False
    return True



def _match(pattern: Any, node: SemanticNode, env: dict[str, Any], *, _allow_orbit: bool = True) -> bool:
    if _allow_orbit and node.kind in {"indexed_tensor", "indexed_add", "indexed_tensor_product", "indexed_expr"}:
        variants = indexed_expression_orbit_nodes(node)
        for variant, sign in variants:
            trial = dict(env)
            if _match(pattern, variant, trial, _allow_orbit=False):
                trial["__orbit_sign__"] = trial.get("__orbit_sign__", 1) * int(sign)
                if isinstance(pattern, SemanticPattern) and pattern.bind is not None:
                    trial[f"{pattern.bind}__orbit_sign__"] = int(sign)
                env.clear()
                env.update(trial)
                return True
        return False

    if isinstance(pattern, SemanticVar):
        if pattern.kind is not None and node.kind != pattern.kind:
            return False
        bound = env.get(pattern.name)
        candidate = materialize_semantic_node(node)
        if pattern.predicate is not None and not pattern.predicate(candidate):
            return False
        if bound is None:
            env[pattern.name] = candidate
            return True
        return semantic_equivalent_objects(bound, candidate)

    if isinstance(pattern, SemanticPattern):
        if pattern.kind is not None and node.kind != pattern.kind:
            return False
        if pattern.value is not None and pattern.value != node.value:
            return False
        if pattern.metadata and not _metadata_matches(node, pattern.metadata):
            return False
        if pattern.bind is not None:
            env[pattern.bind] = materialize_semantic_node(node)
        if pattern.children:
            if len(pattern.children) != len(node.children):
                return False
            if node.kind in {"add", "mul", "indexed_add", "indexed_tensor_product"} or node.metadata.get("commutative", False):
                remaining = list(node.children)
                for p_child in pattern.children:
                    matched = False
                    for idx, n_child in enumerate(list(remaining)):
                        trial = dict(env)
                        if _match(p_child, n_child, trial):
                            env.clear()
                            env.update(trial)
                            remaining.pop(idx)
                            matched = True
                            break
                    if not matched:
                        return False
            else:
                for p_child, n_child in zip(pattern.children, node.children):
                    if not _match(p_child, n_child, env):
                        return False
        return True

    if isinstance(pattern, SemanticNode):
        if str(getattr(pattern, "kind", "")).startswith("indexed") or str(getattr(node, "kind", "")).startswith("indexed"):
            try:
                return indexed_graph_equivalent(pattern, node)
            except Exception:
                pass
        return _match(_semantic_node_to_pattern(pattern), node, env, _allow_orbit=_allow_orbit)

    return semantic_equivalent_objects(materialize_semantic_node(node), pattern)

def semantic_match(obj: Any, pattern: Any) -> Mapping[str, Any] | None:
    node = compile_semantic_node(obj)
    env: dict[str, Any] = {}
    return env if _match(pattern, node, env) else None


def _substitute(template: Any, env: Mapping[str, Any]) -> Any:
    if callable(template):
        return template(env)
    if isinstance(template, SemanticVar):
        return env[template.name]
    if isinstance(template, SemanticPattern):
        children = tuple(_substitute(child, env) for child in template.children)
        node = SemanticNode(kind=template.kind or "leaf", value=template.value, children=tuple(compile_semantic_node(child) for child in children), metadata=template.metadata)
        return materialize_semantic_node(node)
    return template


def _rewrite_node_once(node: SemanticNode, rules: Sequence[SemanticRewriteRule], ctx: SemanticRewriteContext) -> tuple[SemanticNode, SemanticRewriteStep | None, bool]:
    for rule in rules:
        env: dict[str, Any] = {}
        if _match(rule.pattern, node, env) and (rule.condition is None or rule.condition(env)):
            before = materialize_semantic_node(node)
            replaced_obj = _substitute(rule.replacement, env)
            replaced_node = compile_semantic_node(replaced_obj, layer=ctx.layer, dimension=ctx.dimension)
            if rule.normalize_after:
                replaced_node = normalize_semantic_node(replaced_node)
            after = materialize_semantic_node(replaced_node)
            return replaced_node, SemanticRewriteStep(rule=rule.name, before=before, after=after, bindings=dict(env)), True

    for idx, child in enumerate(node.children):
        new_child, step, changed = _rewrite_node_once(child, rules, ctx)
        if changed:
            new_children = list(node.children)
            new_children[idx] = new_child
            updated = SemanticNode(node.kind, value=node.value, children=tuple(new_children), metadata=node.metadata)
            return normalize_semantic_node(updated), step, True

    return node, None, False


def semantic_rewrite_once(obj: Any, rules: Sequence[SemanticRewriteRule], *, layer: str | None = None, dimension: Any = None, metadata: Mapping[str, Any] | None = None) -> tuple[Any, SemanticRewriteStep | None]:
    ctx = SemanticRewriteContext(layer=layer, dimension=dimension, metadata=dict(metadata or {}))
    node = compile_semantic_node(obj, layer=layer, dimension=dimension)
    new_node, step, _ = _rewrite_node_once(node, rules, ctx)
    return materialize_semantic_node(new_node), step


def semantic_rewrite(obj: Any, rules: Sequence[SemanticRewriteRule], *, layer: str | None = None, dimension: Any = None, metadata: Mapping[str, Any] | None = None, max_passes: int = 8) -> tuple[Any, SemanticRewriteReport]:
    current = obj
    steps: list[SemanticRewriteStep] = []
    passes = 0
    fixed = False
    for _ in range(max_passes):
        passes += 1
        current, step = semantic_rewrite_once(current, rules, layer=layer, dimension=dimension, metadata=metadata)
        if step is None:
            fixed = True
            break
        steps.append(step)
    return current, SemanticRewriteReport(original=obj, final=current, steps=tuple(steps), passes=passes, fixed_point=fixed)





def semantic_operator_rules() -> tuple[SemanticRewriteRule, ...]:
    from .semantic_ops import evaluate_semantic_operator

    def _eval(env):
        return evaluate_semantic_operator(env['op'])

    return (
        SemanticRewriteRule('eval_hodge', spat('hodge', bind='op'), _eval),
        SemanticRewriteRule('eval_codifferential', spat('codifferential', bind='op'), _eval),
        SemanticRewriteRule('eval_interior', spat('interior', bind='op'), _eval),
        SemanticRewriteRule('eval_lie', spat('lie', bind='op'), _eval),
        SemanticRewriteRule('eval_gamma_string', spat('gamma_string', bind='op'), _eval),
    )

__all__ = [
    "SemanticVar",
    "SemanticPattern",
    "SemanticRewriteRule",
    "SemanticRewriteStep",
    "SemanticRewriteReport",
    "SemanticRewriteContext",
    "svar",
    "spat",
    "semantic_match",
    "semantic_rewrite_once",
    "semantic_rewrite",
    "semantic_operator_rules",
]

# Keep the historical top-level export `semantic_rewrite` usable even when the
# submodule itself has been imported first and Python has assigned the module to
# tensoratlas.semantic_rewrite.  This makes the module object callable and
# delegates to the function of the same name.
try:  # pragma: no cover - import-system compatibility shim
    import sys as _sys
    import types as _types

    class _CallableSemanticRewriteModule(_types.ModuleType):
        def __call__(self, *args, **kwargs):
            return semantic_rewrite(*args, **kwargs)

    _sys.modules[__name__].__class__ = _CallableSemanticRewriteModule
except Exception:
    pass
