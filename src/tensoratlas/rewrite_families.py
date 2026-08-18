from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence, Mapping

import sympy as sp

from .abstract_tensor import (
    structural_simplify,
    metric_simplify,
    multiterm_simplify,
    invariant_simplify,
    simplify_abstract,
)
from .indexed_api import canonicalize_indexed_expression, normalize_indexed_expression


@dataclass(frozen=True)
class RewriteFamily:
    name: str
    layer: str
    description: str
    priority: int = 100
    fixed_point_policy: str = "single_pass"
    diagnostics: bool = False
    mode_group: str | None = None
    modes: tuple[str, ...] = tuple()




@dataclass(frozen=True)
class RewriteContext:
    layer: str
    operator: str | None = None
    mode: str | None = None
    assumptions: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RewritePolicy:
    name: str
    layer: str
    families: tuple[str, ...]
    mode: str | None = None
    operator: str | None = None
    fixed_point_limit: int = 8
    diagnostics: bool = False
    exclusivity_mode: str = "resolve"
    family_overrides: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    parent_policies: tuple[str, ...] = tuple()


@dataclass(frozen=True)
class RewriteExecutionStep:
    family: str
    before: Any
    after: Any
    changed: bool
    iterations: int = 1


@dataclass(frozen=True)
class RewriteExecutionTrace:
    policy: RewritePolicy
    context: RewriteContext
    final: Any
    steps: tuple[RewriteExecutionStep, ...]
    diagnostics: RewriteDiagnostics | None = None


@dataclass(frozen=True)
class RewriteDiagnostics:
    layer: str
    requested_families: tuple[str, ...]
    applied_families: tuple[str, ...]
    changed_families: tuple[str, ...]
    iterations: Mapping[str, int] = field(default_factory=dict)
    mutually_exclusive_resolutions: tuple[tuple[str, str], ...] = tuple()


def _abstract_linearity(expr, **kwargs):
    return structural_simplify(sp.expand(as_expr(expr)))


def _abstract_metric(expr, **kwargs):
    return metric_simplify(expr, max_passes=int(kwargs.get("max_passes", 8)))


def _abstract_multiterm(expr, **kwargs):
    return multiterm_simplify(expr, dimension=kwargs.get("dimension"))


def _abstract_invariant(expr, **kwargs):
    return invariant_simplify(expr, dimension=kwargs.get("dimension"))


def _abstract_all(expr, **kwargs):
    return simplify_abstract(expr, mode="all", dimension=kwargs.get("dimension"))


def _indexed_normalize(expr, **kwargs):
    return normalize_indexed_expression(expr, config=kwargs.get("config"))


def _indexed_canonicalize(expr, **kwargs):
    return canonicalize_indexed_expression(expr, config=kwargs.get("config"))


def _indexed_all(expr, **kwargs):
    return canonicalize_indexed_expression(expr, config=kwargs.get("config"))


_ABSTRACT_FAMILIES: dict[str, tuple[RewriteFamily, Callable[..., Any]]] = {
    "linearity": (RewriteFamily("linearity", "abstract", "Abstract rewrite family: linearity.", priority=10, fixed_point_policy="single_pass", diagnostics=True), _abstract_linearity),
    "metric_delta": (RewriteFamily("metric_delta", "abstract", "Abstract rewrite family: metric/delta reductions.", priority=20, fixed_point_policy="fixed_point", diagnostics=True, mode_group="canonicalization", modes=("fast", "full")), _abstract_metric),
    "multiterm": (RewriteFamily("multiterm", "abstract", "Abstract rewrite family: multi-term identities.", priority=30, fixed_point_policy="single_pass", diagnostics=True, mode_group="canonicalization", modes=("full",)), _abstract_multiterm),
    "invariant": (RewriteFamily("invariant", "abstract", "Abstract rewrite family: invariant reductions.", priority=40, fixed_point_policy="single_pass", diagnostics=True, mode_group="canonicalization", modes=("full",)), _abstract_invariant),
    "all": (RewriteFamily("all", "abstract", "Abstract rewrite family: full staged simplification.", priority=90, fixed_point_policy="single_pass", diagnostics=True, mode_group="canonicalization", modes=("full",)), _abstract_all),
}

_INDEXED_FAMILIES: dict[str, tuple[RewriteFamily, Callable[..., Any]]] = {
    "normalize": (RewriteFamily("normalize", "indexed", "Indexed rewrite family: normalization.", priority=10, fixed_point_policy="single_pass", diagnostics=True, mode_group="indexed_reduce", modes=("fast", "full")), _indexed_normalize),
    "canonicalize": (RewriteFamily("canonicalize", "indexed", "Indexed rewrite family: canonicalization.", priority=20, fixed_point_policy="fixed_point", diagnostics=True, mode_group="indexed_reduce", modes=("full",)), _indexed_canonicalize),
    "all": (RewriteFamily("all", "indexed", "Indexed rewrite family: full canonicalization.", priority=90, fixed_point_policy="single_pass", diagnostics=True, mode_group="indexed_reduce", modes=("full",)), _indexed_all),
}


def as_expr(expr: Any) -> Any:
    return getattr(expr, "expr", expr)


def list_rewrite_families(layer: str | None = None) -> tuple[RewriteFamily, ...]:
    items: list[RewriteFamily] = []
    if layer in (None, "abstract"):
        items.extend(meta for meta, _ in _ABSTRACT_FAMILIES.values())
    if layer in (None, "indexed"):
        items.extend(meta for meta, _ in _INDEXED_FAMILIES.values())
    if layer not in (None, "abstract", "indexed"):
        raise ValueError(f"Unsupported rewrite family layer: {layer!r}")
    return tuple(sorted(items, key=lambda item: (item.layer, item.priority, item.name)))


def _resolve_registry(layer: str):
    registry = _ABSTRACT_FAMILIES if layer == "abstract" else _INDEXED_FAMILIES if layer == "indexed" else None
    if registry is None:
        raise ValueError(f"Unsupported rewrite family layer: {layer!r}")
    return registry



def _resolve_selected_families(chosen: Sequence[str], metas: Sequence[tuple[RewriteFamily, Callable[..., Any]]], *, explicit: bool) -> tuple[list[tuple[RewriteFamily, Callable[..., Any]]], tuple[tuple[str, str], ...]]:
    grouped: dict[tuple[str | None, str], list[tuple[int, RewriteFamily, Callable[..., Any]]]] = {}
    for position, (meta, fn) in enumerate(metas):
        key = (meta.mode_group, meta.mode_group or meta.name)
        grouped.setdefault(key, []).append((position, meta, fn))
    selected: list[tuple[RewriteFamily, Callable[..., Any]]] = []
    resolutions: list[tuple[str, str]] = []
    for (group_name, _), items in grouped.items():
        if group_name is None:
            selected.extend((meta, fn) for _, meta, fn in items)
            continue
        if explicit:
            chosen_item = max(items, key=lambda item: item[0])
        else:
            chosen_item = min(items, key=lambda item: (item[1].priority, item[1].name))
        _, kept_meta, kept_fn = chosen_item
        selected.append((kept_meta, kept_fn))
        for _, skipped_meta, _ in items:
            if skipped_meta.name != kept_meta.name:
                resolutions.append((skipped_meta.name, kept_meta.name))
    selected.sort(key=lambda pair: (pair[0].priority, pair[0].name))
    return selected, tuple(resolutions)


def _apply_family(meta: RewriteFamily, fn: Callable[..., Any], current: Any, *, max_iterations: int, kwargs: Mapping[str, Any]) -> tuple[Any, int]:
    iters = 0
    if meta.fixed_point_policy == "fixed_point":
        previous = current
        while iters < max_iterations:
            nxt = fn(previous, **kwargs)
            iters += 1
            if nxt == previous:
                return nxt, iters
            previous = nxt
        return previous, iters
    return fn(current, **kwargs), 1

def apply_rewrite_families(expr: Any, families: Sequence[str] | None = None, *, layer: str = "abstract", mode: str | None = None, with_diagnostics: bool = False, max_iterations: int = 8, **kwargs) -> Any:
    registry = _resolve_registry(layer)
    chosen = tuple(registry) if families is None else tuple(families)
    metas = []
    for name in chosen:
        try:
            meta, fn = registry[name]
        except KeyError as exc:
            raise ValueError(f"Unknown rewrite family {name!r} for layer {layer!r}.") from exc
        if mode is not None and meta.modes and mode not in meta.modes:
            continue
        metas.append((meta, fn))
    selected, resolutions = _resolve_selected_families(chosen, metas, explicit=families is not None)
    current = expr
    changed = []
    iterations = {}
    applied = []
    for meta, fn in selected:
        applied.append(meta.name)
        before = current
        current, iters = _apply_family(meta, fn, current, max_iterations=max_iterations, kwargs=kwargs)
        iterations[meta.name] = iters
        if current != before:
            changed.append(meta.name)
    if with_diagnostics:
        return current, RewriteDiagnostics(layer=layer, requested_families=tuple(chosen), applied_families=tuple(applied), changed_families=tuple(changed), iterations=iterations, mutually_exclusive_resolutions=resolutions)
    return current


def abstract_expand(expr: Any, families: Sequence[str] | None = None, **kwargs) -> Any:
    return apply_rewrite_families(expr, families=families, layer="abstract", **kwargs)


def indexed_expand(expr: Any, families: Sequence[str] | None = None, **kwargs) -> Any:
    return apply_rewrite_families(expr, families=families, layer="indexed", **kwargs)


def abstract_reduce(expr: Any, families: Sequence[str] | None = None, **kwargs) -> Any:
    return apply_rewrite_families(expr, families=families, layer="abstract", **kwargs)


def indexed_reduce(expr: Any, families: Sequence[str] | None = None, **kwargs) -> Any:
    return apply_rewrite_families(expr, families=families, layer="indexed", **kwargs)


_DEFAULT_POLICIES: dict[tuple[str, str | None], RewritePolicy] = {
    ("abstract", None): RewritePolicy("abstract_default", "abstract", ("linearity", "metric_delta", "multiterm", "invariant"), mode="full", diagnostics=True),
    ("abstract", "tensor_contract"): RewritePolicy("abstract_contract", "abstract", ("metric_delta", "multiterm"), mode="full", operator="tensor_contract", diagnostics=True),
    ("indexed", None): RewritePolicy("indexed_default", "indexed", ("normalize", "canonicalize"), mode="full", diagnostics=True),
    ("indexed", "tensor_contract"): RewritePolicy("indexed_contract", "indexed", ("normalize", "canonicalize"), mode="full", operator="tensor_contract", diagnostics=True),
}


def rewrite_context(*, layer: str, operator: str | None = None, mode: str | None = None, assumptions: Any | None = None, metadata: Mapping[str, Any] | None = None) -> RewriteContext:
    return RewriteContext(layer=layer, operator=operator, mode=mode, assumptions=assumptions, metadata={} if metadata is None else dict(metadata))


def rewrite_policy(name: str, *, layer: str, families: Sequence[str], mode: str | None = None, operator: str | None = None, fixed_point_limit: int = 8, diagnostics: bool = False, exclusivity_mode: str = "resolve", family_overrides: Mapping[str, Mapping[str, Any]] | None = None, parent_policies: Sequence[str] = ()) -> RewritePolicy:
    return RewritePolicy(name=name, layer=layer, families=tuple(families), mode=mode, operator=operator, fixed_point_limit=int(fixed_point_limit), diagnostics=bool(diagnostics), exclusivity_mode=exclusivity_mode, family_overrides={} if family_overrides is None else {k: dict(v) for k, v in family_overrides.items()}, parent_policies=tuple(parent_policies))


def list_rewrite_policies(layer: str | None = None, *, operator: str | None = None) -> tuple[RewritePolicy, ...]:
    items = []
    for (policy_layer, policy_operator), policy in _DEFAULT_POLICIES.items():
        if layer is not None and policy_layer != layer:
            continue
        if operator is not None and policy_operator != operator:
            continue
        items.append(policy)
    return tuple(sorted(items, key=lambda item: (item.layer, item.operator or "", item.name)))


def select_rewrite_policy(*, layer: str, operator: str | None = None, mode: str | None = None) -> RewritePolicy:
    policy = _DEFAULT_POLICIES.get((layer, operator)) or _DEFAULT_POLICIES.get((layer, None))
    if policy is None:
        raise ValueError(f"No rewrite policy available for layer={layer!r}, operator={operator!r}.")
    if mode is None or policy.mode == mode:
        return policy
    return RewritePolicy(name=policy.name, layer=policy.layer, families=policy.families, mode=mode, operator=policy.operator, fixed_point_limit=policy.fixed_point_limit, diagnostics=policy.diagnostics, exclusivity_mode=policy.exclusivity_mode, family_overrides=policy.family_overrides, parent_policies=policy.parent_policies)




def compose_rewrite_policies(name: str, *policies: RewritePolicy, mode: str | None = None, operator: str | None = None, family_overrides: Mapping[str, Mapping[str, Any]] | None = None) -> RewritePolicy:
    if not policies:
        raise ValueError("At least one policy is required for composition.")
    layer = policies[0].layer
    if any(p.layer != layer for p in policies):
        raise ValueError("All composed policies must target the same layer.")
    families = []
    seen = set()
    parent_names = []
    merged_overrides: dict[str, dict[str, Any]] = {}
    for policy in policies:
        parent_names.append(policy.name)
        for fam in policy.families:
            if fam not in seen:
                seen.add(fam)
                families.append(fam)
        for fam, override in policy.family_overrides.items():
            merged_overrides.setdefault(fam, {}).update(dict(override))
    if family_overrides is not None:
        for fam, override in family_overrides.items():
            merged_overrides.setdefault(fam, {}).update(dict(override))
    return RewritePolicy(
        name=name,
        layer=layer,
        families=tuple(families),
        mode=mode if mode is not None else policies[-1].mode,
        operator=operator if operator is not None else policies[-1].operator,
        fixed_point_limit=max(p.fixed_point_limit for p in policies),
        diagnostics=any(p.diagnostics for p in policies),
        exclusivity_mode=policies[-1].exclusivity_mode,
        family_overrides=merged_overrides,
        parent_policies=tuple(parent_names),
    )


def override_rewrite_policy_families(policy: RewritePolicy, *, append: Sequence[str] = (), remove: Sequence[str] = (), replace: Sequence[str] | None = None, family_overrides: Mapping[str, Mapping[str, Any]] | None = None, name: str | None = None) -> RewritePolicy:
    if replace is not None:
        families = list(replace)
    else:
        families = [fam for fam in policy.families if fam not in set(remove)]
        for fam in append:
            if fam not in families:
                families.append(fam)
    merged = {k: dict(v) for k, v in policy.family_overrides.items()}
    if family_overrides is not None:
        for fam, override in family_overrides.items():
            merged.setdefault(fam, {}).update(dict(override))
    return RewritePolicy(name=policy.name if name is None else name, layer=policy.layer, families=tuple(families), mode=policy.mode, operator=policy.operator, fixed_point_limit=policy.fixed_point_limit, diagnostics=policy.diagnostics, exclusivity_mode=policy.exclusivity_mode, family_overrides=merged, parent_policies=policy.parent_policies or (policy.name,))


def execute_rewrite_policy(expr: Any, policy: RewritePolicy, *, context: RewriteContext | None = None, with_trace: bool = False, **kwargs) -> Any:
    ctx = rewrite_context(layer=policy.layer, operator=policy.operator, mode=policy.mode) if context is None else context
    base_kwargs = dict(kwargs)
    current = expr
    steps = []
    requested = []
    applied = []
    changed = []
    iterations = {}
    resolutions = []
    for family in policy.families:
        fam_kwargs = dict(base_kwargs)
        fam_kwargs.update(dict(policy.family_overrides.get(family, {})))
        requested.append(family)
        before = current
        current, fam_diags = apply_rewrite_families(
            current,
            families=(family,),
            layer=policy.layer,
            mode=policy.mode if ctx.mode is None else ctx.mode,
            with_diagnostics=True,
            max_iterations=policy.fixed_point_limit,
            **fam_kwargs,
        )
        applied.extend(fam_diags.applied_families)
        changed.extend(fam_diags.changed_families)
        iterations.update(fam_diags.iterations)
        resolutions.extend(fam_diags.mutually_exclusive_resolutions)
        steps.append(RewriteExecutionStep(family=family, before=before, after=current, changed=current != before, iterations=fam_diags.iterations.get(family, 1)))
    diags = RewriteDiagnostics(layer=policy.layer, requested_families=tuple(requested), applied_families=tuple(applied), changed_families=tuple(changed), iterations=iterations, mutually_exclusive_resolutions=tuple(resolutions))
    trace = RewriteExecutionTrace(policy=policy, context=ctx, final=current, steps=tuple(steps), diagnostics=diags)
    return (current, trace) if (policy.diagnostics or with_trace) else current


def apply_registered_rewrite_pipeline(expr: Any, *, layer: str, operator: str | None = None, mode: str | None = None, **kwargs) -> Any:
    policy = select_rewrite_policy(layer=layer, operator=operator, mode=mode)
    return execute_rewrite_policy(expr, policy, **kwargs)


__all__ = [
    "RewriteFamily", "RewriteContext", "RewritePolicy", "RewriteExecutionStep", "RewriteExecutionTrace", "RewriteDiagnostics",
    "list_rewrite_families", "apply_rewrite_families", "abstract_expand", "indexed_expand", "abstract_reduce", "indexed_reduce",
    "rewrite_context", "rewrite_policy", "list_rewrite_policies", "select_rewrite_policy", "compose_rewrite_policies", "override_rewrite_policy_families",
    "execute_rewrite_policy", "apply_registered_rewrite_pipeline",
]
