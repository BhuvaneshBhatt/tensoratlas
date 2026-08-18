from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Mapping

from .cache_utils import BoundedCache
from .rewrite_families import RewritePolicy, execute_rewrite_policy, select_rewrite_policy, apply_rewrite_families
from .unified_reduction import unified_reduce_with_trace
from .canonical_keys import canonical_expr_fingerprint


@dataclass(frozen=True)
class PerformanceSample:
    name: str
    duration_seconds: float
    cache_key: object | None = None
    cache_hit: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PerformanceReport:
    operation: str
    result: Any
    samples: tuple[PerformanceSample, ...]
    cache_stats: Mapping[str, int] = field(default_factory=dict)
    family_samples: tuple[PerformanceSample, ...] = tuple()


class ReducerCacheRegistry:
    def __init__(self, maxsize: int = 256):
        self.unified_cache: BoundedCache[object, Any] = BoundedCache(maxsize=maxsize)
        self.policy_cache: BoundedCache[object, Any] = BoundedCache(maxsize=maxsize)

    def clear(self) -> None:
        self.unified_cache.clear()
        self.policy_cache.clear()

    def stats(self) -> dict[str, Mapping[str, int]]:
        return {
            "unified": self.unified_cache.stats(),
            "policy": self.policy_cache.stats(),
        }


_DEFAULT_REGISTRY = ReducerCacheRegistry()


def get_reducer_cache_registry() -> ReducerCacheRegistry:
    return _DEFAULT_REGISTRY


def clear_reducer_caches() -> None:
    _DEFAULT_REGISTRY.clear()


def _policy_cache_key(expr: Any, policy: RewritePolicy):
    return canonical_expr_fingerprint(expr, layer=policy.layer, policy=f'{policy.operator}|{policy.mode}|{policy.families}')


def _unified_cache_key(expr: Any, dimension: Any | None):
    return canonical_expr_fingerprint(expr, dimension=dimension, layer='unified')


def timed_execute_rewrite_policy(expr: Any, policy: RewritePolicy | None = None, *, layer: str | None = None, operator: str | None = None, mode: str | None = None, use_cache: bool = True, with_trace: bool = True, **kwargs) -> tuple[Any, PerformanceReport]:
    chosen = policy
    if chosen is None:
        if layer is None:
            raise ValueError("layer is required when policy is not provided.")
        chosen = select_rewrite_policy(layer=layer, operator=operator, mode=mode)
    key = _policy_cache_key(expr, chosen)
    samples = []
    family_samples = []
    if use_cache:
        cached = _DEFAULT_REGISTRY.policy_cache.get(key)
        if cached is not None:
            samples.append(PerformanceSample(name="policy", duration_seconds=0.0, cache_key=key, cache_hit=True))
            return cached, PerformanceReport(operation="execute_rewrite_policy", result=cached, samples=tuple(samples), cache_stats=_DEFAULT_REGISTRY.policy_cache.stats(), family_samples=tuple())
    current = expr
    start = perf_counter()
    for family in chosen.families:
        fam_kwargs = dict(kwargs)
        fam_kwargs.update(dict(chosen.family_overrides.get(family, {})))
        fam_start = perf_counter()
        current = apply_rewrite_families(current, families=(family,), layer=chosen.layer, mode=chosen.mode if mode is None else mode, with_diagnostics=False, max_iterations=chosen.fixed_point_limit, **fam_kwargs)
        fam_duration = perf_counter() - fam_start
        family_samples.append(PerformanceSample(name=family, duration_seconds=fam_duration, metadata={"policy": chosen.name, "layer": chosen.layer}))
    result = execute_rewrite_policy(expr, chosen, with_trace=with_trace, **kwargs)
    duration = perf_counter() - start
    samples.append(PerformanceSample(name="policy", duration_seconds=duration, cache_key=key, cache_hit=False, metadata={"policy": chosen.name}))
    if use_cache:
        _DEFAULT_REGISTRY.policy_cache[key] = result
    return result, PerformanceReport(operation="execute_rewrite_policy", result=result, samples=tuple(samples), cache_stats=_DEFAULT_REGISTRY.policy_cache.stats(), family_samples=tuple(family_samples))


def timed_unified_reduce(expr: Any, *, dimension: Any | None = None, use_cache: bool = True) -> tuple[Any, PerformanceReport]:
    key = _unified_cache_key(expr, dimension)
    samples = []
    if use_cache:
        cached = _DEFAULT_REGISTRY.unified_cache.get(key)
        if cached is not None:
            samples.append(PerformanceSample(name="unified_reduce", duration_seconds=0.0, cache_key=key, cache_hit=True))
            return cached, PerformanceReport(operation="unified_reduce", result=cached, samples=tuple(samples), cache_stats=_DEFAULT_REGISTRY.unified_cache.stats(), family_samples=tuple())
    start = perf_counter()
    result = unified_reduce_with_trace(expr, dimension=dimension)
    duration = perf_counter() - start
    samples.append(PerformanceSample(name="unified_reduce", duration_seconds=duration, cache_key=key, cache_hit=False, metadata={"dimension": dimension}))
    if use_cache:
        _DEFAULT_REGISTRY.unified_cache[key] = result
    return result, PerformanceReport(operation="unified_reduce", result=result, samples=tuple(samples), cache_stats=_DEFAULT_REGISTRY.unified_cache.stats(), family_samples=tuple())


__all__ = [
    "PerformanceSample",
    "PerformanceReport",
    "ReducerCacheRegistry",
    "get_reducer_cache_registry",
    "clear_reducer_caches",
    "timed_execute_rewrite_policy",
    "timed_unified_reduce",
]
