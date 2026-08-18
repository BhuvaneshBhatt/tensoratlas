from __future__ import annotations

from typing import Any

from .simplification_policy import cheap_simplify, normal_simplify, strong_simplify

VALID_NORMALIZATION_MODES = frozenset({"heuristic", "strict"})
VALID_SIMPLIFICATION_LEVELS = frozenset({"cheap", "normal", "strong"})


def validate_indexed_config(config: Any) -> Any:
    mode = getattr(config, "normalization_mode", "heuristic")
    level = getattr(config, "simplification_level", "normal")
    if mode not in VALID_NORMALIZATION_MODES:
        raise ValueError(f"normalization_mode must be one of {sorted(VALID_NORMALIZATION_MODES)}; got {mode!r}")
    if level not in VALID_SIMPLIFICATION_LEVELS:
        raise ValueError(f"simplification_level must be one of {sorted(VALID_SIMPLIFICATION_LEVELS)}; got {level!r}")
    return config


def resolve_indexed_config(config: Any, factory) -> Any:
    if config is None:
        config = factory()
    return validate_indexed_config(config)


def normalization_mode(config: Any) -> str:
    return validate_indexed_config(config).normalization_mode


def simplification_level(config: Any) -> str:
    return validate_indexed_config(config).simplification_level


def decision_mode(config: Any) -> str:
    return "strict" if normalization_mode(config) == "strict" else "safe"


def heuristic_enabled(config: Any) -> bool:
    return normalization_mode(config) != "strict"


def configured_simplify_expr(expr: Any, config: Any) -> Any:
    level = simplification_level(config)
    if level == "cheap":
        return cheap_simplify(expr)
    if level == "strong":
        return strong_simplify(expr)
    return normal_simplify(expr)
