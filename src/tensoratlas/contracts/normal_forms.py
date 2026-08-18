from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..canonical_keys import structural_key


@dataclass(frozen=True)
class NormalFormContractResult:
    original_key: Any
    normalized_key: Any
    renormalized_key: Any
    idempotent: bool
    stable_under_equivalence: bool


def check_normal_form_contract(expr: Any, normalize: Callable[[Any], Any], equivalent: Callable[[Any], Any] | None = None) -> NormalFormContractResult:
    normalized = normalize(expr)
    renormalized = normalize(normalized)
    equivalent_expr = equivalent(expr) if equivalent is not None else expr
    return NormalFormContractResult(
        original_key=structural_key(expr),
        normalized_key=structural_key(normalized),
        renormalized_key=structural_key(renormalized),
        idempotent=structural_key(normalized) == structural_key(renormalized),
        stable_under_equivalence=structural_key(normalize(equivalent_expr)) == structural_key(normalized),
    )
