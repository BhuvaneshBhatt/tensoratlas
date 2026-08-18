from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..canonical_keys import structural_key


@dataclass(frozen=True)
class BridgeContractResult:
    forward_key: Any
    roundtrip_key: Any
    preserves_canonical_key: bool


def check_bridge_contract(obj: Any, forward: Callable[[Any], Any], backward: Callable[[Any], Any], canonical_key: Callable[[Any], Any] | None = None) -> BridgeContractResult:
    canonical = canonical_key or structural_key
    forward_obj = forward(obj)
    roundtrip_obj = backward(forward_obj)
    return BridgeContractResult(
        forward_key=canonical(forward_obj),
        roundtrip_key=canonical(roundtrip_obj),
        preserves_canonical_key=canonical(obj) == canonical(roundtrip_obj),
    )
