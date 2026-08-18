from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ProvenanceContractResult:
    has_before: bool
    has_after: bool
    has_rule_family: bool
    has_semantic_delta: bool


def check_provenance_contract(provenance: Mapping[str, Any]) -> ProvenanceContractResult:
    keys = set(provenance.keys())
    return ProvenanceContractResult(
        has_before=("before" in keys) or any(k.startswith("before_") for k in keys),
        has_after=("after" in keys) or any(k.startswith("after_") for k in keys),
        has_rule_family=("rule_family" in keys) or ("identity_name" in keys),
        has_semantic_delta=("invariant_signature" in keys) or ("semantic_delta" in keys) or ("before_fingerprint" in keys) or ("after_fingerprint" in keys) or any("semantics" in k for k in keys),
    )
