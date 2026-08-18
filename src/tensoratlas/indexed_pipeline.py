from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .tensor_indices import IndexedCanonicalizationReport, normalize_indexed_expression

INDEXED_NORMALIZATION_STAGES: tuple[str, ...] = (
    "expand_expression",
    "alpha_rename_dummies",
    "lower_metric_and_delta",
    "canonicalize_special_tensors",
    "sort_commutative_factors",
    "rebuild_expression",
)


@dataclass(frozen=True)
class IndexedNormalizationPlan:
    """Documented normalization pipeline for indexed expressions."""

    stages: Sequence[str] = INDEXED_NORMALIZATION_STAGES

    def describe(self) -> str:
        return " -> ".join(self.stages)


def explain_indexed_normalization(expr) -> IndexedCanonicalizationReport:
    """Normalize an indexed expression and return the structured canonicalization report."""
    return normalize_indexed_expression(expr, return_report=True)
