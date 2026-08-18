"""Semantic index declarations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .manifolds import Manifold, TensorKernelError, VectorBundle

Variance = Literal["up", "down"]


def _normalize_variance(value: str) -> Variance:
    key = str(value).lower()
    if key in {"up", "u", "+", "contravariant", "upper"}:
        return "up"
    if key in {"down", "d", "-", "covariant", "lower"}:
        return "down"
    raise TensorKernelError(f"Unsupported index variance: {value!r}.")


@dataclass(frozen=True, slots=True)
class IndexType:
    """A family of indices attached to a manifold or vector bundle."""

    name: str
    manifold: Manifold
    bundle: VectorBundle | None = None
    dimension: int | str | None = None
    metric: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise TensorKernelError("Index type name must be non-empty.")
        if self.bundle is not None and self.bundle.base != self.manifold:
            raise TensorKernelError("Index type bundle must live over its manifold.")

    def index(self, name: str, *, variance: str = "up") -> "AbstractIndex":
        return AbstractIndex(name=name, index_type=self, variance=_normalize_variance(variance))

    def indices(self, names: str, *, variance: str = "up") -> tuple["AbstractIndex", ...]:
        return tuple(self.index(name, variance=variance) for name in names.split())


@dataclass(frozen=True, slots=True)
class AbstractIndex:
    """An abstract index with variance and ownership metadata."""

    name: str
    index_type: IndexType
    variance: Variance = "up"

    def __post_init__(self) -> None:
        if not self.name:
            raise TensorKernelError("Index name must be non-empty.")
        object.__setattr__(self, "variance", _normalize_variance(self.variance))

    @property
    def is_up(self) -> bool:
        return self.variance == "up"

    @property
    def is_down(self) -> bool:
        return self.variance == "down"

    def flipped(self) -> "AbstractIndex":
        return AbstractIndex(self.name, self.index_type, "down" if self.is_up else "up")

    def with_name(self, name: str) -> "AbstractIndex":
        return AbstractIndex(name, self.index_type, self.variance)

    def __neg__(self) -> "AbstractIndex":
        return self.flipped()

    def same_family(self, other: "AbstractIndex") -> bool:
        return isinstance(other, AbstractIndex) and self.name == other.name and self.index_type == other.index_type

    def __repr__(self) -> str:
        return ("^" if self.is_up else "_") + self.name
