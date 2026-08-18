"""Semantic objects for manifolds and tensor bundles.

These classes are intentionally independent of SymPy.  They describe the
mathematical domain on which tensor expressions live; realization as SymPy,
coordinate components, or another backend is a separate layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


class TensorKernelError(ValueError):
    """Raised when the semantic tensor kernel receives inconsistent input."""


@dataclass(frozen=True, slots=True)
class Manifold:
    """A differentiable manifold declaration.

    Parameters
    ----------
    name:
        User-facing manifold name.
    dimension:
        Integer or symbolic dimension.  The semantic kernel stores the value but
        does not perform scalar algebra with it.
    metadata:
        Optional immutable descriptive information such as signature notes or
        provenance.
    """

    name: str
    dimension: int | str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise TensorKernelError("Manifold name must be non-empty.")
        if isinstance(self.dimension, int) and self.dimension <= 0:
            raise TensorKernelError("Manifold dimension must be positive.")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def index_type(self, name: str | None = None, *, metric: str | None = None):
        from .indices import IndexType

        return IndexType(name or self.name, manifold=self, dimension=self.dimension, metric=metric)

    def indices(self, names: str, *, variance: str = "up", index_type=None):
        itype = index_type or self.index_type()
        return tuple(itype.index(name, variance=variance) for name in names.split())


@dataclass(frozen=True, slots=True)
class VectorBundle:
    """A vector bundle over a base manifold."""

    name: str
    base: Manifold
    rank: int | str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise TensorKernelError("Bundle name must be non-empty.")
        if isinstance(self.rank, int) and self.rank <= 0:
            raise TensorKernelError("Bundle rank must be positive.")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def index_type(self, name: str | None = None, *, metric: str | None = None):
        from .indices import IndexType

        return IndexType(name or self.name, manifold=self.base, bundle=self, dimension=self.rank, metric=metric)


@dataclass(frozen=True, slots=True)
class TensorBundle:
    """A tensor bundle assembled from contravariant/covariant vector bundles."""

    base: Manifold
    contravariant: tuple[VectorBundle, ...] = ()
    covariant: tuple[VectorBundle, ...] = ()

    @property
    def rank(self) -> int:
        return len(self.contravariant) + len(self.covariant)
