"""Tensor-head declarations for the semantic tensor kernel."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .indices import AbstractIndex, IndexType
from .manifolds import TensorKernelError
from .symmetries import SlotSymmetry

TensorRole = Literal["tensor", "metric", "inverse_metric", "delta", "epsilon", "generalized_delta"]
VariancePattern = tuple[str | None, ...]


@dataclass(frozen=True, slots=True, init=False)
class TensorHead:
    """A typed abstract tensor head.

    Parameters
    ----------
    name:
        User-facing tensor name.
    index_types:
        One index type per slot.
    symmetry:
        Monoterm slot symmetry for this head.
    variance:
        Optional required variance per slot.  Use ``"up"`` or ``"down"`` for
        fixed slots and ``None`` for slots that accept either variance.
    commutative:
        Whether factors with this head commute with other commutative factors.
    role:
        Semantic role used by structural monoterm contraction.
    """

    name: str
    index_types: tuple[IndexType, ...]
    symmetry: SlotSymmetry
    variance: VariancePattern
    commutative: bool
    role: TensorRole
    parity: int

    def __init__(
        self,
        name: str,
        index_types,
        *,
        symmetry: str | SlotSymmetry = "none",
        variance: tuple[str | None, ...] | None = None,
        commutative: bool = True,
        role: TensorRole = "tensor",
        parity: int | bool = 0,
    ):
        index_types_tuple = tuple(index_types)
        if variance is None:
            variance_tuple: VariancePattern = tuple(None for _ in index_types_tuple)
        else:
            variance_tuple = tuple(variance)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "index_types", index_types_tuple)
        object.__setattr__(self, "symmetry", symmetry if isinstance(symmetry, SlotSymmetry) else SlotSymmetry(symmetry))
        object.__setattr__(self, "variance", variance_tuple)
        object.__setattr__(self, "commutative", bool(commutative))
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "parity", int(parity) % 2)
        self.__post_init__()

    @classmethod
    def metric(cls, name: str, index_type: IndexType) -> "TensorHead":
        return cls(name, (index_type, index_type), symmetry="symmetric", variance=("down", "down"), role="metric")

    @classmethod
    def inverse_metric(cls, name: str, index_type: IndexType) -> "TensorHead":
        return cls(
            name,
            (index_type, index_type),
            symmetry="symmetric",
            variance=("up", "up"),
            role="inverse_metric",
        )

    @classmethod
    def delta(cls, name: str, index_type: IndexType) -> "TensorHead":
        return cls(name, (index_type, index_type), role="delta")


    @classmethod
    def curvature(cls, name: str, index_type: IndexType) -> "TensorHead":
        """Mixed Riemann curvature head R^a{}_{bcd}."""
        return cls(
            name,
            (index_type,) * 4,
            symmetry="antisym_last2",
            variance=("up", "down", "down", "down"),
        )

    @classmethod
    def torsion(cls, name: str, index_type: IndexType) -> "TensorHead":
        """Torsion head T^a{}_{bc}, antisymmetric in the lower slots."""
        return cls(
            name,
            (index_type,) * 3,
            symmetry="antisym_last2",
            variance=("up", "down", "down"),
        )

    @classmethod
    def riemann(cls, name: str, index_type: IndexType, *, variance: tuple[str | None, ...] | None = None) -> "TensorHead":
        return cls(name, (index_type,) * 4, symmetry="riemann", variance=variance)

    @classmethod
    def weyl(cls, name: str, index_type: IndexType, *, variance: tuple[str | None, ...] | None = None) -> "TensorHead":
        return cls(name, (index_type,) * 4, symmetry="weyl", variance=variance)


    @classmethod
    def ricci(cls, name: str, index_type: IndexType) -> "TensorHead":
        """Covariant Ricci tensor head Ric_ab."""
        return cls(name, (index_type, index_type), symmetry="symmetric", variance=("down", "down"))

    @classmethod
    def einstein(cls, name: str, index_type: IndexType) -> "TensorHead":
        """Covariant Einstein tensor head G_ab."""
        return cls(name, (index_type, index_type), symmetry="symmetric", variance=("down", "down"))

    @classmethod
    def scalar(cls, name: str) -> "TensorHead":
        """Rank-zero scalar tensor head."""
        return cls(name, (), symmetry="none")

    @classmethod
    def scalar_curvature(cls, name: str = "R") -> "TensorHead":
        """Rank-zero scalar-curvature head."""
        return cls.scalar(name)

    @classmethod
    def epsilon(cls, name: str, index_type: IndexType, *, rank: int | None = None, variance: str = "down") -> "TensorHead":
        """Fully antisymmetric Levi-Civita tensor/symbol head."""
        use_rank = rank if rank is not None else index_type.dimension
        if not isinstance(use_rank, int):
            raise TensorKernelError("Epsilon rank must be an integer or be inferred from an integer-dimensional index type.")
        if variance not in {"up", "down"}:
            raise TensorKernelError("Epsilon variance must be 'up' or 'down'.")
        return cls(name, (index_type,) * use_rank, symmetry="antisymmetric", variance=(variance,) * use_rank, role="epsilon")

    @classmethod
    def generalized_delta(cls, name: str, index_type: IndexType, order: int) -> "TensorHead":
        """Generalized Kronecker delta head with order upper and order lower slots."""
        if order <= 0:
            raise TensorKernelError("Generalized-delta order must be positive.")
        return cls(
            name,
            (index_type,) * (2 * order),
            variance=("up",) * order + ("down",) * order,
            role="generalized_delta",
        )

    def __post_init__(self) -> None:
        if not self.name:
            raise TensorKernelError("Tensor head name must be non-empty.")
        if self.role not in {"tensor", "metric", "inverse_metric", "delta", "epsilon", "generalized_delta"}:
            raise TensorKernelError(f"Unsupported tensor-head role: {self.role!r}.")
        if self.parity not in {0, 1}:
            raise TensorKernelError("Tensor-head parity must be 0 or 1.")
        if len(self.variance) != len(self.index_types):
            raise TensorKernelError("Variance pattern length must match tensor rank.")
        for item in self.variance:
            if item not in {None, "up", "down"}:
                raise TensorKernelError(f"Unsupported slot variance requirement: {item!r}.")
        if self.role in {"metric", "inverse_metric", "delta"} and len(self.index_types) != 2:
            raise TensorKernelError(f"{self.role} heads must have exactly two slots.")
        if self.role in {"metric", "inverse_metric", "delta"} and self.index_types[0] != self.index_types[1]:
            raise TensorKernelError(f"{self.role} heads require both slots to use the same index type.")
        if self.role == "epsilon":
            if not self.index_types:
                raise TensorKernelError("Epsilon heads require at least one slot.")
            if any(itype != self.index_types[0] for itype in self.index_types):
                raise TensorKernelError("Epsilon heads require a single index type.")
        if self.role == "generalized_delta":
            if len(self.index_types) < 2 or len(self.index_types) % 2:
                raise TensorKernelError("Generalized-delta heads require an even positive rank.")
            if any(itype != self.index_types[0] for itype in self.index_types):
                raise TensorKernelError("Generalized-delta heads require a single index type.")

    @property
    def rank(self) -> int:
        return len(self.index_types)

    def __call__(self, *indices: AbstractIndex):
        from .tensor_expr import TensorExpr, TensorFactor

        return TensorExpr.from_factor(TensorFactor(self, tuple(indices)))

    def _validate_role_variance(self, indices: tuple[AbstractIndex, ...]) -> None:
        for slot, (idx, required) in enumerate(zip(indices, self.variance)):
            if required is not None and idx.variance != required:
                raise TensorKernelError(
                    f"Tensor {self.name} slot {slot} expects {required} index variance, got {idx.variance}."
                )
        if self.role == "metric" and not all(idx.is_down for idx in indices):
            raise TensorKernelError("Metric heads expect two covariant/down indices.")
        if self.role == "inverse_metric" and not all(idx.is_up for idx in indices):
            raise TensorKernelError("Inverse-metric heads expect two contravariant/up indices.")
        if self.role == "delta" and indices[0].variance == indices[1].variance:
            raise TensorKernelError("Delta heads expect one up and one down index.")
