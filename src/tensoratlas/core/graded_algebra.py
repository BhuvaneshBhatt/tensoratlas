"""Graded, Clifford, spinor, and gauge-algebra helpers.

The classes in this module extend the semantic kernel without depending on
SymPy.  They model parity-aware products, Clifford gamma generators, spinor
index families, and compact gauge-algebra declarations.  The implementation is
intentionally structural: it gives deterministic signs and canonical forms while
leaving representation-specific matrix evaluation to component layers.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Mapping, Sequence

from .indices import AbstractIndex, IndexType
from .conventions import CliffordConvention
from .manifolds import Manifold, TensorKernelError
from .tensor_expr import TensorExpr, TensorTerm, canonicalize
from .tensor_heads import TensorHead


Parity = int


def _parity(value: int | bool) -> int:
    return int(value) % 2


@dataclass(frozen=True, slots=True)
class GradedSymbol:
    """A symbolic factor with Z2 parity.

    ``parity=0`` denotes an even/bosonic factor and ``parity=1`` denotes an
    odd/fermionic factor.  Odd factors acquire the Koszul sign under swaps.
    """

    name: str
    parity: Parity = 0
    commutative: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise TensorKernelError("Graded symbols require a non-empty name.")
        object.__setattr__(self, "parity", _parity(self.parity))

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class GradedProduct:
    """A coefficient times an ordered product of graded factors."""

    coefficient: Fraction
    factors: tuple[GradedSymbol, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "coefficient", Fraction(self.coefficient))

    @classmethod
    def one(cls) -> "GradedProduct":
        return cls(Fraction(1), ())

    def canonicalized(self) -> "GradedProduct":
        """Sort commuting graded symbols with the Koszul sign."""
        coeff = self.coefficient
        factors = list(self.factors)
        # Bubble-sort keeps the implementation transparent and makes the sign
        # rule explicit.  Noncommutative factors remain in their relative order.
        changed = True
        while changed:
            changed = False
            for pos in range(len(factors) - 1):
                left = factors[pos]
                right = factors[pos + 1]
                if not (left.commutative and right.commutative):
                    continue
                if left.name <= right.name:
                    continue
                if left.parity and right.parity:
                    coeff *= -1
                factors[pos], factors[pos + 1] = right, left
                changed = True
        return GradedProduct(coeff, tuple(factors))

    def __mul__(self, other: "GradedProduct | GradedSymbol | int") -> "GradedProduct":
        if isinstance(other, int):
            return GradedProduct(self.coefficient * other, self.factors)
        if isinstance(other, GradedSymbol):
            return GradedProduct(self.coefficient, self.factors + (other,)).canonicalized()
        if isinstance(other, GradedProduct):
            return GradedProduct(self.coefficient * other.coefficient, self.factors + other.factors).canonicalized()
        return NotImplemented

    def __rmul__(self, other: "GradedSymbol | int") -> "GradedProduct":
        if isinstance(other, int):
            return GradedProduct(self.coefficient * other, self.factors)
        if isinstance(other, GradedSymbol):
            return GradedProduct(self.coefficient, (other,) + self.factors).canonicalized()
        return NotImplemented

    @property
    def parity(self) -> int:
        total = 0
        for factor in self.factors:
            total ^= factor.parity
        return total

    def __repr__(self) -> str:
        body = "*".join(repr(factor) for factor in self.factors) or "1"
        if self.coefficient == 1:
            return body
        if self.coefficient == -1:
            return f"-{body}"
        return f"{self.coefficient}*{body}"


def graded_commutator(left: GradedProduct, right: GradedProduct) -> tuple[GradedProduct, GradedProduct]:
    """Return the two products in ``A B - (-1)^(|A||B|) B A``."""
    sign = -1 if (left.parity and right.parity) else 1
    return left * right, (-sign) * (right * left)


@dataclass(frozen=True, slots=True)
class CliffordAlgebra:
    """Structural Clifford algebra over one vector-index family."""

    vector_index_type: IndexType
    metric: TensorHead
    gamma_name: str = "gamma"
    convention: CliffordConvention | None = None

    def __post_init__(self) -> None:
        if self.metric.role != "metric" and self.metric.role != "inverse_metric":
            raise TensorKernelError("Clifford algebra requires a metric or inverse-metric head.")
        if self.metric.index_types[0] != self.vector_index_type:
            raise TensorKernelError("Metric index type does not match the Clifford vector index type.")

    def gamma(self, vector_index: AbstractIndex, spinor_from: AbstractIndex, spinor_to: AbstractIndex) -> TensorExpr:
        """Return a noncommuting gamma factor ``gamma^a_A^B`` structurally."""
        head = TensorHead(
            self.gamma_name,
            (self.vector_index_type, spinor_from.index_type, spinor_to.index_type),
            variance=(vector_index.variance, spinor_from.variance, spinor_to.variance),
            commutative=False,
        )
        return head(vector_index, spinor_from, spinor_to)

    def anticommutator(
        self,
        left_vector: AbstractIndex,
        right_vector: AbstractIndex,
        spinor_from: AbstractIndex,
        spinor_to: AbstractIndex,
    ) -> TensorExpr:
        """Return ``{gamma^a, gamma^b}_A^B = 2 g^{ab} delta_A^B`` structurally."""
        if left_vector.index_type != self.vector_index_type or right_vector.index_type != self.vector_index_type:
            raise TensorKernelError("Gamma vector indices must use the Clifford vector index type.")
        spinor_delta = TensorHead.delta(f"delta_{spinor_from.index_type.name}", spinor_from.index_type)
        metric_head = self.metric
        if left_vector.is_up and right_vector.is_up and metric_head.role == "metric":
            metric_head = TensorHead.inverse_metric(f"{metric_head.name}inv", self.vector_index_type)
        elif left_vector.is_down and right_vector.is_down and metric_head.role == "inverse_metric":
            metric_head = TensorHead.metric(f"{metric_head.name}cov", self.vector_index_type)
        factor = self.convention.gamma_factor if self.convention is not None else 2
        return canonicalize(factor * metric_head(left_vector, right_vector) * spinor_delta(spinor_to, spinor_from))


@dataclass(frozen=True, slots=True)
class SpinorIndexFamily:
    """A spinor-index family tied to a manifold and optional chirality."""

    index_type: IndexType
    chirality: str | None = None

    @classmethod
    def create(cls, manifold: Manifold, name: str = "S", *, dimension: int | None = None, chirality: str | None = None) -> "SpinorIndexFamily":
        return cls(IndexType(name, manifold=manifold, dimension=dimension or manifold.dimension), chirality=chirality)

    def indices(self, names: str) -> tuple[AbstractIndex, ...]:
        return self.index_type.indices(names)


@dataclass(frozen=True, slots=True)
class GaugeAlgebra:
    """Compact declaration of a Lie-algebra index family and invariant heads."""

    name: str
    index_type: IndexType
    structure_constant: TensorHead
    killing_form: TensorHead

    @classmethod
    def create(cls, manifold: Manifold, name: str, *, dimension: int | None = None) -> "GaugeAlgebra":
        index_type = IndexType(name, manifold=manifold, dimension=dimension or manifold.dimension)
        structure = TensorHead(
            f"f_{name}",
            (index_type, index_type, index_type),
            symmetry="antisym_last2",
            variance=("up", "down", "down"),
            commutative=True,
        )
        killing = TensorHead.metric(f"kappa_{name}", index_type)
        return cls(name, index_type, structure, killing)

    def adjoint_indices(self, names: str) -> tuple[AbstractIndex, ...]:
        return self.index_type.indices(names)

    def bracket(self, left: AbstractIndex, right: AbstractIndex, out: AbstractIndex) -> TensorExpr:
        """Return the structural structure-constant component ``f^out_{left right}``."""
        if not out.is_up or not left.is_down or not right.is_down:
            raise TensorKernelError("Gauge bracket expects indices (^c, _a, _b).")
        return self.structure_constant(out, left, right)
