"""Derivative and curvature algebra for the semantic tensor kernel.

This module is intentionally lightweight and backend-independent.  It models
covariant derivative operators as structural tensor operations: linearity,
Leibniz expansion, metric compatibility, and the curvature/torsion commutator
on vectors and covectors.  Coordinate component expansion is a separate layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from .indices import AbstractIndex, IndexType
from .manifolds import TensorKernelError
from .tensor_expr import TensorExpr, TensorFactor, TensorTerm, canonicalize
from .tensor_heads import TensorHead

DerivativeKind = Literal["partial", "covariant"]


def _require_down_index(index: AbstractIndex, *, role: str) -> None:
    if not index.is_down:
        raise TensorKernelError(f"{role} index must be covariant/down; got {index!r}.")


def _fresh_index(base: str, index_type: IndexType, used: set[tuple[str, IndexType]]) -> AbstractIndex:
    counter = 1
    while True:
        name = f"{base}{counter}"
        if (name, index_type) not in used:
            return AbstractIndex(name, index_type, "up")
        counter += 1


def _used_index_families(expr: TensorExpr) -> set[tuple[str, IndexType]]:
    used: set[tuple[str, IndexType]] = set()
    for term in expr.terms:
        for factor in term.factors:
            for idx in factor.indices:
                used.add((idx.name, idx.index_type))
    return used


@dataclass(frozen=True, slots=True)
class DerivativeOperator:
    """A semantic derivative operator.

    Parameters
    ----------
    name:
        Symbolic name used when derivative tensor heads are created.
    index_type:
        Index family on which the derivative index lives.
    kind:
        ``"covariant"`` or ``"partial"``.  Only covariant
        derivatives expose curvature/torsion commutators.
    metric_compatible:
        If true, derivatives of metric and inverse-metric factors vanish.
    torsion_head:
        Optional torsion tensor with variance ``(^a, _b, _c)`` and antisymmetry
        in the last two slots.  If omitted, commutators are torsion-free.
    curvature_head:
        Optional curvature tensor with variance ``(^a, _b, _c, _d)`` and
        antisymmetry in the last two slots.  If omitted, a standard head is
        created lazily by :meth:`standard_curvature_head`.
    """

    name: str
    index_type: IndexType
    kind: DerivativeKind = "covariant"
    metric_compatible: bool = True
    torsion_head: TensorHead | None = None
    curvature_head: TensorHead | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise TensorKernelError("Derivative operator name must be non-empty.")
        if self.kind not in {"partial", "covariant"}:
            raise TensorKernelError(f"Unsupported derivative kind: {self.kind!r}.")
        if self.torsion_head is not None:
            _validate_torsion_head(self.torsion_head, self.index_type)
        if self.curvature_head is not None:
            _validate_curvature_head(self.curvature_head, self.index_type)

    def standard_curvature_head(self) -> TensorHead:
        if self.curvature_head is not None:
            return self.curvature_head
        return TensorHead.curvature(f"R_{self.name}", self.index_type)

    def standard_torsion_head(self) -> TensorHead | None:
        return self.torsion_head

    def derivative_head(self, head: TensorHead) -> TensorHead:
        """Return the structural tensor head representing one derivative of ``head``."""
        return TensorHead(
            f"{self.name}{head.name}",
            (self.index_type,) + head.index_types,
            variance=("down",) + head.variance,
            commutative=head.commutative,
        )

    def derivative_factor(self, factor: TensorFactor, derivative_index: AbstractIndex) -> TensorExpr:
        """Differentiate a single factor without applying product rules."""
        _require_down_index(derivative_index, role="Derivative")
        if derivative_index.index_type != self.index_type:
            raise TensorKernelError("Derivative index type does not match the derivative operator.")
        if self.metric_compatible and factor.head.role in {"metric", "inverse_metric"}:
            # Metric compatibility annihilates declared metric fields, but not
            # independent metric variations such as dg_ab or δg_ab.
            if not factor.head.name.startswith(("d", "δ")):
                return TensorExpr.zero()
        head = self.derivative_head(factor.head)
        return head(derivative_index, *factor.indices)

    def apply(self, expr: TensorExpr, derivative_index: AbstractIndex, *, expand_products: bool = True) -> TensorExpr:
        """Apply the derivative, using linearity and optionally the Leibniz rule."""
        expr = canonicalize(expr)
        _require_down_index(derivative_index, role="Derivative")
        if derivative_index.index_type != self.index_type:
            raise TensorKernelError("Derivative index type does not match the derivative operator.")
        if expr.is_zero:
            return TensorExpr.zero()
        out_terms: list[TensorTerm] = []
        for term in expr.terms:
            if not term.factors:
                continue
            if not expand_products or len(term.factors) == 1:
                differentiated = self._differentiate_whole_term(term, derivative_index)
                out_terms.extend(differentiated.terms)
                continue
            for pos, factor in enumerate(term.factors):
                df = self.derivative_factor(factor, derivative_index)
                for dterm in df.terms:
                    new_factors = term.factors[:pos] + dterm.factors + term.factors[pos + 1 :]
                    out_terms.append(TensorTerm(term.coefficient * dterm.coefficient, new_factors))
        return TensorExpr(tuple(out_terms)).canonicalized()

    def _differentiate_whole_term(self, term: TensorTerm, derivative_index: AbstractIndex) -> TensorExpr:
        if len(term.factors) == 1:
            differentiated = self.derivative_factor(term.factors[0], derivative_index)
            return term.coefficient * differentiated
        product_head = TensorHead(
            f"{self.name}Product",
            (self.index_type,),
            variance=("down",),
            commutative=False,
        )
        return TensorExpr((TensorTerm(term.coefficient, (TensorFactor(product_head, (derivative_index,)),) + term.factors),)).canonicalized()

    def commutator_on_factor(self, factor: TensorFactor, left_index: AbstractIndex, right_index: AbstractIndex) -> TensorExpr:
        """Return ``[D_left, D_right]`` acting on a vector or covector factor.

        The current commutator helper supports rank-one vector/covector factors
        over the derivative's index type.  Higher-rank tensor commutators are
        obtained by summing the corresponding slot actions.
        """
        if self.kind != "covariant":
            raise TensorKernelError("Only covariant derivatives have curvature commutators.")
        _require_down_index(left_index, role="Left derivative")
        _require_down_index(right_index, role="Right derivative")
        if left_index.index_type != self.index_type or right_index.index_type != self.index_type:
            raise TensorKernelError("Commutator derivative indices must match the derivative operator.")
        if len(factor.indices) != 1 or factor.indices[0].index_type != self.index_type:
            raise TensorKernelError("This commutator helper supports rank-one vector/covector factors only.")

        vector_index = factor.indices[0]
        used = {(left_index.name, left_index.index_type), (right_index.name, right_index.index_type)}
        used.update((idx.name, idx.index_type) for idx in factor.indices)
        dummy = _fresh_index("q", self.index_type, used)
        curvature = self.standard_curvature_head()
        torsion = self.standard_torsion_head()

        if vector_index.is_up:
            curvature_term = curvature(vector_index, -dummy, left_index, right_index) * factor.head(dummy)
        else:
            curvature_term = -curvature(dummy, vector_index, left_index, right_index) * factor.head(-dummy)

        if torsion is None:
            return canonicalize(curvature_term)

        # -T^q_ab D_q V^c for vectors, and the same transport term for covectors.
        d_index = -dummy
        transported = self.derivative_factor(factor, d_index)
        torsion_term = -(torsion(dummy, left_index, right_index) * transported)
        return canonicalize(curvature_term + torsion_term)


def covariant_derivative(
    name: str,
    index_type: IndexType,
    *,
    metric_compatible: bool = True,
    torsion: bool | TensorHead = False,
    curvature_head: TensorHead | None = None,
) -> DerivativeOperator:
    """Construct a standard covariant derivative operator."""
    torsion_head: TensorHead | None
    if torsion is True:
        torsion_head = TensorHead.torsion(f"T_{name}", index_type)
    elif torsion is False:
        torsion_head = None
    else:
        torsion_head = torsion
    return DerivativeOperator(
        name,
        index_type,
        kind="covariant",
        metric_compatible=metric_compatible,
        torsion_head=torsion_head,
        curvature_head=curvature_head,
    )


def partial_derivative(name: str, index_type: IndexType) -> DerivativeOperator:
    """Construct a structural partial derivative operator."""
    return DerivativeOperator(name, index_type, kind="partial", metric_compatible=False)


def commutator(operator: DerivativeOperator, factor: TensorFactor, left_index: AbstractIndex, right_index: AbstractIndex) -> TensorExpr:
    """Convenience wrapper for a covariant-derivative commutator."""
    return operator.commutator_on_factor(factor, left_index, right_index)


def _validate_torsion_head(head: TensorHead, index_type: IndexType) -> None:
    if head.rank != 3 or head.index_types != (index_type, index_type, index_type):
        raise TensorKernelError("Torsion head must have three slots over the derivative index type.")
    if head.variance != ("up", "down", "down"):
        raise TensorKernelError("Torsion head must have variance (^a, _b, _c).")


def _validate_curvature_head(head: TensorHead, index_type: IndexType) -> None:
    if head.rank != 4 or head.index_types != (index_type, index_type, index_type, index_type):
        raise TensorKernelError("Curvature head must have four slots over the derivative index type.")
    if head.variance != ("up", "down", "down", "down"):
        raise TensorKernelError("Curvature head must have variance (^a, _b, _c, _d).")
