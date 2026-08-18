"""Formal sign and convention policies for geometry calculations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .manifolds import TensorKernelError

CurvatureDerivativeOrder = Literal["plus", "minus"]
RicciContraction = Literal["first_third", "first_fourth"]
LaplacianSign = Literal["positive", "negative"]
CliffordSign = Literal["plus", "minus"]


@dataclass(frozen=True, slots=True)
class SignatureConvention:
    """Metric-signature metadata.

    ``positive`` and ``negative`` count the positive and negative metric
    eigenvalues.  The determinant parity controls Hodge-star and Clifford sign
    conventions in pseudo-Riemannian settings.
    """

    positive: int
    negative: int = 0

    def __post_init__(self) -> None:
        if self.positive < 0 or self.negative < 0:
            raise TensorKernelError("Signature counts must be nonnegative.")
        if self.dimension <= 0:
            raise TensorKernelError("Metric signature must have positive dimension.")

    @property
    def dimension(self) -> int:
        return self.positive + self.negative

    @property
    def timelike_parity(self) -> int:
        return self.negative % 2

    @classmethod
    def riemannian(cls, dimension: int) -> "SignatureConvention":
        return cls(dimension, 0)

    @classmethod
    def lorentzian_mostly_plus(cls, dimension: int) -> "SignatureConvention":
        return cls(dimension - 1, 1)

    @classmethod
    def lorentzian_mostly_minus(cls, dimension: int) -> "SignatureConvention":
        return cls(1, dimension - 1)


@dataclass(frozen=True, slots=True)
class CurvatureConvention:
    """Riemann/Ricci sign and contraction choices.

    ``derivative_order='plus'`` denotes
    ``R^a{}_{bcd}=d_c Gamma^a_{bd}-d_d Gamma^a_{bc}+...``.  ``'minus'`` is
    the opposite Riemann sign.  ``ricci_contraction`` records which curvature
    slots are contracted for Ricci in abstract formulas.
    """

    derivative_order: CurvatureDerivativeOrder = "plus"
    ricci_contraction: RicciContraction = "first_third"

    def __post_init__(self) -> None:
        if self.derivative_order not in {"plus", "minus"}:
            raise TensorKernelError("Unsupported curvature derivative-order convention.")
        if self.ricci_contraction not in {"first_third", "first_fourth"}:
            raise TensorKernelError("Unsupported Ricci contraction convention.")

    @property
    def riemann_sign(self) -> int:
        return 1 if self.derivative_order == "plus" else -1

    def conversion_factor_to(self, other: "CurvatureConvention") -> int:
        return self.riemann_sign * other.riemann_sign


@dataclass(frozen=True, slots=True)
class HodgeConvention:
    """Hodge-star sign convention.

    The default gives ``**omega = (-1)^(p(n-p)+q) omega`` for a metric with
    ``q`` negative eigenvalues.  ``orientation`` flips the chosen volume-form
    orientation without changing the square-sign formula.
    """

    signature: SignatureConvention
    orientation: int = 1
    laplacian_sign: LaplacianSign = "positive"

    def __post_init__(self) -> None:
        if self.orientation not in {-1, 1}:
            raise TensorKernelError("Hodge orientation must be +1 or -1.")
        if self.laplacian_sign not in {"positive", "negative"}:
            raise TensorKernelError("Unsupported Hodge Laplacian sign convention.")

    def star_square_sign(self, degree: int) -> int:
        n = self.signature.dimension
        if degree < 0 or degree > n:
            raise TensorKernelError("Form degree must lie between 0 and the manifold dimension.")
        exponent = degree * (n - degree) + self.signature.negative
        return -1 if exponent % 2 else 1

    @property
    def laplacian_factor(self) -> int:
        return 1 if self.laplacian_sign == "positive" else -1


@dataclass(frozen=True, slots=True)
class CliffordConvention:
    """Clifford-algebra sign convention.

    ``anticommutator_sign='plus'`` means ``{gamma^a,gamma^b}=+2 g^{ab}``;
    ``'minus'`` means ``-2 g^{ab}``.
    """

    signature: SignatureConvention
    anticommutator_sign: CliffordSign = "plus"

    def __post_init__(self) -> None:
        if self.anticommutator_sign not in {"plus", "minus"}:
            raise TensorKernelError("Unsupported Clifford anticommutator convention.")

    @property
    def gamma_factor(self) -> int:
        return 2 if self.anticommutator_sign == "plus" else -2


@dataclass(frozen=True, slots=True)
class GeometryConvention:
    """Bundle of commonly needed geometry sign policies."""

    signature: SignatureConvention
    curvature: CurvatureConvention = CurvatureConvention()
    hodge: HodgeConvention | None = None
    clifford: CliffordConvention | None = None

    def __post_init__(self) -> None:
        if self.hodge is None:
            object.__setattr__(self, "hodge", HodgeConvention(self.signature))
        if self.clifford is None:
            object.__setattr__(self, "clifford", CliffordConvention(self.signature))


def default_riemannian_convention(dimension: int) -> GeometryConvention:
    """Return standard Riemannian conventions in ``dimension`` dimensions."""

    return GeometryConvention(SignatureConvention.riemannian(dimension))


def default_lorentzian_convention(dimension: int, *, mostly_plus: bool = True) -> GeometryConvention:
    """Return standard Lorentzian conventions for a metric of given dimension."""

    signature = (
        SignatureConvention.lorentzian_mostly_plus(dimension)
        if mostly_plus
        else SignatureConvention.lorentzian_mostly_minus(dimension)
    )
    return GeometryConvention(signature)
