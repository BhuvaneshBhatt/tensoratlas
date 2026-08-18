"""Public exception hierarchy for TensorAtlas.

The hierarchy keeps user-facing errors descriptive while letting downstream
projects catch TensorAtlas-specific failures without parsing Python internals.
"""

from __future__ import annotations


class TensorAtlasError(ValueError):
    """Base class for TensorAtlas user-facing errors."""


class CoordinateError(TensorAtlasError):
    """Raised for invalid coordinate charts or coordinate maps."""


class TensorShapeError(TensorAtlasError):
    """Raised when tensor component shapes, ranks, or variance data are invalid."""


class ContractionError(TensorShapeError):
    """Raised when tensor contraction axes are malformed or incompatible."""


class MetricError(TensorAtlasError):
    """Raised for invalid or unsupported metric data."""


class FormDegreeError(TensorAtlasError):
    """Raised for invalid differential-form degrees or tensor-valued components."""


class UnsupportedGeometryError(NotImplementedError, TensorAtlasError):
    """Raised when a mathematically meaningful operation is intentionally unsupported."""
