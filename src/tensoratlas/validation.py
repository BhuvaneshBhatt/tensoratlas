"""Shared validation helpers for TensorAtlas public APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ValidationReport:
    """Structured validation result for user-facing diagnostics.

    ``validate()`` methods still raise on invalid objects for backwards
    compatibility.  ``validation_report()`` methods can return this lightweight
    object when a caller wants diagnostics without exception handling.
    """

    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.ok

    def raise_if_invalid(self) -> None:
        """Raise ``ValueError`` with all errors if the report is not valid."""
        self.raise_as(ValueError)

    def raise_as(self, error_type=ValueError) -> None:
        """Raise ``error_type`` with all errors if the report is invalid."""
        if not self.ok:
            raise error_type("; ".join(self.errors) or "validation failed")


def valid_report(*, warnings: Iterable[str] = ()) -> ValidationReport:
    """Return a successful validation report."""
    return ValidationReport(True, (), tuple(warnings))


def invalid_report(*errors: str, warnings: Iterable[str] = ()) -> ValidationReport:
    """Return a failed validation report with one or more messages."""
    return ValidationReport(False, tuple(errors), tuple(warnings))


def check_indices(context: str, dimension: int, *indices: int) -> None:
    """Raise ``IndexError`` if any index is outside ``range(dimension)``."""
    bad = [index for index in indices if index < 0 or index >= dimension]
    if bad:
        raise IndexError(f"{context} index out of range for dimension {dimension}: {tuple(bad)}")
