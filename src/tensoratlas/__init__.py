"""Public TensorAtlas package interface."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from .public_api import (
    __all__,
    _LAZY_EXPORTS,
    chart_property_names,
    mapping_property_names,
    resolve_public_attribute,
)

try:
    __version__ = version("tensoratlas")
except PackageNotFoundError:
    __version__ = "0.1.0"

if "__version__" not in __all__:
    __all__ = tuple(__all__) + ("__version__",)


def __getattr__(name: str) -> Any:
    value = resolve_public_attribute(__name__, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


# Matrix simplification guards are intentionally not installed at package import time.
# Use tensoratlas.symbolic_simplification_policy.install_matrix_simplify_guard
# explicitly in workflows that need a guarded replacement for broad sympy.simplify calls.
