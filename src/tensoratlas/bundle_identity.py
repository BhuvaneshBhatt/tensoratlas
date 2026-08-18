from __future__ import annotations

from typing import Any

from .tensorform_types import TensorSpace
from .basis import IndexBundle


def _safe_name(obj: Any) -> str | None:
    name = getattr(obj, "name", None)
    if name is None:
        try:
            return str(obj)
        except Exception:
            return None
    return name


def bundle_parent_key(bundle: Any) -> tuple[Any, ...] | None:
    parent = getattr(bundle, "parent", None)
    if parent is None:
        return None
    return (type(parent).__name__, _safe_name(parent), getattr(parent, "dimension", None))


def bundle_key(bundle: Any) -> tuple[Any, ...]:
    if bundle is None:
        return ("none", None, None, None)
    if isinstance(bundle, TensorSpace):
        return ("TensorSpace", bundle.name, bundle.dimension, bundle_parent_key(bundle))
    if isinstance(bundle, IndexBundle):
        return ("IndexBundle", getattr(bundle, "name", None), getattr(bundle, "dimension", None), bundle_parent_key(bundle))
    return (type(bundle).__name__, _safe_name(bundle), getattr(bundle, "dimension", None), bundle_parent_key(bundle))


def bundle_metadata(bundle: Any) -> tuple[str | None, int | None]:
    key = bundle_key(bundle)
    return key[1], key[2]


def bundle_name(bundle: Any) -> str:
    return bundle_metadata(bundle)[0] or ""


def bundle_dim(bundle: Any) -> int | None:
    return bundle_metadata(bundle)[1]


def basis_bundle(basis: Any) -> Any | None:
    return getattr(basis, "metadata", {}).get("bundle") if basis is not None else None


def infer_bundle_from_basis(basis: Any) -> Any | None:
    return basis_bundle(basis)


def index_bundle_compatible(index_bundle: Any, basis: Any, *, allow_missing: bool = False) -> bool:
    if basis is None:
        return True
    basis_obj = basis_bundle(basis)
    if index_bundle is None:
        return True
    if basis_obj is None:
        return allow_missing
    left = bundle_key(index_bundle)
    right = bundle_key(basis_obj)
    if left == right:
        return True
    left_name, left_dim = left[1], left[2]
    right_name, right_dim = right[1], right[2]
    if left_name is not None and right_name is not None and left_name != right_name:
        return False
    if left_dim is not None and right_dim is not None and left_dim != right_dim:
        return False
    return True
