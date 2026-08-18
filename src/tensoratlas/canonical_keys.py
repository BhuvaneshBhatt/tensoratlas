from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping

import sympy as sp

from .bundle_identity import bundle_key
from .symbolic_decision import light_simplify


def _stable_text(obj: Any) -> str:
    try:
        return str(obj)
    except Exception:
        return f"<{type(obj).__name__}>"


def _sympy_tensor_key(expr: Any):
    """Cheap structural keys for SymPy abstract-tensor objects.

    SymPy's generic Basic traversal can touch deprecated TensorIndexType.data
    and other expensive internals.  These duck-typed keys are sufficient for
    TensorAtlas compatibility tests and keep classic abstract tensors out of
    pytest-sensitive recursive SymPy introspection.
    """
    cls = type(expr).__name__
    if cls == "TensorIndexType":
        return ("sympy_tensor_index_type", _stable_text(getattr(expr, "name", expr)))
    if cls == "TensorIndex":
        name = _stable_text(getattr(expr, "name", expr))
        typ = _sympy_tensor_key(getattr(expr, "tensor_index_type", None))
        is_up = bool(getattr(expr, "is_up", not str(expr).startswith("-")))
        return ("sympy_tensor_index", name, typ, "u" if is_up else "l")
    if cls == "TensorHead":
        name = _stable_text(getattr(expr, "name", expr))
        index_types = tuple(_sympy_tensor_key(t) for t in getattr(expr, "index_types", ()) or ())
        symmetry = _stable_text(getattr(expr, "symmetry", ""))
        comm = int(getattr(expr, "comm", 0) or 0)
        return ("sympy_tensor_head", name, index_types, symmetry, comm)
    if cls == "Tensor":
        comp = _sympy_tensor_key(getattr(expr, "component", None))
        try:
            indices = tuple(_sympy_tensor_key(i) for i in expr.get_indices())
        except Exception:
            indices = tuple(_stable_text(a) for a in getattr(expr, "args", ()))
        # Respect simple fully symmetric/antisymmetric heads by sorting labels
        # for symmetric heads; antisymmetric heads are left order-sensitive here.
        try:
            md = getattr(getattr(expr, "component", None), "_tensoratlas_metadata", {}) or {}
            if str(md.get("symmetry_kind", "")).lower() == "symmetric":
                indices = tuple(sorted(indices))
        except Exception:
            pass
        return ("sympy_tensor", comp, indices)
    if cls in {"TensMul", "TensAdd"}:
        op = "mul" if cls == "TensMul" else "add"
        args = tuple(_sympy_tensor_key(a) if type(a).__name__ in {"Tensor", "TensorHead", "TensorIndex", "TensorIndexType", "TensMul", "TensAdd"} else structural_key(a) for a in getattr(expr, "args", ()))
        if op in {"mul", "add"}:
            args = tuple(sorted(args))
        return ("sympy_tensor_expr", op, args)
    return None

@lru_cache(maxsize=4096)
def _sympy_structural_key_cached(expr: sp.Basic):
    # SymPy abstract tensor expressions implement legacy iteration through
    # the deprecated `.data` API. Generic scalar simplifiers such as
    # `cancel` / `factor_terms` probe iterability and therefore trigger that
    # deprecated path. Build tensor keys directly before any scalar
    # simplification.
    tensor_key = _sympy_tensor_key(expr)
    if tensor_key is not None:
        return tensor_key

    try:
        expr = light_simplify(expr)
    except Exception:
        pass

    return _sympy_structural_key(expr)


def _sympy_structural_key(expr: Any):
    if not isinstance(expr, sp.Basic):
        return structural_key(expr)
    tensor_key = _sympy_tensor_key(expr)
    if tensor_key is not None:
        return tensor_key
    if expr is sp.S.NaN:
        return ("sympy", "NaN")
    if expr is sp.S.Infinity:
        return ("sympy", "Infinity")
    if expr is sp.S.NegativeInfinity:
        return ("sympy", "NegativeInfinity")
    if expr is sp.S.ComplexInfinity:
        return ("sympy", "ComplexInfinity")
    if expr is sp.S.true:
        return ("sympy", "BooleanTrue")
    if expr is sp.S.false:
        return ("sympy", "BooleanFalse")
    if isinstance(expr, sp.Symbol):
        return ("sympy", "Symbol", expr.name, tuple(sorted(expr.assumptions0.items())))
    if isinstance(expr, sp.Dummy):
        return ("sympy", "Dummy", expr.name, tuple(sorted(expr.assumptions0.items())))
    if isinstance(expr, sp.Integer):
        return ("sympy", "Integer", int(expr))
    if isinstance(expr, sp.Rational):
        return ("sympy", "Rational", int(expr.p), int(expr.q))
    if isinstance(expr, sp.Float):
        return ("sympy", "Float", repr(float(expr)), int(getattr(expr, '_prec', 0)))
    if expr.is_Number:
        return ("sympy", type(expr).__name__, _stable_text(expr))
    if isinstance(expr, sp.Basic):
        return ("sympy", expr.func.__name__, tuple(_sympy_structural_key_cached(arg) if isinstance(arg, sp.Basic) else structural_key(arg) for arg in expr.args))
    return ("sympy", type(expr).__name__, _stable_text(expr))


def scalar_key(obj: Any):
    if hasattr(obj, "expr") and type(obj).__name__ == "ScalarField":
        return ("scalar", _sympy_structural_key_cached(sp.sympify(obj.expr)))
    if isinstance(obj, sp.Basic):
        return _sympy_structural_key_cached(obj)
    return structural_key(obj)


def structural_key(obj: Any):
    return _structural_key(obj, set())


def _structural_key(obj: Any, seen: set[int]):
    if obj is None:
        return ("none", "")
    if isinstance(obj, (str, int, float, bool)):
        return (type(obj).__name__, obj)
    if isinstance(obj, sp.Basic):
        return _sympy_structural_key_cached(obj)
    cls = type(obj).__name__
    if cls == "ScalarField" and hasattr(obj, "expr"):
        return ("scalar", _sympy_structural_key_cached(sp.sympify(obj.expr)))
    if cls in {"AbstractTensorExpr", "TensorAtlasAbstractExpr"} and hasattr(obj, "expr"):
        return (cls, structural_key(getattr(obj, "expr", None)))
    if hasattr(obj, "name") and hasattr(obj, "variance") and hasattr(obj, "bundle"):
        return ("index", obj.name, obj.variance, bundle_key(obj.bundle))
    if hasattr(obj, "chart") and hasattr(obj, "variance_spec") and hasattr(obj, "slot_bases") and hasattr(obj, "components"):
        return (
            "tensor_object",
            getattr(obj, "name", "") or "",
            getattr(obj, "variance_spec", ""),
            tuple((_structural_key(getattr(b, "name", None), seen), _structural_key(getattr(b, "kind", None), seen), bundle_key(getattr(b, "metadata", {}).get("bundle", None)), _structural_key(b, seen)) for b in getattr(obj, "slot_bases", tuple())),
        )
    # semantic-core integration for advanced geometry objects.
    if cls == "ExteriorFormNF" and hasattr(obj, "dimension") and hasattr(obj, "terms"):
        terms = tuple(sorted(((tuple(int(i) for i in blade), _sympy_structural_key_cached(sp.sympify(coeff))) for blade, coeff in getattr(obj, "terms", {}).items())))
        return (
            "exterior_form_nf",
            int(getattr(obj, "dimension", 0)),
            terms,
            tuple(getattr(obj, "basis_labels", tuple())),
            canonical_mapping_key(getattr(obj, "metadata", {})),
        )
    if cls == "SpinConnectionDef" and hasattr(obj, "coefficients") and hasattr(obj, "frame"):
        coeffs = tuple(sorted(((tuple(int(i) for i in k), _sympy_structural_key_cached(sp.sympify(v))) for k, v in getattr(obj, "coefficients", {}).items())))
        return (
            "spin_connection",
            getattr(obj, "name", ""),
            _structural_key(getattr(obj, "frame", None), seen),
            coeffs,
            tuple(getattr(obj, "metric_signature", tuple())),
            canonical_mapping_key(getattr(obj, "metadata", {})),
        )

    if cls == "HodgeExpr" and hasattr(obj, "form"):
        return (
            "hodge_expr",
            _structural_key(getattr(obj, "form", None), seen),
            _structural_key(getattr(obj, "clifford", None), seen),
            tuple(getattr(obj, "metric_signature", tuple())),
            canonical_mapping_key(getattr(obj, "metadata", {})),
        )
    if cls == "CodifferentialExpr" and hasattr(obj, "form"):
        return (
            "codifferential_expr",
            _structural_key(getattr(obj, "form", None), seen),
            tuple(_sympy_structural_key_cached(sp.sympify(c)) for c in getattr(obj, "coordinates", tuple())),
            _structural_key(getattr(obj, "clifford", None), seen),
            tuple(getattr(obj, "metric_signature", tuple())),
            canonical_mapping_key(getattr(obj, "metadata", {})),
        )
    if cls == "InteriorExpr" and hasattr(obj, "form"):
        return (
            "interior_expr",
            tuple(_sympy_structural_key_cached(sp.sympify(v)) for v in getattr(obj, "vector_components", tuple())),
            _structural_key(getattr(obj, "form", None), seen),
            canonical_mapping_key(getattr(obj, "metadata", {})),
        )
    if cls == "LieExpr" and hasattr(obj, "form"):
        return (
            "lie_expr",
            tuple(_sympy_structural_key_cached(sp.sympify(v)) for v in getattr(obj, "vector_components", tuple())),
            _structural_key(getattr(obj, "form", None), seen),
            tuple(_sympy_structural_key_cached(sp.sympify(c)) for c in getattr(obj, "coordinates", tuple())),
            canonical_mapping_key(getattr(obj, "metadata", {})),
        )
    if cls == "GammaStringExpr" and hasattr(obj, "factors"):
        return (
            "gamma_string_expr",
            _structural_key(getattr(obj, "clifford", None), seen),
            tuple(int(i) for i in getattr(obj, "factors", tuple())),
            _sympy_structural_key_cached(sp.sympify(getattr(obj, "scalar", 1))),
            canonical_mapping_key(getattr(obj, "metadata", {})),
        )
    if cls == "CliffordAlgebraDef" and hasattr(obj, "dimension") and hasattr(obj, "signature"):
        return (
            "clifford_algebra",
            getattr(obj, "name", ""),
            int(getattr(obj, "dimension", 0)),
            tuple(int(x) for x in getattr(obj, "signature", tuple())),
            getattr(obj, "generator_prefix", "gamma"),
            tuple(getattr(obj, "basis_labels", tuple())),
            canonical_mapping_key(getattr(obj, "metadata", {})),
        )
    if cls in {"TensorBasis", "TensorFrame"} and hasattr(obj, "kind") and hasattr(obj, "dimension"):
        chart = getattr(obj, "chart", None)
        chart_key = _structural_key(chart, seen) if chart is not None else ("none", "")
        md = dict(getattr(obj, "metadata", {}) or {})
        if "transform_to_chart" in md and chart is not None:
            try:
                coords = chart.symbols()
                matrix = sp.Matrix(md["transform_to_chart"](coords))
                md["transform_matrix_entries"] = [[structural_key(sp.sympify(matrix[i, j])) for j in range(matrix.cols)] for i in range(matrix.rows)]
            except Exception:
                md["transform_matrix_entries"] = "<unavailable>"
            md.pop("transform_to_chart", None)
        bundle = getattr(obj, "bundle", None) if cls == "TensorFrame" else md.get("bundle")
        return (
            cls.lower(),
            getattr(obj, "name", ""),
            getattr(obj, "kind", ""),
            int(getattr(obj, "dimension", 0) or 0),
            getattr(obj, "dual_name", None),
            chart_key,
            bundle_key(bundle),
            canonical_mapping_key(md),
        )

    if hasattr(obj, "variance_spec") and hasattr(obj, "chart") and hasattr(obj, "components"):
        return ("tensor_field", getattr(obj, "variance_spec", ""), _structural_key(getattr(obj, "chart", None), seen))
    if hasattr(obj, "tensor") and hasattr(obj, "indices"):
        return ("indexed_tensor", _structural_key(obj.tensor, seen), tuple(_structural_key(i, seen) for i in obj.indices))
    if hasattr(obj, "op") and hasattr(obj, "args"):
        return ("indexed_expr", obj.op, tuple(_structural_key(a, seen) for a in obj.args))
    if hasattr(obj, "kind") and hasattr(obj, "typed_slots") and hasattr(obj, "special_signature"):
        return factor_key(obj)
    if isinstance(obj, (tuple, list)):
        oid = id(obj)
        if oid in seen:
            return ("cycle", type(obj).__name__)
        seen.add(oid)
        try:
            return (type(obj).__name__, tuple(_structural_key(x, seen) for x in obj))
        finally:
            seen.discard(oid)
    if isinstance(obj, dict):
        oid = id(obj)
        if oid in seen:
            return ("cycle", "dict")
        seen.add(oid)
        try:
            return ("dict", tuple(sorted((_structural_key(k, seen), _structural_key(v, seen)) for k, v in obj.items())))
        finally:
            seen.discard(oid)
    return (type(obj).__name__, _stable_text(obj))


def factor_key(f: Any):
    raw = (
        getattr(f, "kind", None), getattr(f, "name", None), getattr(f, "variance_spec", tuple()),
        getattr(f, "tensor_space_sig", tuple()), getattr(f, "basis_names", tuple()),
        getattr(f, "typed_slots", tuple()), getattr(f, "symmetry", tuple()), getattr(f, "role", None),
        getattr(f, "dimension_hint", None), getattr(f, "orientation_hint", None), getattr(f, "frame_hint", None),
        getattr(f, "bundle_hint", tuple()), getattr(f, "chart_hint", None), getattr(f, "metric_hint", None),
        getattr(f, "parameter_hint", tuple()), getattr(f, "young_hint", tuple()), getattr(f, "signature_hint", None),
        getattr(f, "special_signature", tuple()), getattr(f, "variance_pattern_hint", tuple()),
        getattr(f, "contraction_hint", tuple()), getattr(f, "reduction_class_hint", tuple()),
        getattr(f, "symmetry_class_hint", tuple()), getattr(f, "rank_hint", None),
    )
    return tuple(structural_key(x) for x in raw)


def term_group_key(term: Any):
    return (
        tuple(factor_key(f) for f in getattr(term, "factors", tuple())),
        tuple(structural_key(x) for x in getattr(term, "free_signature", tuple())),
        tuple(structural_key(x) for x in getattr(term, "bundle_signature", tuple())),
    )


def term_sort_key(term: Any):
    return term_group_key(term) + (scalar_key(getattr(term, "scalar", None)),)


def tensor_key(obj: Any):
    return structural_key(obj)


def canonical_sort_key(obj: Any):
    return structural_key(obj)


def canonical_sequence_key(seq: Any):
    return tuple(structural_key(x) for x in seq)


def canonical_mapping_key(mapping: Mapping[Any, Any]):
    return tuple(sorted((structural_key(k), structural_key(v)) for k, v in mapping.items()))


def canonical_expr_fingerprint(obj: Any, *, dimension: Any | None = None, layer: str | None = None, policy: str | None = None):
    base = structural_key(obj)
    meta = []
    if dimension is not None:
        meta.append(("dimension", structural_key(sp.sympify(dimension))))
    if layer is not None:
        meta.append(("layer", layer))
    if policy is not None:
        meta.append(("policy", policy))
    return ("fingerprint", tuple(meta), base)


def canonical_named_aliases(name: str):
    key = str(name).strip().lower()
    aliases = {key}
    if key == "conversion":
        aliases.update({"schouten", "weyl_schouten_ricci_family"})
    elif key == "schouten":
        aliases.update({"conversion", "weyl_schouten_ricci_family"})
    elif key.startswith("dim_"):
        aliases.add("dimension_specific")
    elif key == "dimension_specific":
        aliases.add("dimension_specific")
    elif key == "low_dimensional":
        aliases.add("dimension_specific")
    return aliases
