from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence, TYPE_CHECKING

import sympy as sp

from .semantic_core import compile_semantic_node, normalize_semantic_node, semantic_node_fingerprint
if TYPE_CHECKING:
    from .geometry_components import ComponentTensorField
    from .exterior_geometry import ExteriorFormNF
    from .tensor_indices import IndexedTensor, IndexedTensorExpr


class TensorExprKind(str, Enum):
    SCALAR = "scalar"
    SYMBOL = "symbol"
    ABSTRACT_TENSOR = "abstract_tensor"
    INDEXED_TENSOR = "indexed_tensor"
    TENSOR_FORM = "tensor_form"
    EXTERIOR_FORM = "exterior_form"
    COVARIANT_DERIVATIVE = "covariant_derivative"
    CURVATURE = "curvature"
    SPIN_OBJECT = "spin_object"
    GAMMA_OBJECT = "gamma_object"
    VARIATION = "variation"
    ADD = "add"
    MUL = "mul"
    NEG = "neg"
    CONTRACT = "contract"
    WEDGE = "wedge"
    ZERO = "zero"
    LEGACY = "classic"


@dataclass(frozen=True)
class IRProvenanceStep:
    rule: str
    source: str = "unknown"
    before_key: tuple[Any, ...] | None = None
    after_key: tuple[Any, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IRProvenance:
    origin: str = "direct"
    steps: tuple[IRProvenanceStep, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def append(self, rule: str, *, source: str = "unknown", before_key: tuple[Any, ...] | None = None, after_key: tuple[Any, ...] | None = None, **metadata: Any) -> "IRProvenance":
        return IRProvenance(
            origin=self.origin,
            steps=self.steps + (IRProvenanceStep(rule=rule, source=source, before_key=before_key, after_key=after_key, metadata=metadata),),
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class TensorExpr:
    """Canonical TensorAtlas internal representation.

    All symbolic subsystems should either produce this type directly or use
    ``to_tensor_expr`` as an adapter.  ``kind`` is intentionally a string for
    compatibility with the older TensorExpr API, but constructors in this
    module use the typed ``TensorExprKind`` vocabulary where possible.
    """

    kind: str
    payload: Any = None
    children: tuple["TensorExpr", ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provenance: IRProvenance = field(default_factory=IRProvenance)

    def with_children(self, children: Sequence["TensorExpr"]) -> "TensorExpr":
        return TensorExpr(self.kind, self.payload, tuple(children), dict(self.metadata), self.provenance)

    def with_metadata(self, **metadata: Any) -> "TensorExpr":
        md = dict(self.metadata)
        md.update(metadata)
        return TensorExpr(self.kind, self.payload, self.children, md, self.provenance)

    def with_provenance(self, provenance: IRProvenance) -> "TensorExpr":
        return TensorExpr(self.kind, self.payload, self.children, dict(self.metadata), provenance)




def _coerce_provenance(provenance: IRProvenance | Mapping[str, Any] | None, origin: str | None = None) -> IRProvenance:
    if isinstance(provenance, IRProvenance):
        return provenance
    md = dict(provenance or {})
    return IRProvenance(origin=origin or str(md.pop("origin", "direct")), metadata=md)


def ir_node(kind: str | TensorExprKind, *children: TensorExpr, payload: Any = None, metadata: Mapping[str, Any] | None = None, provenance: IRProvenance | Mapping[str, Any] | None = None, **metadata_kwargs: Any) -> TensorExpr:
    combined = dict(metadata or {})
    combined.update(metadata_kwargs)
    k = kind.value if isinstance(kind, TensorExprKind) else str(kind)
    return TensorExpr(kind=k, payload=payload, children=tuple(children), metadata=combined, provenance=_coerce_provenance(provenance))


def scalar_ir(value: Any) -> TensorExpr:
    return ir_node(TensorExprKind.SCALAR, payload=sp.sympify(value), provenance={"origin": "scalar"})


def symbol_ir(name: str) -> TensorExpr:
    return ir_node(TensorExprKind.SYMBOL, payload=str(name), provenance={"origin": "symbol"})


def abstract_tensor_expr(name: str, *, rank: int | None = None, variance: Sequence[str] = (), symmetries: Mapping[str, Any] | None = None, indices: Sequence[Any] = ()) -> TensorExpr:
    return ir_node(TensorExprKind.ABSTRACT_TENSOR, payload=str(name), rank=rank, variance=tuple(variance), indices=tuple(indices), symmetry_metadata=dict(symmetries or {}), provenance={"origin": "abstract_tensor"})


def indexed_tensor_expr(name: str, indices: Sequence[Any], *, variance_spec: str = "", symmetries: Mapping[str, Any] | None = None) -> TensorExpr:
    return ir_node(TensorExprKind.INDEXED_TENSOR, payload=str(name), tensor_name=str(name), variance_spec=variance_spec, indices=tuple(indices), symmetry_metadata=dict(symmetries or {}), provenance={"origin": "indexed_tensor"})


def tensor_form_ir(name: str, *, degree: int | None = None, basis: Sequence[Any] = (), terms: Any = None) -> TensorExpr:
    return ir_node(TensorExprKind.TENSOR_FORM, payload=terms if terms is not None else str(name), name=str(name), degree=degree, basis_labels=tuple(basis), provenance={"origin": "tensor_form"})


def exterior_form_ir(terms: Any, *, dimension: int, degree: int | None = None, basis_labels: Sequence[Any] = ()) -> TensorExpr:
    return ir_node(TensorExprKind.EXTERIOR_FORM, payload=terms, dimension=dimension, degree=degree, basis_labels=tuple(basis_labels), provenance={"origin": "exterior_form"})


def covariant_derivative_ir(expr: TensorExpr, *, index: Any = None, connection: str | None = None) -> TensorExpr:
    return ir_node(TensorExprKind.COVARIANT_DERIVATIVE, expr, index=index, connection=connection, provenance={"origin": "covariant_derivative"})


def curvature_ir(family: str, *, name: str | None = None, rank: int | None = None, dimension: int | None = None, indices: Sequence[Any] = (), convention: str | None = None) -> TensorExpr:
    return ir_node(TensorExprKind.CURVATURE, payload=name or family, family=family, rank=rank, dimension=dimension, indices=tuple(indices), convention=convention, provenance={"origin": "curvature"})


def spin_object_ir(name: str, *, indices: Sequence[Any] = (), bundle: str | None = None) -> TensorExpr:
    return ir_node(TensorExprKind.SPIN_OBJECT, payload=str(name), indices=tuple(indices), bundle=bundle, provenance={"origin": "spin"})


def gamma_object_ir(name: str = "gamma", *, indices: Sequence[Any] = (), dimension: int | None = None, signature: Any = None) -> TensorExpr:
    return ir_node(TensorExprKind.GAMMA_OBJECT, payload=str(name), indices=tuple(indices), dimension=dimension, signature=signature, provenance={"origin": "gamma"})


def variation_ir(expr: TensorExpr, *, field: str = "g", order: int = 1) -> TensorExpr:
    return ir_node(TensorExprKind.VARIATION, expr, field=field, order=order, provenance={"origin": "variation"})


def rewrite_with_provenance(ir: TensorExpr, rewritten: TensorExpr, *, rule: str, source: str = "rewrite", **metadata: Any) -> TensorExpr:
    before = canonical_ir_key(ir)
    after = canonical_ir_key(rewritten)
    return rewritten.with_provenance(ir.provenance.append(rule, source=source, before_key=before, after_key=after, **metadata))


def ir_to_dict(ir: TensorExpr) -> dict[str, Any]:
    return {
        "kind": ir.kind,
        "payload": ir.payload,
        "metadata": dict(ir.metadata),
        "provenance": {
            "origin": ir.provenance.origin,
            "metadata": dict(ir.provenance.metadata),
            "steps": [
                {"rule": s.rule, "source": s.source, "before_key": s.before_key, "after_key": s.after_key, "metadata": dict(s.metadata)}
                for s in ir.provenance.steps
            ],
        },
        "children": [ir_to_dict(ch) for ch in ir.children],
    }


def map_tensor_expr(ir: TensorExpr, fn: Callable[[TensorExpr], TensorExpr]) -> TensorExpr:
    children = tuple(map_tensor_expr(ch, fn) for ch in ir.children)
    rebuilt = TensorExpr(kind=ir.kind, payload=ir.payload, children=children, metadata=dict(ir.metadata), provenance=ir.provenance)
    return fn(rebuilt)


def _canonicalize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((_canonicalize_value(k), _canonicalize_value(v)) for k, v in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_canonicalize_value(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted((_canonicalize_value(v) for v in value), key=repr))
    if isinstance(value, sp.Basic):
        return ("sympy", sp.srepr(value))
    return value


_COMMUTATIVE_KINDS = {
    "add", "mul", "indexed_expr:add", "indexed_expr:tensor_product", "indexed_expr:mul",
    "scalar:add", "scalar:mul", "curvature_linear_combo", "contraction", "wedge",
}


def canonical_ir_key(ir: TensorExpr) -> tuple[Any, ...]:
    md = tuple(sorted((k, _canonicalize_value(v)) for k, v in dict(ir.metadata).items() if k != "provenance"))
    payload = _canonicalize_value(ir.payload)
    child_keys = tuple(canonical_ir_key(ch) for ch in ir.children)
    if ir.kind in _COMMUTATIVE_KINDS:
        child_keys = tuple(sorted(child_keys, key=repr))
    return (ir.kind, payload, md, child_keys)


def tensor_expr_sort_key(ir: TensorExpr) -> str:
    return repr(canonical_ir_key(ir))


def _flatten_associative(ir: TensorExpr, kind: str) -> tuple[TensorExpr, ...]:
    parts: list[TensorExpr] = []
    for ch in ir.children:
        if ch.kind == kind:
            parts.extend(_flatten_associative(ch, kind))
        else:
            parts.append(ch)
    return tuple(parts)


def _is_zero_ir(ir: TensorExpr) -> bool:
    if ir.kind == "zero":
        return True
    if ir.kind in {"scalar:zero", "scalar:number", "scalar:integer", "scalar:rational", "scalar"}:
        try:
            return sp.sympify(ir.payload) == 0
        except Exception:
            return False
    return False


def _scalar_payload(ir: TensorExpr) -> sp.Expr | None:
    if ir.kind in {"scalar", "scalar:number", "scalar:integer", "scalar:rational"}:
        try:
            return sp.sympify(ir.payload)
        except Exception:
            return None
    return None


def normalize_tree_ir(ir: TensorExpr) -> TensorExpr:
    children = tuple(normalize_tree_ir(ch) for ch in ir.children)
    md = dict(ir.metadata)
    rebuilt = TensorExpr(kind=ir.kind, payload=ir.payload, children=children, metadata=md, provenance=ir.provenance)

    if rebuilt.kind in {"neg", "unary:neg"} and len(children) == 1:
        child = children[0]
        if child.kind in {"neg", "unary:neg"} and len(child.children) == 1:
            return child.children[0]
        scalar_value = _scalar_payload(child)
        if scalar_value is not None:
            return scalar_ir(-scalar_value)
        if _is_zero_ir(child):
            return ir_node("zero")

    if rebuilt.kind in {"add", "indexed_expr:add", "scalar:add", "curvature_linear_combo"}:
        parts = [p for p in _flatten_associative(rebuilt, rebuilt.kind) if not _is_zero_ir(p)]
        if not parts:
            return ir_node("zero")
        if len(parts) == 1 and rebuilt.kind == "add":
            return parts[0]
        return TensorExpr(kind=rebuilt.kind, payload=rebuilt.payload, children=tuple(sorted(parts, key=tensor_expr_sort_key)), metadata=md, provenance=ir.provenance)

    if rebuilt.kind in {"mul", "indexed_expr:mul", "scalar:mul", "indexed_expr:tensor_product"}:
        parts = list(_flatten_associative(rebuilt, rebuilt.kind))
        coeff = md.get("coefficient")
        new_parts: list[TensorExpr] = []
        for part in parts:
            scalar_value = _scalar_payload(part)
            if scalar_value is not None:
                coeff = scalar_value if coeff is None else sp.sympify(coeff) * scalar_value
            elif _is_zero_ir(part):
                return ir_node("zero")
            else:
                new_parts.append(part)
        if coeff is not None and sp.sympify(coeff) == 0:
            return ir_node("zero")
        if rebuilt.kind == "mul" and len(new_parts) == 1 and (coeff is None or sp.sympify(coeff) == 1):
            return new_parts[0]
        new_md = dict(md)
        if coeff is None:
            new_md.pop("coefficient", None)
        else:
            new_md["coefficient"] = sp.simplify(coeff)
        return TensorExpr(kind=rebuilt.kind, payload=rebuilt.payload, children=tuple(sorted(new_parts, key=tensor_expr_sort_key)), metadata=new_md, provenance=ir.provenance)

    return rebuilt


@dataclass(frozen=True)
class TensorExprCompilationReport:
    original: Any
    ir: TensorExpr
    semantic_fingerprint: tuple[Any, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TensorExprExecutionReport:
    original: Any
    ir_kind: str
    normalized_ir: TensorExpr
    materialized: Any
    semantic_fingerprint: tuple[Any, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _to_tuple_terms(terms_dict):
    return tuple(sorted(((tuple(k), sp.sympify(v)) for k, v in terms_dict.items()), key=repr))


def _index_tuple(indices: Sequence[Any]) -> tuple[Any, ...]:
    return tuple((getattr(i, "name", str(i)), getattr(i, "variance", "")) for i in indices)


def _duck_curvature(obj: Any) -> TensorExpr | None:
    cls = obj.__class__.__name__
    if cls == "CurvatureSymbol" and all(hasattr(obj, a) for a in ("family", "rank", "dimension", "name")):
        return curvature_ir(obj.family, name=obj.name, rank=obj.rank, dimension=obj.dimension, convention=getattr(obj, "metadata", {}).get("convention"), indices=getattr(obj, "metadata", {}).get("indices", ()))
    if cls == "CurvatureExpr" and hasattr(obj, "op") and hasattr(obj, "args"):
        children = tuple(to_tensor_expr(arg) for arg in obj.args if not isinstance(arg, str))
        md = dict(getattr(obj, "metadata", {}) or {})
        if obj.op in {"contract", "decompose"} and len(obj.args) >= 2:
            md["target_family"] = obj.args[1]
        return ir_node(f"curvature_expr:{obj.op}", *children, metadata=md, provenance={"origin": "curvature_expr"})
    return None


def _duck_spin_gamma(obj: Any) -> TensorExpr | None:
    name = obj.__class__.__name__.lower()
    if "gamma" in name:
        return gamma_object_ir(getattr(obj, "name", obj.__class__.__name__), indices=getattr(obj, "indices", ()), dimension=getattr(obj, "dimension", None), signature=getattr(obj, "signature", None))
    if "spin" in name or "dirac" in name:
        return spin_object_ir(getattr(obj, "name", obj.__class__.__name__), indices=getattr(obj, "indices", ()), bundle=getattr(obj, "bundle", None))
    return None


def _duck_declaration(obj: Any) -> TensorExpr | None:
    if hasattr(obj, "to_ir") and obj.__class__.__module__.endswith("declarations"):
        result = obj.to_ir()
        if isinstance(result, TensorExpr):
            return result
    return None



def _duck_sympy_tensor_expr(obj: Any) -> TensorExpr | None:
    cls = obj.__class__.__name__
    if cls == "Tensor":
        comp = getattr(obj, "component", None)
        name = str(getattr(comp, "name", comp))
        try:
            indices = _index_tuple(obj.get_indices())
        except Exception:
            indices = tuple()
        md = {"tensor_name": name, "indices": indices}
        try:
            md["rank"] = len(indices)
        except Exception:
            pass
        return ir_node(TensorExprKind.ABSTRACT_TENSOR, payload=name, metadata=md, provenance={"origin": "sympy_tensor"})
    if cls in {"TensAdd", "Add"} and hasattr(obj, "args"):
        return ir_node(TensorExprKind.ADD, *(to_tensor_expr(a) for a in obj.args), provenance={"origin": "sympy_tensor_add"})
    if cls in {"TensMul", "Mul"} and hasattr(obj, "args"):
        return ir_node(TensorExprKind.MUL, *(to_tensor_expr(a) for a in obj.args), provenance={"origin": "sympy_tensor_mul"})
    return None

def to_tensor_expr(obj: Any) -> TensorExpr:
    if isinstance(obj, TensorExpr):
        return obj

    duck = _duck_declaration(obj)
    if duck is not None:
        return duck

    duck = _duck_curvature(obj)
    if duck is not None:
        return duck
    duck = _duck_spin_gamma(obj)
    if duck is not None:
        return duck

    duck = _duck_sympy_tensor_expr(obj)
    if duck is not None:
        return duck

    if obj.__class__.__name__ == "ExteriorFormNF" and hasattr(obj, "terms") and hasattr(obj, "dimension"):
        return exterior_form_ir(
            _to_tuple_terms(obj.terms),
            dimension=obj.dimension,
            degree=obj.degree,
            basis_labels=tuple(obj.basis_labels),
        )

    if obj.__class__.__name__ == "ComponentTensorField" and hasattr(obj, "components") and hasattr(obj, "variance_spec"):
        shape = tuple(getattr(obj.components, "shape", ()))
        flat = []
        if len(shape) == 0:
            flat = [sp.sympify(obj.components[()])]
        else:
            for idx in sp.utilities.iterables.cartes(*[range(s) for s in shape]):
                flat.append((tuple(idx), sp.sympify(obj.components[idx])))
        return ir_node(
            "component_tensor",
            payload=tuple(flat),
            name=obj.name,
            variance_spec=obj.variance_spec,
            basis_kind=obj.basis_kind,
            chart_name=getattr(obj.chart, "chart_name", ""),
            shape=shape,
            provenance={"origin": "component_tensor"},
        )

    if obj.__class__.__name__ == "IndexedTensor" and hasattr(obj, "tensor") and hasattr(obj, "indices"):
        return indexed_tensor_expr(
            getattr(obj.tensor, "name", ""),
            _index_tuple(obj.indices),
            variance_spec=getattr(obj.tensor, "variance_spec", ""),
            symmetries=dict(getattr(obj.tensor, "symmetry_metadata", {}) or {}),
        )

    if obj.__class__.__name__ == "IndexedTensorExpr" and hasattr(obj, "op") and hasattr(obj, "args"):
        return ir_node(
            f"indexed_expr:{obj.op}",
            *(to_tensor_expr(a) for a in obj.args),
            op=obj.op,
            provenance={"origin": "indexed_expr"},
        )

    if isinstance(obj, (sp.Basic, int, float)):
        node = normalize_semantic_node(compile_semantic_node(obj))
        return ir_node(
            f"scalar:{node.kind}",
            *(to_tensor_expr(ch) for ch in getattr(node, "children", ())),
            payload=getattr(node, "value", None),
            provenance={"origin": "sympy_scalar"},
        )

    node = normalize_semantic_node(compile_semantic_node(obj))
    return ir_node(
        f"semantic:{node.kind}",
        *(to_tensor_expr(ch) for ch in getattr(node, "children", ())),
        payload=getattr(node, "value", None),
        provenance={"origin": "semantic_node"},
    )


def compile_tensor_expr(obj: Any) -> TensorExpr:
    return to_tensor_expr(obj)


def normalize_tensor_expr(ir: TensorExpr) -> TensorExpr:
    return normalize_tree_ir(ir)


def materialize_tensor_expr(ir: TensorExpr) -> Any:
    if ir.kind == "exterior_form":
        from .exterior_geometry import ExteriorFormNF
        dim = int(ir.metadata["dimension"])
        basis_labels = tuple(ir.metadata.get("basis_labels", ()))
        terms = {tuple(k): v for k, v in ir.payload}
        return ExteriorFormNF(dim, terms, basis_labels=basis_labels, metadata={})

    if ir.kind == "component_tensor":
        from .geometry_components import ComponentTensorField
        shape = tuple(ir.metadata.get("shape", ()))
        if len(shape) == 0:
            arr = sp.MutableDenseNDimArray([ir.payload[0]])
        else:
            arr = sp.MutableDenseNDimArray.zeros(*shape)
            for idx, val in ir.payload:
                arr[idx] = val
        class _ChartStub:
            def __init__(self, name): self.chart_name = name
        chart_stub = _ChartStub(ir.metadata.get("chart_name", ""))
        return ComponentTensorField(
            name=ir.metadata.get("name", ""),
            chart=chart_stub,
            variance_spec=ir.metadata.get("variance_spec", ""),
            components=arr,
            basis_kind=ir.metadata.get("basis_kind", "canonical"),
            metadata={},
        )

    if ir.kind.startswith("scalar:"):
        kind = ir.kind.split(":", 1)[1]
        if not ir.children:
            return sp.sympify(ir.payload) if ir.payload is not None else sp.Symbol("u")
        mats = [materialize_tensor_expr(ch) for ch in ir.children]
        if kind == "add":
            return sp.Add(*mats)
        if kind == "mul":
            return sp.Mul(*mats)
        return mats[0] if len(mats) == 1 else tuple(mats)

    if ir.kind.startswith("indexed_expr:"):
        op = ir.kind.split(":", 1)[1]
        return {"op": op, "children": tuple(materialize_tensor_expr(ch) for ch in ir.children)}

    if ir.kind in {"indexed_tensor", "abstract_tensor", "curvature", "spin_object", "gamma_object"}:
        return {
            "kind": ir.kind,
            "name": ir.payload,
            "metadata": dict(ir.metadata),
        }

    return {
        "kind": ir.kind,
        "payload": ir.payload,
        "children": tuple(materialize_tensor_expr(ch) for ch in ir.children),
        "metadata": dict(ir.metadata),
    }


def compile_tensor_expr_report(obj: Any) -> TensorExprCompilationReport:
    semantic = normalize_semantic_node(compile_semantic_node(obj))
    ir = to_tensor_expr(obj)
    return TensorExprCompilationReport(
        original=obj,
        ir=ir,
        semantic_fingerprint=semantic_node_fingerprint(semantic),
        metadata={"semantic_kind": semantic.kind, "canonical_key": canonical_ir_key(ir)},
    )


def execute_tensor_expr(obj: Any) -> TensorExprExecutionReport:
    semantic = normalize_semantic_node(compile_semantic_node(obj))
    ir = to_tensor_expr(obj)
    nir = normalize_tensor_expr(ir)
    materialized = materialize_tensor_expr(nir)
    extra = {"canonical_key": canonical_ir_key(nir)}
    if obj.__class__.__name__ == "ComponentTensorField" and hasattr(obj, "components") and hasattr(obj, "variance_spec"):
        try:
            geom = component_geometry_report(obj.chart, include_curvature=True)
            extra["scalar_curvature"] = getattr(geom, "scalar_curvature", None)
        except Exception:
            pass
    return TensorExprExecutionReport(
        original=obj,
        ir_kind=ir.kind,
        normalized_ir=nir,
        materialized=materialized,
        semantic_fingerprint=semantic_node_fingerprint(semantic),
        metadata=extra,
    )
