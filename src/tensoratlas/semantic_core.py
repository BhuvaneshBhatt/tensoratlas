from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import sympy as sp

from .canonical_keys import canonical_expr_fingerprint, canonical_mapping_key, canonical_sequence_key, structural_key


@dataclass(frozen=True)
class SemanticNode:
    """Universal typed node used by the semantic execution core.

    The goal is not to replace every public TensorAtlas object, but to give all
    major object families one structural execution substrate for compilation,
    normalization, fingerprinting, and semantic rewriting.
    """

    kind: str
    value: Any = None
    children: tuple["SemanticNode", ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TensorSemanticIR:
    """Typed intermediate representation describing a tensor object at a semantic layer."""

    layer: str
    expr: Any
    dimension: int | sp.Expr | None = None
    tensor_heads: tuple[str, ...] = tuple()
    free_indices: tuple[Any, ...] = tuple()
    dummy_indices: tuple[Any, ...] = tuple()
    contraction_pairs: tuple[tuple[Any, Any], ...] = tuple()
    ordered_factors: tuple[Any, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    root: SemanticNode | None = None


@dataclass(frozen=True)
class CanonicalSemanticForm:
    ir: TensorSemanticIR
    fingerprint: tuple[Any, ...]
    key: tuple[Any, ...]


def _stable_tuple(seq: Any) -> tuple[Any, ...]:
    if seq is None:
        return tuple()
    if isinstance(seq, tuple):
        return seq
    if isinstance(seq, list):
        return tuple(seq)
    return (seq,)


def semantic_layer_of(obj: Any, default: str = "abstract") -> str:
    cls = type(obj).__name__
    if cls == "ExteriorFormNF":
        return "exterior"
    if cls == "SpinConnectionDef":
        return "spin_connection"
    if cls == "CliffordAlgebraDef":
        return "clifford"
    if cls in {"HodgeExpr", "CodifferentialExpr", "InteriorExpr", "LieExpr"}:
        return "exterior_operator"
    if cls == "GammaStringExpr":
        return "gamma_string"
    if cls in {"TensorBasis", "TensorFrame"}:
        return "frame"
    if cls in {"IndexedTensor", "IndexedTensorExpr"}:
        return "indexed"
    return default


def _node_metadata_key(metadata: Mapping[str, Any]) -> tuple[Any, ...]:
    return canonical_mapping_key(metadata)


def semantic_node_fingerprint(node: SemanticNode) -> tuple[Any, ...]:
    return (
        node.kind,
        structural_key(node.value),
        tuple(semantic_node_fingerprint(child) for child in node.children),
        _node_metadata_key(node.metadata),
    )


def _sorted_nodes(nodes: Sequence[SemanticNode]) -> tuple[SemanticNode, ...]:
    return tuple(sorted(nodes, key=semantic_node_fingerprint))


def compile_semantic_node(obj: Any, *, layer: str | None = None, dimension: int | sp.Expr | None = None) -> SemanticNode:
    """Compile an object into the universal semantic node language.

    This intentionally covers the common symbolic execution cases directly
    (SymPy scalars, sums/products/powers/functions) and treats richer tensor and
    geometry objects as typed leaves carrying structural metadata.
    """

    layer_name = layer or semantic_layer_of(obj, default="abstract")
    cls = type(obj).__name__

    if isinstance(obj, sp.Basic):
        if obj.is_Atom:
            return SemanticNode("atom", value=obj, metadata={"layer": layer_name})
        if isinstance(obj, sp.Add):
            return SemanticNode("add", children=tuple(compile_semantic_node(arg, layer=layer_name, dimension=dimension) for arg in obj.args), metadata={"layer": layer_name, "commutative": True})
        if isinstance(obj, sp.Mul):
            return SemanticNode("mul", children=tuple(compile_semantic_node(arg, layer=layer_name, dimension=dimension) for arg in obj.args), metadata={"layer": layer_name, "commutative": bool(obj.is_commutative)})
        if isinstance(obj, sp.Pow):
            return SemanticNode("pow", children=(compile_semantic_node(obj.base, layer=layer_name, dimension=dimension), compile_semantic_node(obj.exp, layer=layer_name, dimension=dimension)), metadata={"layer": layer_name})
        return SemanticNode("call", value=getattr(obj.func, "__name__", str(obj.func)), children=tuple(compile_semantic_node(arg, layer=layer_name, dimension=dimension) for arg in obj.args), metadata={"layer": layer_name})

    if cls == "ExteriorFormNF":
        terms = getattr(obj, "terms", {})
        term_nodes = []
        for blade, coeff in sorted(terms.items(), key=lambda kv: tuple(int(i) for i in kv[0])):
            term_nodes.append(
                SemanticNode(
                    "exterior_term",
                    value=tuple(int(i) for i in blade),
                    children=(compile_semantic_node(coeff, layer="scalar", dimension=dimension),),
                    metadata={"dimension": getattr(obj, "dimension", None)},
                )
            )
        return SemanticNode(
            "exterior_form",
            value=tuple(getattr(obj, "basis_labels", tuple())),
            children=tuple(term_nodes),
            metadata={"layer": "exterior", "dimension": getattr(obj, "dimension", None)},
        )

    if cls == "SpinConnectionDef":
        coeffs = getattr(obj, "coefficients", {})
        coeff_nodes = tuple(
            SemanticNode(
                "spin_connection_component",
                value=tuple(int(i) for i in key),
                children=(compile_semantic_node(val, layer="scalar", dimension=dimension),),
            )
            for key, val in sorted(coeffs.items(), key=lambda kv: tuple(int(i) for i in kv[0]))
        )
        return SemanticNode(
            "spin_connection",
            value=tuple(getattr(obj, "metric_signature", tuple())),
            children=coeff_nodes,
            metadata={"layer": "spin_connection", "dimension": len(getattr(obj, "metric_signature", tuple()))},
        )

    if cls == "HodgeExpr":
        return SemanticNode(
            "hodge",
            children=(compile_semantic_node(getattr(obj, "form"), layer="exterior", dimension=getattr(getattr(obj, "form", None), "dimension", dimension)),),
            metadata={"layer": "exterior_operator", "clifford": getattr(obj, "clifford", None), "metric_signature": tuple(getattr(obj, "metric_signature", tuple())), "extra": dict(getattr(obj, "metadata", {}) or {})},
        )

    if cls == "CodifferentialExpr":
        return SemanticNode(
            "codifferential",
            children=(compile_semantic_node(getattr(obj, "form"), layer="exterior", dimension=getattr(getattr(obj, "form", None), "dimension", dimension)),),
            metadata={"layer": "exterior_operator", "coordinates": tuple(getattr(obj, "coordinates", tuple())), "clifford": getattr(obj, "clifford", None), "metric_signature": tuple(getattr(obj, "metric_signature", tuple())), "extra": dict(getattr(obj, "metadata", {}) or {})},
        )

    if cls == "InteriorExpr":
        vec_children = tuple(compile_semantic_node(v, layer="scalar", dimension=dimension) for v in getattr(obj, "vector_components", tuple()))
        return SemanticNode(
            "interior",
            children=(compile_semantic_node(getattr(obj, "form"), layer="exterior", dimension=getattr(getattr(obj, "form", None), "dimension", dimension)),) + vec_children,
            metadata={"layer": "exterior_operator", "extra": dict(getattr(obj, "metadata", {}) or {})},
        )

    if cls == "LieExpr":
        vec_children = tuple(compile_semantic_node(v, layer="scalar", dimension=dimension) for v in getattr(obj, "vector_components", tuple()))
        return SemanticNode(
            "lie",
            children=(compile_semantic_node(getattr(obj, "form"), layer="exterior", dimension=getattr(getattr(obj, "form", None), "dimension", dimension)),) + vec_children,
            metadata={"layer": "exterior_operator", "coordinates": tuple(getattr(obj, "coordinates", tuple())), "extra": dict(getattr(obj, "metadata", {}) or {})},
        )

    if cls == "GammaStringExpr":
        scalar_child = compile_semantic_node(getattr(obj, "scalar", 1), layer="scalar", dimension=dimension)
        factor_children = tuple(SemanticNode("gamma_index", value=int(i), metadata={"layer": "gamma_string"}) for i in getattr(obj, "factors", tuple()))
        return SemanticNode(
            "gamma_string",
            children=(scalar_child,) + factor_children,
            metadata={"layer": "gamma_string", "clifford": getattr(obj, "clifford", None), "extra": dict(getattr(obj, "metadata", {}) or {})},
        )

    if cls == "CliffordAlgebraDef":
        return SemanticNode(
            "clifford_algebra",
            value=getattr(obj, "name", None),
            children=tuple(SemanticNode("generator", value=label) for label in getattr(obj, "basis_labels", tuple())),
            metadata={
                "layer": "clifford",
                "dimension": getattr(obj, "dimension", None),
                "signature": tuple(getattr(obj, "signature", tuple())),
                "generator_prefix": getattr(obj, "generator_prefix", "gamma"),
            },
        )


    if cls == "IndexedTensor":
        from .tensor_indices import alpha_rename_dummies
        obj = alpha_rename_dummies(obj, prefix="d")
        try:
            obj = obj.canonicalize()
        except Exception:
            pass
        index_children = tuple(
            SemanticNode(
                "tensor_index",
                value=(idx.variance, idx.bundle),
                children=(SemanticNode("atom", value=sp.Symbol(idx.name), metadata={"layer": "indexed"}),),
                metadata={"layer": "indexed"},
            )
            for idx in getattr(obj, "indices", tuple())
        )
        return SemanticNode(
            "indexed_tensor",
            value=getattr(getattr(obj, "tensor", None), "name", None),
            children=index_children,
            metadata={
                "layer": "indexed",
                "variance_spec": getattr(getattr(obj, "tensor", None), "variance_spec", None),
                "dimension": dimension,
                "_original_obj": obj,
            },
        )

    if cls == "IndexedTensorExpr":
        from .tensor_indices import alpha_rename_dummies
        obj = alpha_rename_dummies(obj, prefix="d")
        op = getattr(obj, "op", None)
        kind = "indexed_add" if op == "add" else "indexed_tensor_product" if op == "tensor_product" else "indexed_expr"
        children = tuple(compile_semantic_node(arg, layer="indexed", dimension=dimension) for arg in getattr(obj, "args", tuple()))
        return SemanticNode(kind, value=op, children=children, metadata={"layer": "indexed", "dimension": dimension, "_original_obj": obj})

    if cls in {"TensorBasis", "TensorFrame"}:
        return SemanticNode(
            cls.lower(),
            value=getattr(obj, "name", None),
            children=tuple(),
            metadata={
                "layer": "frame",
                "dimension": getattr(obj, "dimension", None),
                "kind": getattr(obj, "kind", None),
                "dual_name": getattr(obj, "dual_name", None),
            },
        )

    return SemanticNode("leaf", value=obj, metadata={"layer": layer_name, "dimension": dimension})


def normalize_semantic_node(node: SemanticNode) -> SemanticNode:
    normalized_children = tuple(normalize_semantic_node(child) for child in node.children)

    if node.kind == "add":
        flat: list[SemanticNode] = []
        for child in normalized_children:
            if child.kind == "add":
                flat.extend(child.children)
            else:
                flat.append(child)
        kept = [child for child in flat if not (child.kind == "atom" and child.value == 0)]
        if not kept:
            return SemanticNode("atom", value=sp.Integer(0), metadata={"layer": node.metadata.get("layer", "scalar")})
        if len(kept) == 1:
            return kept[0]
        return SemanticNode("add", children=_sorted_nodes(kept), metadata=node.metadata)

    if node.kind == "mul":
        flat: list[SemanticNode] = []
        coeff = sp.Integer(1)
        for child in normalized_children:
            if child.kind == "mul" and child.metadata.get("commutative", False):
                flat.extend(child.children)
            elif child.kind == "atom" and isinstance(child.value, sp.Basic) and child.value.is_number:
                coeff *= child.value
            else:
                flat.append(child)
        if coeff == 0:
            return SemanticNode("atom", value=sp.Integer(0), metadata={"layer": node.metadata.get("layer", "scalar")})
        kept = [child for child in flat if not (child.kind == "atom" and child.value == 1)]
        if coeff != 1:
            kept.insert(0, SemanticNode("atom", value=sp.simplify(coeff), metadata={"layer": node.metadata.get("layer", "scalar")}))
        if not kept:
            return SemanticNode("atom", value=sp.Integer(1), metadata={"layer": node.metadata.get("layer", "scalar")})
        if len(kept) == 1:
            return kept[0]
        children = _sorted_nodes(kept) if node.metadata.get("commutative", False) else tuple(kept)
        return SemanticNode("mul", children=children, metadata=node.metadata)

    if node.kind == "pow" and len(normalized_children) == 2:
        base, exp = normalized_children
        if exp.kind == "atom" and exp.value == 1:
            return base
        if exp.kind == "atom" and exp.value == 0:
            return SemanticNode("atom", value=sp.Integer(1), metadata={"layer": node.metadata.get("layer", "scalar")})
        return SemanticNode("pow", children=(base, exp), metadata=node.metadata)

    if node.kind == "call":
        return SemanticNode(node.kind, value=node.value, children=normalized_children, metadata=node.metadata)


    if node.kind in {"indexed_add", "indexed_tensor_product"}:
        flat: list[SemanticNode] = []
        for child in normalized_children:
            if child.kind == node.kind:
                flat.extend(child.children)
            else:
                flat.append(child)
        if not flat:
            return SemanticNode("leaf", value=None, metadata=node.metadata)
        if len(flat) == 1:
            return flat[0]
        return SemanticNode(node.kind, value=node.value, children=_sorted_nodes(flat), metadata=node.metadata)

    if node.kind in {"indexed_tensor", "tensor_index"}:
        return SemanticNode(node.kind, value=node.value, children=normalized_children, metadata=node.metadata)

    if node.kind == "exterior_form":
        return SemanticNode(node.kind, value=node.value, children=_sorted_nodes(normalized_children), metadata=node.metadata)

    if node.kind in {"hodge", "codifferential", "interior", "lie"}:
        return SemanticNode(node.kind, value=node.value, children=normalized_children, metadata=node.metadata)

    if node.kind == "gamma_string":
        scalar = normalized_children[0] if normalized_children else SemanticNode("atom", value=sp.Integer(1), metadata={"layer": "scalar"})
        factors = [child for child in normalized_children[1:] if child.kind == "gamma_index"]
        clifford = node.metadata.get("clifford")
        scalar_val = materialize_semantic_node(scalar) if scalar.kind == "atom" else materialize_semantic_node(scalar)
        if clifford is not None and factors:
            facs = [int(child.value) for child in factors]
            changed = True
            while changed:
                changed = False
                i = 0
                while i < len(facs) - 1:
                    if facs[i] == facs[i+1]:
                        try:
                            scalar_val = sp.simplify(scalar_val * clifford.eta(facs[i], facs[i]))
                        except Exception:
                            pass
                        del facs[i:i+2]
                        changed = True
                        continue
                    i += 1
            scalar = compile_semantic_node(sp.sympify(scalar_val), layer="scalar")
            factors = tuple(SemanticNode("gamma_index", value=int(i), metadata={"layer": "gamma_string"}) for i in facs)
        return SemanticNode(node.kind, children=(scalar,) + tuple(factors), metadata=node.metadata)

    return SemanticNode(node.kind, value=node.value, children=normalized_children, metadata=node.metadata)


def materialize_semantic_node(node: SemanticNode) -> Any:
    if node.kind == "atom":
        return node.value
    if node.kind == "add":
        return sp.Add(*(materialize_semantic_node(child) for child in node.children))
    if node.kind == "mul":
        return sp.Mul(*(materialize_semantic_node(child) for child in node.children))
    if node.kind == "pow":
        return materialize_semantic_node(node.children[0]) ** materialize_semantic_node(node.children[1])
    if node.kind == "call":
        fn = getattr(sp, str(node.value), None)
        args = [materialize_semantic_node(child) for child in node.children]
        return fn(*args) if callable(fn) else sp.Function(str(node.value))(*args)

    if node.kind == "tensor_index":
        from .tensor_indices import TensorIndex
        name_node = node.children[0] if node.children else SemanticNode("atom", value=sp.Symbol("i"))
        name = str(materialize_semantic_node(name_node))
        variance, bundle = node.value if isinstance(node.value, tuple) and len(node.value) == 2 else ("u", None)
        return TensorIndex(name, variance, bundle)

    if node.kind == "indexed_tensor":
        original = node.metadata.get("_original_obj")
        if original is not None:
            return original
        from .tensor_indices import IndexedTensor
        from .tensor_core import TensorObject
        name = node.value or "T"
        variance_spec = node.metadata.get("variance_spec") or "".join((child.value[0] if isinstance(child.value, tuple) else "u") for child in node.children)
        dim = len(node.children)
        tensor = TensorObject(chart=None, components=sp.MutableDenseNDimArray.zeros(*([1] * max(dim, 1))), variance_spec=variance_spec, slot_bases=tuple(None for _ in range(dim)), name=name)
        indices = tuple(materialize_semantic_node(child) for child in node.children)
        return IndexedTensor(tensor, indices)

    if node.kind in {"indexed_add", "indexed_tensor_product", "indexed_expr"}:
        original = node.metadata.get("_original_obj")
        if original is not None:
            return original
        from .tensor_indices import IndexedTensorExpr
        op = node.value if node.value is not None else ("add" if node.kind == "indexed_add" else "tensor_product" if node.kind == "indexed_tensor_product" else "expr")
        args = tuple(materialize_semantic_node(child) for child in node.children)
        if len(args) == 1:
            return args[0]
        return IndexedTensorExpr(op, args)
    if node.kind == "hodge":
        from .semantic_ops import HodgeExpr
        form = materialize_semantic_node(node.children[0]) if node.children else node.value
        return HodgeExpr(form=form, clifford=node.metadata.get("clifford"), metric_signature=tuple(node.metadata.get("metric_signature", tuple())), metadata=dict(node.metadata.get("extra", {}) or {}))
    if node.kind == "codifferential":
        from .semantic_ops import CodifferentialExpr
        form = materialize_semantic_node(node.children[0]) if node.children else node.value
        return CodifferentialExpr(form=form, coordinates=tuple(node.metadata.get("coordinates", tuple())), clifford=node.metadata.get("clifford"), metric_signature=tuple(node.metadata.get("metric_signature", tuple())), metadata=dict(node.metadata.get("extra", {}) or {}))
    if node.kind == "interior":
        from .semantic_ops import InteriorExpr
        form = materialize_semantic_node(node.children[0]) if node.children else node.value
        vec = tuple(materialize_semantic_node(child) for child in node.children[1:])
        return InteriorExpr(vector_components=vec, form=form, metadata=dict(node.metadata.get("extra", {}) or {}))
    if node.kind == "lie":
        from .semantic_ops import LieExpr
        form = materialize_semantic_node(node.children[0]) if node.children else node.value
        vec = tuple(materialize_semantic_node(child) for child in node.children[1:])
        return LieExpr(vector_components=vec, form=form, coordinates=tuple(node.metadata.get("coordinates", tuple())), metadata=dict(node.metadata.get("extra", {}) or {}))
    if node.kind == "gamma_string":
        from .semantic_ops import GammaStringExpr
        scalar = materialize_semantic_node(node.children[0]) if node.children else sp.Integer(1)
        facs = tuple(int(child.value) for child in node.children[1:] if child.kind == "gamma_index")
        return GammaStringExpr(clifford=node.metadata.get("clifford"), factors=facs, scalar=sp.sympify(scalar), metadata=dict(node.metadata.get("extra", {}) or {}))
    if node.kind == "exterior_form":
        try:
            from .exterior_geometry import ExteriorFormNF
            terms = {}
            for term in node.children:
                if term.kind != "exterior_term":
                    continue
                blade = tuple(int(i) for i in term.value)
                coeff = materialize_semantic_node(term.children[0]) if term.children else sp.Integer(1)
                if coeff != 0:
                    terms[blade] = coeff
            return ExteriorFormNF(dimension=int(node.metadata.get("dimension", 0) or 0), basis_labels=tuple(node.value or ()), terms=terms)
        except Exception:
            return node.value
    if node.kind == "leaf":
        return node.value
    return node.value if node.value is not None else node


def semantic_ir(
    obj: Any,
    *,
    layer: str | None = None,
    dimension: int | sp.Expr | None = None,
    tensor_heads: Any = None,
    free_indices: Any = None,
    dummy_indices: Any = None,
    contraction_pairs: Any = None,
    ordered_factors: Any = None,
    metadata: Mapping[str, Any] | None = None,
    root: SemanticNode | None = None,
) -> TensorSemanticIR:
    md = dict(metadata or {})
    return TensorSemanticIR(
        layer=str(layer or semantic_layer_of(obj)),
        expr=obj,
        dimension=dimension,
        tensor_heads=_stable_tuple(tensor_heads),
        free_indices=_stable_tuple(free_indices),
        dummy_indices=_stable_tuple(dummy_indices),
        contraction_pairs=tuple(tuple(pair) for pair in _stable_tuple(contraction_pairs)),
        ordered_factors=_stable_tuple(ordered_factors),
        metadata=md,
        root=root if root is not None else compile_semantic_node(obj, layer=layer, dimension=dimension),
    )


def semantic_ir_for_object(obj: Any, *, default_layer: str = "abstract", metadata: Mapping[str, Any] | None = None) -> TensorSemanticIR:
    cls = type(obj).__name__
    md = dict(metadata or {})
    root = compile_semantic_node(obj, layer=semantic_layer_of(obj, default=default_layer), dimension=getattr(obj, "dimension", None))
    if cls == "ExteriorFormNF":
        md.setdefault("basis_labels", tuple(getattr(obj, "basis_labels", tuple())))
        ordered = tuple(sorted(tuple(int(i) for i in blade) for blade in getattr(obj, "terms", {}).keys()))
        return semantic_ir(
            obj,
            layer="exterior",
            dimension=int(getattr(obj, "dimension", 0)),
            ordered_factors=ordered,
            metadata=md,
            root=root,
        )
    if cls == "SpinConnectionDef":
        frame = getattr(obj, "frame", None)
        if frame is not None:
            md.setdefault("frame", structural_key(frame))
        md.setdefault("metric_signature", tuple(getattr(obj, "metric_signature", tuple())))
        ordered = tuple(sorted(tuple(int(i) for i in k) for k in getattr(obj, "coefficients", {}).keys()))
        return semantic_ir(
            obj,
            layer="spin_connection",
            dimension=len(getattr(obj, "metric_signature", tuple())) or getattr(getattr(obj, "frame", None), "dimension", None),
            ordered_factors=ordered,
            metadata=md,
            root=root,
        )
    if cls in {"HodgeExpr", "CodifferentialExpr", "InteriorExpr", "LieExpr"}:
        md.setdefault("operator", cls)
        return semantic_ir(
            obj,
            layer="exterior_operator",
            dimension=getattr(getattr(obj, "form", None), "dimension", None),
            ordered_factors=(cls,),
            metadata=md,
            root=root,
        )
    if cls == "GammaStringExpr":
        md.setdefault("operator", cls)
        md.setdefault("clifford", structural_key(getattr(obj, "clifford", None)))
        return semantic_ir(
            obj,
            layer="gamma_string",
            dimension=getattr(getattr(obj, "clifford", None), "dimension", None),
            ordered_factors=tuple(getattr(obj, "factors", tuple())),
            metadata=md,
            root=root,
        )
    if cls == "CliffordAlgebraDef":
        md.setdefault("signature", tuple(getattr(obj, "signature", tuple())))
        md.setdefault("generator_prefix", getattr(obj, "generator_prefix", "gamma"))
        md.setdefault("basis_labels", tuple(getattr(obj, "basis_labels", tuple())))
        return semantic_ir(
            obj,
            layer="clifford",
            dimension=int(getattr(obj, "dimension", 0)),
            ordered_factors=tuple(getattr(obj, "basis_labels", tuple())),
            metadata=md,
            root=root,
        )
    if cls in {"TensorBasis", "TensorFrame"}:
        chart = getattr(obj, "chart", None)
        if chart is not None:
            md.setdefault("chart", structural_key(chart))
        md.setdefault("kind", getattr(obj, "kind", ""))
        md.setdefault("dual_name", getattr(obj, "dual_name", None))
        return semantic_ir(
            obj,
            layer="frame",
            dimension=getattr(obj, "dimension", None),
            ordered_factors=(getattr(obj, "name", ""), getattr(obj, "kind", "")),
            metadata=md,
            root=root,
        )
    return semantic_ir(obj, layer=default_layer, metadata=md, root=root)


def semantic_fingerprint(ir: TensorSemanticIR) -> tuple[Any, ...]:
    if ir.root is not None:
        return (
            "semantic-node-v1",
            semantic_node_fingerprint(normalize_semantic_node(ir.root)),
            ir.layer,
            structural_key(ir.dimension),
        )
    return canonical_expr_fingerprint(
        ir.expr,
        dimension=ir.dimension,
        layer=ir.layer,
        policy=ir.metadata.get("policy") if isinstance(ir.metadata, Mapping) else None,
    )


def canonical_semantic_form(ir: TensorSemanticIR) -> CanonicalSemanticForm:
    key = (
        semantic_fingerprint(ir),
        canonical_sequence_key(ir.tensor_heads),
        canonical_sequence_key(ir.free_indices),
        canonical_sequence_key(ir.dummy_indices),
        tuple(tuple(structural_key(v) for v in pair) for pair in ir.contraction_pairs),
        canonical_sequence_key(ir.ordered_factors),
        canonical_mapping_key(ir.metadata),
    )
    return CanonicalSemanticForm(ir=ir, fingerprint=key[0], key=key)


def semantic_equal(left: CanonicalSemanticForm, right: CanonicalSemanticForm) -> bool:
    return left.key == right.key


def compile_semantic_ir(obj: Any, *, layer: str | None = None, dimension: int | sp.Expr | None = None, metadata: Mapping[str, Any] | None = None) -> TensorSemanticIR:
    return semantic_ir_for_object(obj, default_layer=layer or semantic_layer_of(obj, default="abstract"), metadata=metadata)


def semantic_normalize_object(obj: Any, *, layer: str | None = None, dimension: int | sp.Expr | None = None) -> Any:
    root = compile_semantic_node(obj, layer=layer, dimension=dimension)
    norm = normalize_semantic_node(root)
    return materialize_semantic_node(norm)


def semantic_execute(obj: Any, *, layer: str | None = None, dimension: int | sp.Expr | None = None, metadata: Mapping[str, Any] | None = None) -> CanonicalSemanticForm:
    ir = compile_semantic_ir(obj, layer=layer, dimension=dimension, metadata=metadata)
    norm_root = normalize_semantic_node(ir.root) if ir.root is not None else None
    norm_expr = materialize_semantic_node(norm_root) if norm_root is not None else ir.expr
    norm_ir = semantic_ir(
        norm_expr,
        layer=ir.layer,
        dimension=ir.dimension,
        tensor_heads=ir.tensor_heads,
        free_indices=ir.free_indices,
        dummy_indices=ir.dummy_indices,
        contraction_pairs=ir.contraction_pairs,
        ordered_factors=ir.ordered_factors,
        metadata=ir.metadata,
        root=norm_root,
    )
    return canonical_semantic_form(norm_ir)


__all__ = [
    "SemanticNode",
    "TensorSemanticIR",
    "CanonicalSemanticForm",
    "semantic_layer_of",
    "compile_semantic_node",
    "normalize_semantic_node",
    "materialize_semantic_node",
    "semantic_ir",
    "semantic_ir_for_object",
    "semantic_fingerprint",
    "canonical_semantic_form",
    "semantic_equal",
    "compile_semantic_ir",
    "semantic_normalize_object",
    "semantic_execute",
    "semantic_node_fingerprint",
]
