from __future__ import annotations

from typing import Any

from .fields import ScalarField
from .canonical_keys import structural_key, canonical_expr_fingerprint


def _indexed_provenance_payload(obj, norm, nf, config):
    payload = {
        'input_fingerprint': canonical_expr_fingerprint(obj, layer='indexed_input'),
        'normalized_fingerprint': canonical_expr_fingerprint(norm, layer='indexed_normalized'),
        'term_count': len(getattr(nf, 'terms', tuple())),
        'config': getattr(config, '__dict__', {}),
    }
    central = indexed_expr_canonical_report(obj)
    if central is not None:
        payload['central_tensor_expr_key'] = central.canonical_key
        payload['central_tensor_expr_steps'] = tuple(step.rule for step in central.steps)
    return payload


def indexed_expr_canonical_report(obj: Any, *, registry: Any | None = None):
    """Canonicalize an indexed object through the central TensorExpr engine.

    The classic indexed normal form still owns reconstruction of classic indexed
    objects, but comparison/provenance can now use the shared TensorExpr
    canonical key.  This is the routing shim used while the older indexed
    pipeline is gradually retired.
    """
    try:
        from .tensor_expr_canonicalization import canonicalize_tensor_expr
        return canonicalize_tensor_expr(obj, registry=registry)
    except Exception:
        return None


def indexed(tensor, *idx):
    from . import tensor_indices as ti
    if len(idx) != len(tensor.variance_spec):
        return ti.IndexedTensor(tensor, tuple(idx))
    adapted = tensor
    requested = list(idx)
    if any(i.variance != k for i, k in zip(requested, adapted.variance_spec)):
        tf = adapted.to_tensor_field()
        changed = False
        for pos, (i, k) in enumerate(zip(requested, adapted.variance_spec)):
            if i.variance == k:
                continue
            if k == 'u' and i.variance == 'l':
                tf = tf.lower_index(pos)
                changed = True
            elif k == 'l' and i.variance == 'u':
                tf = tf.raise_index(pos)
                changed = True
        if changed:
            adapted = ti.TensorObject.from_tensor_field(tf, name=adapted.name, symmetry_metadata=dict(adapted.symmetry_metadata))
    return ti.IndexedTensor(adapted, tuple(requested))


def normalize_indexed_expression(obj: Any, config: Any | None = None) -> Any:
    from . import tensor_indices as ti
    config = ti._resolve_indexed_config(config, ti.IndexedNormalizationConfig)
    try:
        ti._register_tnf_helper("normalize_indexed_expression", "boundary_only", False, False, "public normalize entry delegates to TNF")
    except Exception:
        pass
    nf = ti.to_indexed_tensor_form(obj, config=config)
    ti._LAST_TNF_DISPATCHER_REPORT = ti.TNFDispatcherReport(parsed_from_boundary=True, reduced_in_nf=True, reconstructed_at_boundary=True)
    ti._LAST_NORMALIZATION_DIAGNOSTICS = ti.NormalizationDiagnostics(
        used_cache=False, used_optimizer_prepass=True, used_component_expansion=False,
        passes=1, tier=getattr(config, "tier", 2), contraction_plan_cost=0, removed_zero_terms=0, removed_identity_terms=0,
    )
    try:
        return ti._authoritative_tnf_to_expr(nf)
    except Exception:
        if isinstance(obj, ti.IndexedTensor):
            return obj.canonicalize().contract_repeated()
        if isinstance(obj, ti.IndexedTensorExpr):
            try:
                return obj.simplify()
            except Exception:
                return obj
        return obj


def canonicalize_indexed_expression(obj: Any, config: Any | None = None) -> Any:
    from . import tensor_indices as ti
    try:
        ti._register_tnf_helper("canonicalize_indexed_expression", "boundary_only", False, False, "public canonicalization delegates to TNF")
    except Exception:
        pass
    try:
        return normalize_indexed_expression(obj, config=config)
    except (ti.BundleCompatibilityError, ValueError):
        raise
    except Exception:
        if isinstance(obj, ti.IndexedTensor):
            return obj.canonicalize().contract_repeated()
        if isinstance(obj, ti.IndexedTensorExpr):
            try:
                return obj.simplify()
            except Exception:
                return ti.rewrite_fixed_point(obj)
        return obj


def indexed_canonical_report(obj: Any, config: Any | None = None):
    from . import tensor_indices as ti
    config = ti._resolve_indexed_config(config, ti.IndexedNormalizationConfig)
    nf = ti.to_indexed_tensor_form(obj, config=config)
    norm = ti._authoritative_tnf_to_expr(nf)
    tensor_kinds = []
    symmetry_tags = set()
    for term in nf.terms:
        for factor in term.factors:
            tensor_kinds.append(factor.kind)
            if factor.symmetry:
                symmetry_tags.add(tuple(factor.symmetry))
    renorm = canonicalize_indexed_expression(norm, config=config)
    return ti.IndexedCanonicalizationReport(
        normalized=norm,
        free_signature=tuple(t.free_signature for t in nf.terms),
        bundle_signature=tuple(t.bundle_signature for t in nf.terms),
        tensor_kinds=tuple(tensor_kinds),
        symmetry_tags=tuple(sorted(symmetry_tags)),
        structural_signature=ti.indexed_signature(norm, config=config),
        idempotent=structural_key(renorm) == structural_key(norm),
        provenance=_indexed_provenance_payload(obj, norm, nf, config),
    )


def indexed_equivalence_report(obj1: Any, obj2: Any, config: Any | None = None):
    left_central = indexed_expr_canonical_report(obj1)
    right_central = indexed_expr_canonical_report(obj2)
    if left_central is not None and right_central is not None:
        return {
            'equal': left_central.canonical_key == right_central.canonical_key,
            'left_signature': left_central.canonical_key,
            'right_signature': right_central.canonical_key,
            'left_provenance': {'central_tensor_expr_key': left_central.canonical_key},
            'right_provenance': {'central_tensor_expr_key': right_central.canonical_key},
        }
    from . import tensor_indices as ti
    config = ti._resolve_indexed_config(config, ti.IndexedNormalizationConfig)
    left_report = indexed_canonical_report(obj1, config=config)
    right_report = indexed_canonical_report(obj2, config=config)
    equal = left_report.structural_signature == right_report.structural_signature
    return {
        'equal': equal,
        'left_signature': left_report.structural_signature,
        'right_signature': right_report.structural_signature,
        'left_provenance': left_report.provenance,
        'right_provenance': right_report.provenance,
    }


def indexed_equivalent(obj1: Any, obj2: Any, config: Any | None = None) -> bool:
    central = indexed_equivalence_report(obj1, obj2, config=config)
    if 'central_tensor_expr_key' in central.get('left_provenance', {}) and bool(central['equal']):
        return True
    from . import tensor_indices as ti
    try:
        ti._register_tnf_helper("indexed_equivalent", "boundary_only", False, False, "public equality compares central TensorExpr keys before classic fallback")
    except Exception:
        pass
    try:
        left = canonicalize_indexed_expression(ti._normalize_basis_for_indexed(obj1), config=config)
        right = canonicalize_indexed_expression(ti._normalize_basis_for_indexed(obj2), config=config)
        if isinstance(left, ScalarField) or isinstance(right, ScalarField):
            return isinstance(left, ScalarField) and isinstance(right, ScalarField) and ti.is_zero(ti._configured_simplify_expr(left.expr - right.expr, config), mode=ti._decision_mode_from_config(config))
        return structural_key(left) == structural_key(right)
    except Exception:
        pass
    # Last-resort classic comparison: normalize to the indexed tensor-form
    # signature.  This deliberately avoids broad object simplification while
    # still treating grouping differences and reconstructed identity tensors as
    # equal when their canonical tensor-form signatures agree.
    try:
        return ti.indexed_signature(obj1, config=config) == ti.indexed_signature(obj2, config=config)
    except Exception:
        return False


def indexed_equal(obj1: Any, obj2: Any, config: Any | None = None) -> bool:
    return indexed_equivalent(obj1, obj2, config=config)


def stronger_indexed_equal(left: Any, right: Any, config: Any | None = None) -> bool:
    return indexed_equivalent(left, right, config=config)
