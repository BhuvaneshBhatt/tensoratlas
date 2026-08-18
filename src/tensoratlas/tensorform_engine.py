from __future__ import annotations
from dataclasses import dataclass
import sympy as sp
from .symbolic_decision import is_zero, light_simplify, canonical_simplify
from .simplification_policy import cheap_simplify, normal_simplify, strong_simplify
from .indexed_config import heuristic_enabled, resolve_indexed_config
from .indexed_reconstruct import tnf_term_to_expr
from .canonical_keys import structural_key, factor_key as canonical_factor_key, term_group_key as canonical_term_group_key, term_sort_key as canonical_term_sort_key
from .tensor_core import TensorObject
from .tensorform_types import IndexedTensorForm, TensorFormTerm, NormalizationDiagnostics
from .fields import ScalarField
from .normal_forms import tnf_build_array, tnf_iter_indices

def _ti():
    from . import tensor_indices as ti
    return ti
@dataclass(frozen=True)
class IndexedFactor:
    factor: object
@dataclass(frozen=True)
class TNFFactorAtom:
    kind: str; name: str; variance_spec: tuple; basis_names: tuple; typed_indices: tuple; symmetry: tuple; payload: object | None = None
@dataclass(frozen=True)
class IndexedNormalFactor:
    kind: str; name: str; variance_spec: tuple; tensor_space_sig: tuple; basis_names: tuple; typed_slots: tuple; symmetry: tuple; role: str; dimension_hint: object | None = None; orientation_hint: object | None = None; frame_hint: object | None = None; bundle_hint: tuple = tuple(); chart_hint: object | None = None; metric_hint: object | None = None; parameter_hint: tuple = tuple(); young_hint: tuple = tuple(); signature_hint: object | None = None; special_signature: tuple = tuple(); variance_pattern_hint: tuple = tuple(); contraction_hint: tuple = tuple(); reduction_class_hint: tuple = tuple(); symmetry_class_hint: tuple = tuple(); rank_hint: object | None = None
def _authoritative_tnf_from_factor(f):
    if isinstance(f, IndexedNormalFactor): return f
    return IndexedNormalFactor(kind=getattr(f,"kind","generic"), name=getattr(f,"name","<anon>"), variance_spec=getattr(f,"variance_spec",tuple()), tensor_space_sig=getattr(f,"tensor_space_sig",tuple()), basis_names=getattr(f,"basis_names",tuple()), typed_slots=getattr(f,"typed_slots",tuple()), symmetry=getattr(f,"symmetry",tuple()), role=getattr(f,"role","generic"), dimension_hint=getattr(f,"dimension_hint",None), orientation_hint=getattr(f,"orientation_hint",None), frame_hint=getattr(f,"frame_hint",None), bundle_hint=getattr(f,"bundle_hint",tuple()), chart_hint=getattr(f,"chart_hint",None), metric_hint=getattr(f,"metric_hint",None), parameter_hint=getattr(f,"parameter_hint",tuple()), young_hint=getattr(f,"young_hint",tuple()), signature_hint=getattr(f,"signature_hint",None), special_signature=getattr(f,"special_signature",tuple()), variance_pattern_hint=getattr(f,"variance_pattern_hint",tuple()), contraction_hint=getattr(f,"contraction_hint",tuple()), reduction_class_hint=getattr(f,"reduction_class_hint",tuple()), symmetry_class_hint=getattr(f,"symmetry_class_hint",tuple()), rank_hint=getattr(f,"rank_hint",None))
def _authoritative_tnf_special_tensor_scale(tensor: TensorObject, kind: str):
    from .tensor_algebra import metric_tensor, permutation_tensor
    if kind=="delta":
        if len(tensor.variance_spec)!=2: return None
        ref=TensorObject(tensor.chart, tnf_build_array((tensor.chart.dimension,tensor.chart.dimension), lambda idx: sp.Integer(1) if idx[0]==idx[1] else sp.Integer(0)), tensor.variance_spec, tensor.slot_bases, name=tensor.name, symmetry_metadata=dict(getattr(tensor,"symmetry_metadata",{}) or {}))
    elif kind=="metric_ll":
        if tensor.variance_spec!="ll": return None
        ref=TensorObject.from_tensor_field(metric_tensor(tensor.chart,"ll"))
    elif kind=="metric_uu":
        if tensor.variance_spec!="uu": return None
        ref=TensorObject.from_tensor_field(metric_tensor(tensor.chart,"uu"))
    elif kind=="epsilon":
        if len(tensor.variance_spec)!=tensor.chart.dimension: return None
        ref=TensorObject.from_tensor_field(permutation_tensor(tensor.chart,tensor.variance_spec))
    else: return None
    scale=None
    for idx in tnf_iter_indices(tensor.components.shape):
        ref_entry=ref.components[idx]; cur_entry=tensor.components[idx]
        if is_zero(ref_entry):
            if not is_zero(light_simplify(cur_entry)): return None
            continue
        candidate=normal_simplify(cur_entry/ref_entry)
        scale=candidate if scale is None else normal_simplify(scale)
        if not is_zero(light_simplify(candidate-scale)): return None
    return sp.Integer(1) if scale is None else normal_simplify(scale)
def _tensorform_normalized_symmetry(symmetry_metadata):
    metadata=symmetry_metadata or {}; out=[]
    for key,value in metadata.items(): out.append((key, tuple(sorted(tuple(sorted(group)) for group in value))))
    return tuple(sorted(out))
def _tensorform_space_name(bundle):
    try:
        if hasattr(bundle,"chart_name"): return bundle.chart_name
        if hasattr(bundle,"name"): return bundle.name
        return str(bundle)
    except Exception: return str(bundle)
def _tensorform_space_dim(bundle):
    try:
        if hasattr(bundle,"dimension"): return bundle.dimension
        if hasattr(bundle,"dim"): return bundle.dim
    except Exception: pass
    return None
def _tensorform_slot_signature(idx): return (_tensorform_space_name(idx.bundle), _tensorform_space_dim(idx.bundle), idx.variance)
def _tensorform_basis_bundle_hint(bases):
    out=[]
    for basis in bases:
        metadata=getattr(basis,"metadata",{}) or {}; bundle=metadata.get("bundle",None); out.append((_tensorform_space_name(bundle), _tensorform_space_dim(bundle), getattr(basis,"name",str(basis))))
    return tuple(out)
def _tensorform_canonical_slots(indices,symmetry):
    slots=[_tensorform_slot_signature(i) for i in indices]; symm=dict(symmetry) if isinstance(symmetry,(tuple,list)) else {}
    for group_name in ("symmetric","antisymmetric"):
        for group in symm.get(group_name, tuple()):
            group=tuple(group); ordered=sorted((slots[s] for s in group), key=lambda x:(x[0],x[1],x[2]))
            for pos,val in zip(group,ordered): slots[pos]=val
    return tuple(slots)
def _tensorform_factor_sign_from_indices(indices,symmetry):
    ti=_ti(); sign=1; symm=dict(symmetry) if isinstance(symmetry,(tuple,list)) else {}; raw=[_tensorform_slot_signature(i) for i in indices]
    for group in symm.get("antisymmetric", tuple()):
        group=tuple(group); cur=[raw[s] for s in group]; order=sorted(range(len(cur)), key=lambda k:(cur[k][0],cur[k][1],cur[k][2])); sign *= ti._perm_parity(order)
    return sign
def _tensorform_special_signature_from_leaf(leaf):
    ti=_ti(); kind=ti._leaf_tensor_kind(leaf)
    return (kind, tuple(_tensorform_slot_signature(i) for i in leaf.indices)) if kind in {"delta","identity","metric_ll","metric_uu","epsilon"} else tuple()
def _tensorform_symmetry_class(symmetry):
    symm=dict(symmetry) if isinstance(symmetry,(tuple,list)) else {}; return (tuple(sorted(tuple(g) for g in symm.get("symmetric",tuple()))), tuple(sorted(tuple(g) for g in symm.get("antisymmetric",tuple()))), tuple(sorted(tuple(g) for g in symm.get("young",tuple()))) if "young" in symm else tuple())
def _tensorform_leaf_to_indexed_normal_factor(leaf):
    ti=_ti(); leaf=ti._symmetry_aware_leaf(leaf); tensor=leaf.tensor; chart=getattr(tensor,"chart",None); symmetry=_tensorform_normalized_symmetry(getattr(tensor,"symmetry_metadata",{}) or {}); sign=sp.Integer(_tensorform_factor_sign_from_indices(leaf.indices,symmetry)); kind=ti._leaf_tensor_kind(leaf); common_kwargs=dict(variance_spec=tuple(tensor.variance_spec), tensor_space_sig=tuple(_tensorform_slot_signature(i) for i in leaf.indices), basis_names=tuple(getattr(b,"name",str(b)) for b in tensor.slot_bases), typed_slots=_tensorform_canonical_slots(leaf.indices,symmetry), symmetry=symmetry, dimension_hint=getattr(chart,"dimension",None), orientation_hint=None, frame_hint=tuple(getattr(b,"name",str(b)) for b in tensor.slot_bases), bundle_hint=_tensorform_basis_bundle_hint(tensor.slot_bases), chart_hint=getattr(chart,"chart_name",None), metric_hint=getattr(chart,"metric_name",None), parameter_hint=tuple(getattr(chart,"parameter_symbols",lambda: tuple())()) if chart is not None and hasattr(chart,"parameter_symbols") else tuple(), young_hint=tuple(sorted((k, tuple(v)) for k,v in (getattr(tensor,"symmetry_metadata",{}) or {}).items() if "young" in k.lower())), signature_hint=getattr(chart,"signature",None) if chart is not None else None, variance_pattern_hint=tuple(tensor.variance_spec), contraction_hint=tuple(sorted(_tensorform_canonical_slots(leaf.indices,symmetry))), symmetry_class_hint=_tensorform_symmetry_class(symmetry), rank_hint=len(leaf.indices)); factor=IndexedNormalFactor(kind=kind,name=tensor.name or "<anon>",role=kind if kind is not None else "generic",special_signature=_tensorform_special_signature_from_leaf(leaf),reduction_class_hint=tuple(),**common_kwargs)
    if kind is None: kind=ti._classify_special_tensor(tensor)
    probe_kinds=[kind] if kind is not None else ["delta","metric_ll","metric_uu","epsilon"]
    for probe_kind in probe_kinds:
        if probe_kind not in {"delta","metric_ll","metric_uu","epsilon"}: continue
        scaled_coeff=_authoritative_tnf_special_tensor_scale(tensor,probe_kind)
        if scaled_coeff is None: continue
        sign=canonical_simplify(sign*scaled_coeff); reduction_class_hint={"delta":("delta_like",),"metric_ll":("metric_like",),"metric_uu":("metric_like",),"epsilon":("epsilon_like",len(leaf.indices))}[probe_kind]; canon_name=("δ" if probe_kind=="delta" else ("eps" if probe_kind=="epsilon" else "g"))
        factor=IndexedNormalFactor(kind=probe_kind,name=canon_name,role=probe_kind,special_signature=(probe_kind,tuple(_tensorform_slot_signature(i) for i in leaf.indices)),reduction_class_hint=reduction_class_hint,**common_kwargs); break
    return factor, sign
def _authoritative_tnf_leaf_to_factor_and_coeff(leaf): return _tensorform_leaf_to_indexed_normal_factor(leaf)
def _authoritative_tnf_zero(): return IndexedTensorForm(tuple())
def _authoritative_tnf_one(): return IndexedTensorForm((TensorFormTerm(sp.Integer(1), tuple(), tuple(), tuple()),))
def _authoritative_tnf_from_scalar(expr): return IndexedTensorForm((TensorFormTerm(normal_simplify(expr), tuple(), tuple(), tuple()),))
def _authoritative_tnf_factor_zero_from_symmetry(f):
    f=_authoritative_tnf_from_factor(f); symm=dict(f.symmetry) if isinstance(f.symmetry,(tuple,list)) else {}
    for group in symm.get("antisymmetric", tuple()):
        seen=[]
        for pos in tuple(group):
            slot=f.typed_slots[pos]
            if slot in seen: return True
            seen.append(slot)
    return False
def _authoritative_tnf_from_leaf(leaf):
    ti=_ti();
    if ti._detect_obvious_zero_leaf(leaf): return _authoritative_tnf_zero()
    factor,coeff=_authoritative_tnf_leaf_to_factor_and_coeff(leaf)
    if _authoritative_tnf_factor_zero_from_symmetry(factor) or is_zero(coeff): return _authoritative_tnf_zero()
    term=TensorFormTerm(normal_simplify(coeff), (factor,), tuple(sorted(_tensorform_slot_signature(i) for i in leaf.indices)), tuple(sorted((_tensorform_space_name(i.bundle), _tensorform_space_dim(i.bundle)) for i in leaf.indices if i.bundle is not None)))
    return IndexedTensorForm((term,))
def _authoritative_tnf_term_group_key(term):
    return canonical_term_group_key(term)

def _authoritative_tnf_term_key(term):
    return canonical_term_sort_key(term)
def _authoritative_tnf_collect(nf):
    out={}
    for t in nf.terms:
        if is_zero(t.scalar):
            continue
        key = _authoritative_tnf_term_group_key(t)
        if key in out:
            scalar, facs, free_sig, bundle_sig = out[key]
            out[key] = (normal_simplify(scalar + t.scalar), facs, free_sig, bundle_sig)
        else:
            out[key] = (normal_simplify(t.scalar), tuple(sorted(t.factors, key=_authoritative_tnf_factor_key)), tuple(t.free_signature), tuple(t.bundle_signature))
    terms=[]
    for scalar, facs, free_sig, bundle_sig in out.values():
        if not is_zero(scalar):
            terms.append(TensorFormTerm(canonical_simplify(scalar), facs, free_sig, bundle_sig))
    return IndexedTensorForm(tuple(sorted(terms, key=_authoritative_tnf_term_key)))
def _authoritative_tnf_add(nf1,nf2): return _authoritative_tnf_collect(IndexedTensorForm(tuple(nf1.terms)+tuple(nf2.terms)))
def _authoritative_tnf_mul(nf1,nf2):
    terms=[]
    for a in nf1.terms:
        for b in nf2.terms:
            scalar=normal_simplify(a.scalar*b.scalar)
            if is_zero(scalar): continue
            terms.append(TensorFormTerm(scalar, tuple(sorted(tuple(a.factors)+tuple(b.factors), key=_authoritative_tnf_factor_key)), tuple(sorted(tuple(a.free_signature)+tuple(b.free_signature))), tuple(sorted(tuple(a.bundle_signature)+tuple(b.bundle_signature)))))
    return _authoritative_tnf_collect(IndexedTensorForm(tuple(terms)))
def _authoritative_tnf_rebuild_factor(f, typed_slots=None, symmetry=None, kind=None):
    ff=_authoritative_tnf_from_factor(f); data=ff.__dict__.copy()
    if typed_slots is not None: data["typed_slots"]=tuple(typed_slots)
    if symmetry is not None: data["symmetry"]=tuple(symmetry)
    if kind is not None: data["kind"]=kind; data["role"]=kind
    return IndexedNormalFactor(**data)
def _authoritative_tnf_factor_is_delta(f): return _authoritative_tnf_from_factor(f).kind in {"delta","identity"}
def _authoritative_tnf_factor_is_metric_up(f): return _authoritative_tnf_from_factor(f).kind=="metric_uu"
def _authoritative_tnf_factor_is_metric_down(f): return _authoritative_tnf_from_factor(f).kind=="metric_ll"
def _authoritative_tnf_factor_is_epsilon(f): return _authoritative_tnf_from_factor(f).kind=="epsilon"
def _authoritative_tnf_factor_is_gdelta(f): return _authoritative_tnf_from_factor(f).kind=="gdelta"
def _authoritative_tnf_replace_slot(slots, oldslot, newslot): return tuple(newslot if s==oldslot else s for s in slots)
def _authoritative_tnf_choose_shared_slot(a_slots,b_slots):
    for sa in a_slots:
        for sb in b_slots:
            if sa[0]==sb[0] and sa[1]==sb[1] and sa[2]!=sb[2]: return sa,sb
    return None
def _authoritative_tnf_gdelta_from_slots(up_slots, down_slots, dim=None): return IndexedNormalFactor(kind="gdelta", name="Δ", variance_spec=tuple("u"*len(up_slots)+"l"*len(down_slots)), tensor_space_sig=tuple(up_slots)+tuple(down_slots), basis_names=tuple(), typed_slots=tuple(up_slots)+tuple(down_slots), symmetry=tuple(), role="gdelta", dimension_hint=dim, special_signature=("gdelta",tuple(up_slots),tuple(down_slots)), reduction_class_hint=("gdelta_like",len(up_slots)), rank_hint=len(up_slots)+len(down_slots))
def _authoritative_tnf_apply_gdelta_to_factor(gd,f):
    gd=_authoritative_tnf_from_factor(gd); f=_authoritative_tnf_from_factor(f); ups=[s for s in gd.typed_slots if len(s)>=3 and s[2]=='u']; downs=[s for s in gd.typed_slots if len(s)>=3 and s[2]=='l']; slots=list(f.typed_slots)
    for down,up in zip(downs,ups): slots=[up if s==down else s for s in slots]
    return _authoritative_tnf_rebuild_factor(f, typed_slots=tuple(slots))
def _authoritative_tnf_normalize_symmetry_slots(f):
    ff=_authoritative_tnf_from_factor(f); symm=dict(ff.symmetry) if isinstance(ff.symmetry,(tuple,list)) else {}; slots=list(ff.typed_slots); sign=1; ti=_ti()
    for group_name in ("symmetric","antisymmetric"):
        for group in symm.get(group_name, tuple()):
            group=tuple(group); cur=[slots[p] for p in group]; order=sorted(range(len(cur)), key=lambda k:cur[k]); ordered=[cur[k] for k in order]
            if group_name=='antisymmetric': sign *= ti._perm_parity(order)
            for pos,val in zip(group,ordered): slots[pos]=val
    return sign, _authoritative_tnf_rebuild_factor(ff, typed_slots=tuple(slots))
def _authoritative_tnf_reduce_symmetry_only(factors, scalar):
    out=[]
    for f in factors:
        sign,newf=_authoritative_tnf_normalize_symmetry_slots(f); scalar=normal_simplify(scalar*sign)
        if _authoritative_tnf_factor_zero_from_symmetry(newf) or is_zero(scalar): return tuple(), sp.Integer(0)
        out.append(newf)
    return tuple(out), scalar
def _authoritative_tnf_reduce_special_factors(factors, scalar):
    factors=tuple(_authoritative_tnf_from_factor(f) for f in factors); factors,scalar=_authoritative_tnf_reduce_symmetry_only(factors,scalar)
    if is_zero(scalar): return tuple(), sp.Integer(0)
    changed=True
    while changed:
        changed=False
        for i,a in enumerate(list(factors)):
            for j,b in enumerate(list(factors)):
                if i>=j: continue
                aa,bb=_authoritative_tnf_from_factor(a),_authoritative_tnf_from_factor(b)
                if _authoritative_tnf_factor_is_delta(aa) and _authoritative_tnf_factor_is_delta(bb): continue
                pair=None
                if (_authoritative_tnf_factor_is_delta(aa) and _authoritative_tnf_factor_is_metric_up(bb)) or (_authoritative_tnf_factor_is_metric_down(aa) and _authoritative_tnf_factor_is_metric_up(bb)) or (_authoritative_tnf_factor_is_metric_down(aa) and _authoritative_tnf_factor_is_delta(bb)): pair=(i,j,aa,bb)
                if pair is None: continue
                i0,j0,fa,fb=pair; shared=_authoritative_tnf_choose_shared_slot(fa.typed_slots,fb.typed_slots)
                if shared is None: continue
                sa,sb=shared; rem_a=[s for s in fa.typed_slots if s!=sa]; rem_b=[s for s in fb.typed_slots if s!=sb]
                if _authoritative_tnf_factor_is_metric_up(fa) or _authoritative_tnf_factor_is_metric_up(fb):
                    up=rem_a[0] if _authoritative_tnf_factor_is_metric_up(fa) else rem_b[0]; down=rem_b[0] if _authoritative_tnf_factor_is_metric_up(fa) else rem_a[0]; repl=_authoritative_tnf_gdelta_from_slots((up,),(down,),fa.dimension_hint or fb.dimension_hint); factors=tuple(f for k,f in enumerate(factors) if k not in {i0,j0})+(repl,); changed=True; break
            if changed: break
        if changed: continue
        for i,a in enumerate(list(factors)):
            for j,b in enumerate(list(factors)):
                if i>=j: continue
                aa,bb=_authoritative_tnf_from_factor(a),_authoritative_tnf_from_factor(b)
                if not (_authoritative_tnf_factor_is_epsilon(aa) and _authoritative_tnf_factor_is_epsilon(bb)): continue
                if len(aa.typed_slots)!=len(bb.typed_slots): continue
                if aa.typed_slots==bb.typed_slots:
                    dim=aa.dimension_hint or len(aa.typed_slots); scalar=normal_simplify(scalar*sp.factorial(dim)); factors=tuple(f for k,f in enumerate(factors) if k not in {i,j}); changed=True; break
                shared=[]; free_a=[]; free_b=list(range(len(bb.typed_slots)))
                for pa,sa in enumerate(aa.typed_slots):
                    found=False
                    for pb in list(free_b):
                        sb=bb.typed_slots[pb]
                        if sa[0]==sb[0] and sa[1]==sb[1] and sa[2]!=sb[2]: shared.append((pa,pb)); free_b.remove(pb); found=True; break
                    if not found: free_a.append(pa)
                if shared and len(free_a)==len(free_b):
                    up_slots=tuple(bb.typed_slots[p] for p in free_b); down_slots=tuple(aa.typed_slots[p] for p in free_a); scalar=normal_simplify(scalar*sp.factorial(len(shared))); factors=tuple(f for k,f in enumerate(factors) if k not in {i,j}) + (_authoritative_tnf_gdelta_from_slots(up_slots,down_slots, aa.dimension_hint or bb.dimension_hint),); changed=True; break
            if changed: break
        if changed: continue
        for i,gd in enumerate(list(factors)):
            gg=_authoritative_tnf_from_factor(gd)
            if not _authoritative_tnf_factor_is_gdelta(gg): continue
            factors=tuple(_authoritative_tnf_apply_gdelta_to_factor(gg,f) for j,f in enumerate(factors) if i!=j); changed=True; break
    return tuple(sorted((_authoritative_tnf_from_factor(f) for f in factors), key=_authoritative_tnf_factor_key)), scalar
def _authoritative_tnf_reduce_term(term):
    scalar=normal_simplify(term.scalar)
    if is_zero(scalar): return _authoritative_tnf_zero()
    factors,scalar=_authoritative_tnf_reduce_special_factors(term.factors, scalar)
    if is_zero(scalar): return _authoritative_tnf_zero()
    return _authoritative_tnf_collect(IndexedTensorForm((TensorFormTerm(scalar, factors, tuple(sorted(term.free_signature,key=str)), tuple(sorted(term.bundle_signature,key=str))),)))
def _authoritative_tnf_reduce(nf):
    out=_authoritative_tnf_zero()
    for t in nf.terms: out=_authoritative_tnf_add(out, _authoritative_tnf_reduce_term(t))
    return _authoritative_tnf_collect(out)
def _authoritative_tnf_expr_to_tnf(obj):
    ti=_ti(); obj=ti._unwrap_layer(obj)
    key=("authoritative_nf", structural_key(obj))
    if key in ti._NORMAL_FORM_CACHE: return ti._NORMAL_FORM_CACHE[key]
    if isinstance(obj, ScalarField): nf=_authoritative_tnf_from_scalar(obj.expr); ti._NORMAL_FORM_CACHE[key]=nf; return nf
    if isinstance(obj, ti.IndexedTensor): nf=_authoritative_tnf_from_leaf(obj); ti._NORMAL_FORM_CACHE[key]=nf; return nf
    if not isinstance(obj, ti.IndexedTensorExpr):
        raw_label=(("raw", structural_key(obj)),)
        nf=IndexedTensorForm((TensorFormTerm(sp.Integer(1), raw_label, tuple(), tuple()),)); ti._NORMAL_FORM_CACHE[key]=nf; return nf
    if obj.op=="tensor": nf=_authoritative_tnf_expr_to_tnf(obj.args[0]); ti._NORMAL_FORM_CACHE[key]=nf; return nf
    if obj.op=="add":
        nf=_authoritative_tnf_zero()
        for a in ti._flatten_add(obj): nf=_authoritative_tnf_add(nf,_authoritative_tnf_expr_to_tnf(a))
        nf=_authoritative_tnf_reduce(nf); ti._NORMAL_FORM_CACHE[key]=nf; return nf
    if obj.op=="tensor_product":
        nf=_authoritative_tnf_one(); scalar=sp.Integer(1)
        for a in ti._flatten_product(obj):
            raw=ti._unwrap_layer(a)
            if isinstance(raw, ScalarField): scalar*=raw.expr
            else: nf=_authoritative_tnf_mul(nf,_authoritative_tnf_expr_to_tnf(raw))
        if scalar!=1: nf=_authoritative_tnf_mul(nf,_authoritative_tnf_from_scalar(scalar))
        nf=_authoritative_tnf_reduce(nf); ti._NORMAL_FORM_CACHE[key]=nf; return nf
    raw_label=(("raw", structural_key(obj)),)
    nf=IndexedTensorForm((TensorFormTerm(sp.Integer(1), raw_label, tuple(), tuple()),)); ti._NORMAL_FORM_CACHE[key]=nf; return nf
def _authoritative_tnf_chart_from_factor(f):
    from .api import coordinate_chart
    dim=getattr(f,"dimension_hint",None)
    if dim is None: dims=[slot[1] for slot in getattr(f,"typed_slots",tuple()) if isinstance(slot,tuple) and len(slot)>1 and isinstance(slot[1],int)]; dim=dims[0] if dims else 1
    metric_name=getattr(f,"metric_hint",None) or "Euclidean"; chart_name=getattr(f,"chart_hint",None) or "Cartesian"
    try: return coordinate_chart(str(metric_name),str(chart_name),int(dim))
    except Exception: return coordinate_chart("Euclidean","Cartesian",int(dim))
def _authoritative_tnf_symbolic_tensor_components(name, variance_spec, dim):
    base=name or "T"; shape=(dim,)*len(variance_spec)
    return tnf_build_array(tuple(), lambda _: sp.Symbol(base)) if not shape else tnf_build_array(shape, lambda idx: sp.Symbol(base+"_"+"_".join(str(i) for i in idx)))
def _authoritative_tnf_tensor_from_factor(f):
    from .basis import tangent_basis,cotangent_basis
    from .tensor_algebra import metric_tensor, permutation_tensor
    f=_authoritative_tnf_from_factor(f); chart=_authoritative_tnf_chart_from_factor(f); dim=chart.dimension; variance_spec=''.join(f.variance_spec) if isinstance(f.variance_spec,tuple) else str(f.variance_spec); kind=getattr(f,"kind","generic"); name=getattr(f,"name",None) or "T"; symmetry_metadata={}; sym=getattr(f,"symmetry",tuple())
    if isinstance(sym,tuple):
        try: symmetry_metadata={k: tuple(tuple(g) for g in v) for k,v in sym}
        except Exception: symmetry_metadata={}
    if kind in {"delta","identity"} and len(variance_spec)==2:
        components=tnf_build_array((dim,dim), lambda idx: sp.Integer(1) if idx[0]==idx[1] else sp.Integer(0)); slot_bases=tuple(tangent_basis(chart) if v=='u' else cotangent_basis(chart) for v in variance_spec); return TensorObject(chart,components,variance_spec,slot_bases,name=name or "δ",symmetry_metadata=symmetry_metadata)
    if kind=="metric_ll" and variance_spec=="ll": return TensorObject.from_tensor_field(metric_tensor(chart,"ll"), name=name or "g", symmetry_metadata=symmetry_metadata)
    if kind=="metric_uu" and variance_spec=="uu": return TensorObject.from_tensor_field(metric_tensor(chart,"uu"), name=name or "g", symmetry_metadata=symmetry_metadata)
    if kind=="epsilon": return TensorObject.from_tensor_field(permutation_tensor(chart,variance_spec or "lll"), name=name or "eps", symmetry_metadata=symmetry_metadata)
    components=_authoritative_tnf_symbolic_tensor_components(name,variance_spec,dim); slot_bases=tuple(tangent_basis(chart) if v=='u' else cotangent_basis(chart) for v in variance_spec); return TensorObject(chart,components,variance_spec,slot_bases,name=name,symmetry_metadata=symmetry_metadata)
def _authoritative_tnf_term_to_expr(term):
    ti = _ti()
    return tnf_term_to_expr(
        term,
        factor_converter=lambda f: _authoritative_tnf_tensor_from_factor(_authoritative_tnf_from_factor(f)),
        tensor_index_cls=ti.TensorIndex,
        indexed_tensor_cls=ti.IndexedTensor,
        indexed_expr_cls=ti.IndexedTensorExpr,
        scalar_field_cls=ScalarField,
        scalar_cleanup=normal_simplify,
    )

def _authoritative_tnf_to_expr(nf):
    ti=_ti()
    if not nf.terms: return ScalarField(None, sp.Integer(0))
    if len(nf.terms)==1: return _authoritative_tnf_term_to_expr(nf.terms[0])
    terms=[_authoritative_tnf_term_to_expr(t) for t in nf.terms]; expr=terms[0]
    for term in terms[1:]: expr=ti.IndexedTensorExpr("add", (expr,term))
    return expr
def _authoritative_tnf_factor_key(f):
    return canonical_factor_key(_authoritative_tnf_from_factor(f))

def _normal_form_key_safe_key_atom(x):
    return structural_key(x)
def _normalize_indexed_once(obj, config):
    ti=_ti(); config = resolve_indexed_config(config, ti.IndexedNormalizationConfig); cur=ti._unwrap_layer(obj); tier=ti._normalization_tier(config); pre=ti.optimizer_report(cur)
    if heuristic_enabled(config):
        cur=ti._unwrap_layer(ti.optimizer_prepass(cur))
    if getattr(config,"strengthen_bundles",True): cur=ti.strengthen_index_bundles(cur)
    if getattr(config,"normalize_basis",True): cur=ti._normalize_basis_for_indexed(cur)
    if getattr(config,"validate_bundles",True) or getattr(config,"validate_indices",True): cur=ti._validate_indexed_object(cur)
    nf=_authoritative_tnf_expr_to_tnf(cur); nf=ti._combine_like_terms_nf(nf)
    if tier>=1:
        nf=IndexedTensorForm(tuple(TensorFormTerm(sp.factor(normal_simplify(t.scalar)), t.factors, t.free_signature, t.bundle_signature) for t in nf.terms if not is_zero(normal_simplify(t.scalar))))
        nf=ti._combine_like_terms_nf(nf)
    if tier>=2: nf=ti._combine_like_terms_nf(nf)
    cur=_authoritative_tnf_to_expr(nf); ti._LAST_NORMALIZATION_DIAGNOSTICS=NormalizationDiagnostics(False,True,False,1,tier,0,pre.removed_zero_terms,pre.removed_identity_terms); return cur


def tensor_form_expr_canonical_report(obj, *, registry=None):
    """Route tensor-form/indexed normal-form objects through central TensorExpr canonicalization.

    The tensor-form engine still reconstructs classic normal-form objects, but
    canonical comparison and provenance can be delegated to the shared TensorExpr
    engine through this adapter.
    """
    try:
        from .tensor_expr_canonicalization import canonicalize_tensor_expr
        return canonicalize_tensor_expr(obj, registry=registry)
    except Exception:
        return None
