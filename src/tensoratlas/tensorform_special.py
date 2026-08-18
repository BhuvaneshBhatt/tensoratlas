from __future__ import annotations
import sympy as sp
from .tensor_core import TensorObject

def _ti():
    from . import tensor_indices as ti
    return ti

def _epsilon_rank(f):
    ti=_ti(); cls=ti._classify_special_tensor(f.tensor); return len(f.indices) if cls=="epsilon" else None

def _epsilon_epsilon_delta_rewrite(indexed_factors, scalar_expr):
    ti=_ti()
    for i in range(len(indexed_factors)):
        for j in range(i+1,len(indexed_factors)):
            a,b=indexed_factors[i],indexed_factors[j]
            if ti._classify_special_tensor(a.tensor)!='epsilon' or ti._classify_special_tensor(b.tensor)!='epsilon': continue
            if len(a.indices)!=len(b.indices): continue
            n=len(a.indices); names_a={x.name:x for x in a.indices}; names_b={x.name:x for x in b.indices}; common=[name for name in names_a if name in names_b]
            if len(common)==n and all(names_a[nm].variance != names_b[nm].variance for nm in common):
                scalar_expr *= sp.factorial(n); rest=[f for k,f in enumerate(indexed_factors) if k not in {i,j}]; return True, rest, scalar_expr
            if len(common)==n-1:
                free_a=[idx for idx in a.indices if idx.name not in common]; free_b=[idx for idx in b.indices if idx.name not in common]
                if len(free_a)==1 and len(free_b)==1 and free_a[0].variance != free_b[0].variance:
                    from .tensor_algebra import kronecker_delta_tensor
                    delta=TensorObject.from_tensor_field(kronecker_delta_tensor(a.tensor.chart), name='δ')
                    repl=ti.IndexedTensor(delta,(ti.TensorIndex(free_b[0].name,'u',free_b[0].bundle),ti.TensorIndex(free_a[0].name,'l',free_a[0].bundle))).canonicalize()
                    rest=[f for k,f in enumerate(indexed_factors) if k not in {i,j}] + [repl]
                    scalar_expr *= sp.factorial(n-1); return True, rest, scalar_expr
    return False, indexed_factors, scalar_expr

def _epsilon_metric_raise_lower(indexed_factors, scalar_expr):
    ti=_ti()
    for i in range(len(indexed_factors)):
        for j in range(len(indexed_factors)):
            if i==j: continue
            eps,met=indexed_factors[i],indexed_factors[j]
            if ti._classify_special_tensor(eps.tensor) != 'epsilon': continue
            mcls=ti._classify_special_tensor(met.tensor)
            if mcls not in {'metric_ll','metric_uu'}: continue
            eps_pos=met_pos=None
            for p,eidx in enumerate(eps.indices):
                for q,midx in enumerate(met.indices):
                    if eidx.name==midx.name and eidx.bundle==midx.bundle and eidx.variance!=midx.variance: eps_pos,met_pos=p,q; break
                if eps_pos is not None: break
            if eps_pos is None: continue
            other_q=1-met_pos; other_idx=met.indices[other_q]
            if mcls=='metric_uu' and eps.indices[eps_pos].variance=='l': new_tensor=eps.tensor.raise_slots([eps_pos]); new_var='u'
            elif mcls=='metric_ll' and eps.indices[eps_pos].variance=='u': new_tensor=eps.tensor.lower_slots([eps_pos]); new_var='l'
            else: continue
            new_indices=list(eps.indices); new_indices[eps_pos]=ti.TensorIndex(other_idx.name,new_var,other_idx.bundle)
            repl=ti.IndexedTensor(new_tensor,tuple(new_indices)).canonicalize()
            rest=[indexed_factors[k] for k in range(len(indexed_factors)) if k not in {i,j}] + [repl]
            return True, rest, scalar_expr
    return False, indexed_factors, scalar_expr

def _special_tensor_network_simplify(indexed_factors, scalar_expr):
    changed=True; cur_factors=indexed_factors; cur_scalar=scalar_expr
    while changed:
        changed=False
        try: done,cur_factors,cur_scalar=_more_complete_special_simplify(cur_factors,cur_scalar)
        except Exception: done=False
        if done: changed=True; continue
        done,cur_factors,cur_scalar=_epsilon_metric_raise_lower(cur_factors,cur_scalar)
        if done: changed=True
    return cur_factors, cur_scalar

def _delta_chain_simplify(indexed_factors):
    ti=_ti(); changed=False; factors=list(indexed_factors); out=[]; consumed=set(); rename_map={}
    for i,f in enumerate(factors):
        if i in consumed: continue
        cls=ti._leaf_tensor_kind(f)
        if cls not in {"delta","identity"}: out.append(f); continue
        inds=list(f.indices)
        if len(inds)!=2: out.append(f); continue
        src=inds[1].name if inds[0].variance=='u' else inds[0].name
        dst=inds[0].name if inds[0].variance=='u' else inds[1].name
        rename_map[src]=(dst, inds[0].bundle if inds[0].variance=='u' else inds[1].bundle); consumed.add(i); changed=True
    if rename_map:
        new_out=[]
        for f in out+[factors[i] for i in range(len(factors)) if i not in consumed and ti._leaf_tensor_kind(factors[i]) not in {"delta","identity"}]:
            idxs=[]
            for idx in f.indices:
                if idx.name in rename_map:
                    new_name,new_bundle=rename_map[idx.name]; idxs.append(ti.TensorIndex(new_name,idx.variance, idx.bundle if idx.bundle is not None else new_bundle))
                else: idxs.append(idx)
            new_out.append(ti.IndexedTensor(f.tensor,tuple(idxs)).canonicalize())
        return True, new_out
    return changed,out

def _metric_chain_simplify(indexed_factors):
    ti=_ti(); changed=False; factors=list(indexed_factors)
    for i in range(len(factors)):
        for j in range(len(factors)):
            if i==j: continue
            fi,fj=factors[i],factors[j]; ki,kj=ti._leaf_tensor_kind(fi), ti._leaf_tensor_kind(fj)
            if {ki,kj}!={"metric_ll","metric_uu"}: continue
            shared=None
            for p,idxi in enumerate(fi.indices):
                for q,idxj in enumerate(fj.indices):
                    if idxi.name==idxj.name and idxi.bundle==idxj.bundle and idxi.variance!=idxj.variance: shared=(p,q); break
                if shared: break
            if shared is None: continue
            p,q=shared; other_i=fi.indices[1-p]; other_j=fj.indices[1-q]
            from .tensor_algebra import kronecker_delta_tensor
            delta_to=TensorObject.from_tensor_field(kronecker_delta_tensor(fi.tensor.chart), name='δ')
            repl=ti.IndexedTensor(delta_to,(ti.TensorIndex(other_i.name,'u',other_i.bundle), ti.TensorIndex(other_j.name,'l',other_j.bundle))).canonicalize()
            rest=[factors[k] for k in range(len(factors)) if k not in {i,j}] + [repl]
            return True, rest
    return changed, factors

def _epsilon_epsilon_extended(indexed_factors, scalar_expr):
    ti=_ti()
    for i in range(len(indexed_factors)):
        for j in range(i+1,len(indexed_factors)):
            fi,fj=indexed_factors[i],indexed_factors[j]
            if ti._leaf_tensor_kind(fi)!='epsilon' or ti._leaf_tensor_kind(fj)!='epsilon': continue
            if len(fi.indices)!=len(fj.indices): continue
            if all(any(a.name==b.name and a.bundle==b.bundle and a.variance!=b.variance for b in fj.indices) for a in fi.indices):
                n=len(fi.indices); scalar_expr *= sp.factorial(n); rest=[indexed_factors[k] for k in range(len(indexed_factors)) if k not in {i,j}]; return True, rest, scalar_expr
    return False, indexed_factors, scalar_expr

def _epsilon_partial_contraction(indexed_factors, scalar_expr):
    ti=_ti()
    for i in range(len(indexed_factors)):
        for j in range(i+1,len(indexed_factors)):
            fi,fj=indexed_factors[i],indexed_factors[j]
            if ti._leaf_tensor_kind(fi)!='epsilon' or ti._leaf_tensor_kind(fj)!='epsilon': continue
            if len(fi.indices)!=len(fj.indices): continue
            n=len(fi.indices); pairs=[]; free_i=[]
            for p,idxi in enumerate(fi.indices):
                matched=False
                for q,idxj in enumerate(fj.indices):
                    if q in [qq for _,qq in pairs]: continue
                    if idxi.name==idxj.name and idxi.bundle==idxj.bundle and idxi.variance!=idxj.variance: pairs.append((p,q)); matched=True; break
                if not matched: free_i.append(p)
            free_j=[q for q in range(n) if q not in [qq for _,qq in pairs]]; k=len(pairs)
            if k==0 or len(free_i)!=len(free_j): continue
            if n-k==0:
                scalar_expr *= sp.factorial(n); rest=[indexed_factors[m] for m in range(len(indexed_factors)) if m not in {i,j}]; return True, rest, scalar_expr
            up_indices=[ti.TensorIndex(fj.indices[p].name,'u',fj.indices[p].bundle) for p in free_j]
            down_indices=[ti.TensorIndex(fi.indices[p].name,'l',fi.indices[p].bundle) for p in free_i]
            gdo=TensorObject.from_tensor_field(ti.generalized_kronecker_delta_tensor(fi.tensor.chart, n-k), name='Δ')
            repl=ti.IndexedTensor(gdo, tuple(up_indices+down_indices)).canonicalize()
            rest=[indexed_factors[m] for m in range(len(indexed_factors)) if m not in {i,j}] + [repl]
            scalar_expr *= sp.factorial(k); return True, rest, scalar_expr
    return False, indexed_factors, scalar_expr

def _epsilon_metric_chain_simplify(indexed_factors, scalar_expr):
    changed=False
    for _ in range(max(1,len(indexed_factors))):
        done,indexed_factors,scalar_expr=_epsilon_metric_raise_lower(indexed_factors, scalar_expr)
        if not done: break
        changed=True
    done2,indexed_factors,scalar_expr=_epsilon_partial_contraction(indexed_factors, scalar_expr)
    return changed or done2, indexed_factors, scalar_expr

def _more_complete_special_simplify(indexed_factors, scalar_expr):
    ti=_ti(); done,indexed_factors=_delta_chain_simplify(indexed_factors)
    if done: return True,indexed_factors,scalar_expr
    done,indexed_factors=_metric_chain_simplify(indexed_factors)
    if done: return True,indexed_factors,scalar_expr
    done,indexed_factors,scalar_expr=_epsilon_metric_chain_simplify(indexed_factors, scalar_expr)
    if done: return True,indexed_factors,scalar_expr
    return ti._fallback_special_tensor_simplify(indexed_factors, scalar_expr)
