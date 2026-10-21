"""Strict numeric and GO-retrieval compatibility; never changes acceptance limits."""
import json,sys
from pathlib import Path
import numpy as np

def numerical(candidate,baseline):
    a=np.asarray(candidate,dtype=np.float64);b=np.asarray(baseline,dtype=np.float64)
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    error=np.max(np.abs(a-b),axis=1)
    cosine=np.sum(a*b,axis=1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1))
    bad=np.flatnonzero((error>1e-5)|(cosine<.999999))
    worst=np.argsort(error,kind='stable')[-12:][::-1]
    return {'rows':len(a),'max_abs':float(error.max()),'min_cosine':float(cosine.min()),'exact':bool(np.array_equal(a,b)),'violating_rows':len(bad),'violating_indices':bad.tolist(),'worst_rows':[{'row':int(i),'max_abs':float(error[i]),'cosine':float(cosine[i])} for i in worst]}

def retrieval(bank0,queries0,bank1,queries1,go_ids,query_ids,cutoffs=(.90,.92,.935)):
    norm=lambda x:np.asarray(x,dtype=np.float64)/np.linalg.norm(np.asarray(x,dtype=np.float64),axis=1,keepdims=True)
    s0=norm(queries0)@norm(bank0).T;s1=norm(queries1)@norm(bank1).T
    # Stable source order is the declared tie break for both versions.
    rank0=np.argsort(-s0,axis=1,kind='stable');rank1=np.argsort(-s1,axis=1,kind='stable')
    ids=np.arange(len(s0))
    ranks1=np.empty_like(rank1);ranks1[ids[:,None],rank1]=np.arange(rank1.shape[1])[None,:]
    delta=np.abs(s0-s1)
    out={'query_count':len(s0),'go_count':s0.shape[1],'max_score_abs_change':float(delta.max()),'top1_changed':int((rank0[:,0]!=rank1[:,0]).sum()),'topk':{},'cutoffs':{},'near_ties':[],'largest_changes':[],'score_precision':'float64 cosine on float32 model vectors; stable source-order ties'}
    for k in [3,10,20]:
        overlap=np.array([len(set(a[:k])&set(b[:k]))/k for a,b in zip(rank0,rank1)])
        displacement=np.abs(ranks1[ids[:,None],rank0[:,:k]]-np.arange(k)[None,:])
        out['topk'][str(k)]={'mean_overlap':float(overlap.mean()),'min_overlap':float(overlap.min()),'ordered_list_changed':int(np.any(rank0[:,:k]!=rank1[:,:k],axis=1).sum()),'max_rank_displacement_of_baseline_topk':int(displacement.max())}
    for cutoff in cutoffs:
        flip=(s0>cutoff)!=(s1>cutoff)
        proximity=np.abs(s0-cutoff);flat=np.argsort(proximity.ravel(),kind='stable')[:12]
        out['cutoffs'][str(cutoff)]={'decision_flips_all_pairs':int(flip.sum()),'query_top3_selection_changed':int(sum(set(a[:3][s0[i,a[:3]]>cutoff])!=set(b[:3][s1[i,b[:3]]>cutoff]) for i,(a,b) in enumerate(zip(rank0,rank1)))),'nearest_cases':[{'query':query_ids[i],'go':go_ids[j],'baseline':float(s0[i,j]),'candidate':float(s1[i,j]),'margin':float(proximity[i,j]),'decision_flipped':bool(flip[i,j])} for i,j in [np.unravel_index(v,s0.shape) for v in flat]]}
    top_scores=np.take_along_axis(s0,rank0[:,:21],axis=1);margins=top_scores[:,:-1]-top_scores[:,1:]
    smallest=np.argsort(margins.ravel(),kind='stable')[:16]
    for x in smallest:
        i,j=np.unravel_index(x,margins.shape);a,b=rank0[i,j:j+2]
        out['near_ties'].append({'query':query_ids[i],'baseline_positions':[int(j+1),int(j+2)],'go':[go_ids[a],go_ids[b]],'baseline_gap':float(margins[i,j]),'candidate_gap':float(s1[i,a]-s1[i,b]),'order_flipped':bool(ranks1[i,a]>ranks1[i,b])})
    for x in np.argsort(delta.ravel(),kind='stable')[-12:][::-1]:
        i,j=np.unravel_index(x,s0.shape)
        out['largest_changes'].append({'query':query_ids[i],'go':go_ids[j],'baseline':float(s0[i,j]),'candidate':float(s1[i,j]),'abs_change':float(delta[i,j])})
    return out

if __name__=='__main__':
    raise SystemExit('Import numerical() and retrieval() from the experiment runner; no production writes.')
