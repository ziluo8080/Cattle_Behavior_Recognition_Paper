"""Pure NumPy fusion recipes; no labels used in prediction rules."""
import numpy as np


def validate_probabilities(p):
    p=np.asarray(p,dtype=np.float64)
    assert p.ndim==2 and p.shape[1]==7
    assert np.isfinite(p).all() and (p>=0).all() and (p<=1+1e-6).all()
    assert np.allclose(p.sum(1),1.,rtol=0,atol=2e-6)
    return p/p.sum(1,keepdims=True)


def temper(p,t,floor=1e-12):
    assert t>0
    log=np.log(np.maximum(p,floor))/t
    log-=log.max(1,keepdims=True)
    q=np.exp(log)
    return q/q.sum(1,keepdims=True)


def confidence(p,kind,floor=1e-12):
    if kind=='entropy':
        return np.clip(1+(p*np.log(np.maximum(p,floor))).sum(1)/np.log(7.),0,1)
    if kind=='margin':
        top=np.partition(p,-2,axis=1)[:,-2:]
        return top[:,1]-top[:,0]
    raise ValueError(kind)


def borda(p):
    order=np.argsort(-p,axis=1,kind='stable')
    result=np.empty_like(p)
    np.put_along_axis(result,order,np.broadcast_to(np.arange(7,0,-1)/28.,p.shape),axis=1)
    return result


def fuse(px,pb,family,w,floor=1e-12):
    assert px.shape==pb.shape and 0<=w<=1
    if family=='arithmetic':return w*px+(1-w)*pb
    if family=='geometric':
        log=w*np.log(np.maximum(px,floor))+(1-w)*np.log(np.maximum(pb,floor))
        log-=log.max(1,keepdims=True);q=np.exp(log)
        return q/q.sum(1,keepdims=True)
    if family=='rank_average':return w*borda(px)+(1-w)*borda(pb)
    kind,operation=family.split('_')
    cx,cb=confidence(px,kind,floor),confidence(pb,kind,floor)
    if operation=='select':return np.where((w*cx>=(1-w)*cb)[:,None],px,pb)
    if operation=='average':
        denominator=w*cx+(1-w)*cb
        effective=np.divide(w*cx,denominator,out=np.full_like(cx,w),where=denominator>1e-15)
        return effective[:,None]*px+(1-effective[:,None])*pb
    raise ValueError(family)


def recipes(cfg):
    result=[]
    for view in cfg['B_views']:
        for family in cfg['families']:
            temperatures=cfg['rank_temperature_pairs'] if family=='rank_average' else cfg['temperature_pairs_XGB_B']
            for tx,tb in temperatures:
                for w in cfg['XGB_weights']:
                    result.append(dict(id=f'fusion_{len(result):03d}',view=view,family=family,
                                       XGB_weight=w,XGB_temperature=tx,B_temperature=tb))
    assert len(result)==cfg['expected_fusion_recipes']
    return result
