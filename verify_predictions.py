"""Recompute saved-prediction scores/paired intervals. No model or raw stream load."""
from pathlib import Path
import argparse,csv,json
import numpy as np
ROOT=Path(__file__).resolve().parent
def f1(cm):
    den=cm.sum(-2)+cm.sum(-1)
    return np.divide(2*np.diagonal(cm,axis1=-2,axis2=-1),den,out=np.zeros(den.shape,dtype=float),where=den>0)
def group_cm(y,p,group,n):
    return np.stack([np.bincount(group*49+(y-1)*7+row-1,minlength=n*49).reshape(n,7,7) for row in p])
def validate(family,protocol,replicates):
    z=np.load(ROOT/'predictions'/f'{family}_{protocol}.npz',allow_pickle=False)
    y=z['label'].astype(np.int64);pred=z['prediction'].astype(np.int64)
    model=z['model'];fold=z['fold'];cow=z['cow_id'];t=z['timestamp'];seed=z['seed']
    assert len(y)==46821 and len(set(zip(cow.tolist(),t.tolist())))==46821
    assert np.isin(y,np.arange(1,8)).all() and np.isin(pred,np.arange(1,8)).all()
    model_names=list(dict.fromkeys(model.tolist()));ix={m:np.flatnonzero(model==m) for m in model_names}
    rows=[];scores={m:[] for m in model_names};classes={m:[] for m in model_names}
    for fi in range(1,6):
        use=fold==fi;cm=group_cm(y[use],pred[:,use],np.zeros(use.sum(),dtype=np.int64),1)[:,0]
        cls=f1(cm)
        for m in model_names:scores[m].append(cls[ix[m]].mean(-1));classes[m].append(cls[ix[m]])
        rows.append({'mask':use,'cm':cm})
    means={m:float(np.asarray(scores[m]).mean()) for m in model_names}
    if family=='main':
        expected=json.loads((ROOT/'results/all11_fast/verified_summary.json').read_text())['protocols'][protocol]
        pairs={k:(k.split('_minus_',1)[0],k.split('_minus_',1)[1],v) for k,v in expected['comparisons'].items()}
        for m,v in means.items():
            np.testing.assert_allclose(v,expected['scores'][m]['mean'],rtol=0,atol=1e-12)
            np.testing.assert_allclose(np.asarray(classes[m]).mean((0,1)),expected['scores'][m]['per_class_mean'],rtol=0,atol=1e-12)
        rng_seed=20261014
    else:
        expected=json.loads((ROOT/'results/fixed20/verified_summary.json').read_text())['protocols'][protocol]
        for m,v in means.items():np.testing.assert_allclose(v,expected['scores'][m]['mean'],rtol=0,atol=1e-12)
        with (ROOT/'results/fixed20/key_ablations.csv').open(encoding='utf-8-sig') as f:ab=[r for r in csv.DictReader(f) if r['protocol']==protocol and r['method']!='Base']
        pairs={r['method']:('Base',r['method'],{'delta':float(r['Base_minus_variant']),'conditional_time_CI95':json.loads(r['time_CI']),'conditional_cow_CI95':json.loads(r['cow_CI']) if r['cow_CI'] else None}) for r in ab}
        rng_seed=20261007
    result={'family':family,'protocol':protocol,'targets':len(y),'model_seed_rows':len(model),'folds':5,'mean_checks':len(means),'means':means,'interval_checks':0}
    if not replicates:return result
    assert replicates==2000,'Paper intervals use exactly 2000 draws'
    blocks=np.unique(t//300);draw=np.random.default_rng(rng_seed).multinomial(len(blocks),np.full(len(blocks),1/len(blocks)),size=replicates)
    tm=np.zeros((replicates,len(model_names)));cw=np.zeros_like(tm) if protocol=='OS' else None
    shared_cow_rng=np.random.default_rng(rng_seed)
    for fi,row in enumerate(rows,1):
        use=row['mask'];g=group_cm(y[use],pred[:,use],np.searchsorted(blocks,t[use]//300),len(blocks))
        samples=f1(np.einsum('rb,sbij->rsij',draw,g)).mean(-1)
        for j,m in enumerate(model_names):tm[:,j]+=samples[:,ix[m]].mean(1)/5
        if cw is not None:
            cows,ids=np.unique(cow[use],return_inverse=True);assert len(cows)==2
            rng=shared_cow_rng if family=='main' else np.random.default_rng(20261007+fi-1)
            cd=rng.multinomial(2,[.5,.5],size=replicates);cg=group_cm(y[use],pred[:,use],ids,2)
            cs=f1(np.einsum('rb,sbij->rsij',cd,cg)).mean(-1)
            for j,m in enumerate(model_names):cw[:,j]+=cs[:,ix[m]].mean(1)/5
    comparisons={}
    for key,(left,right,stored) in pairs.items():
        li,ri=model_names.index(left),model_names.index(right)
        delta=means[left]-means[right];time_ci=np.quantile(tm[:,li]-tm[:,ri],[.025,.975])
        np.testing.assert_allclose(delta,stored['delta'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(time_ci,stored['conditional_time_CI95'],rtol=0,atol=1e-10)
        cow_ci=None if cw is None else np.quantile(cw[:,li]-cw[:,ri],[.025,.975])
        if cow_ci is not None:np.testing.assert_allclose(cow_ci,stored['conditional_cow_CI95'],rtol=0,atol=1e-10)
        comparisons[key]={'delta':delta,'time_CI95':time_ci.tolist(),'cow_CI95':None if cow_ci is None else cow_ci.tolist()}
    result.update(interval_checks=len(comparisons),comparisons=comparisons,bootstrap_seed=rng_seed,replicates=replicates)
    return result
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--bootstrap',action='store_true');ap.add_argument('--output',type=Path,default=ROOT/'validation/prediction_verification.json');args=ap.parse_args()
    results=[]
    for family in ['main','fixed20']:
        for protocol in ['TS','OS']:
            results.append(validate(family,protocol,2000 if args.bootstrap else 0));print(f'CHECKED {family} {protocol}',flush=True)
    report={'status':'PASS','scope':'saved_predictions_and_conditional_intervals' if args.bootstrap else 'saved_predictions_mean_scores','results':results,'training':False,'model_inference':False,'raw_streams_reloaded':False}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SAVED_PREDICTIONS_VERIFIED',flush=True)
if __name__=='__main__':main()
