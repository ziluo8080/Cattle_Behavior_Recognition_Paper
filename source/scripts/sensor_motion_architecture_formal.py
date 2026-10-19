"""Complete official frozen ablations; score TEST only after all prescribed fits."""
import time, hashlib, gc, shutil
START, WALL = time.monotonic(), time.time()
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.sensor_motion_architecture_formal_common import CONFIG, ROOT, read, write, sha, f1, metrics, get_data, setup, load_base, frozen_features, FormalBase, FormalRefiner, original_path, drop_batch
from scripts.sensor_motion_architecture_model import MaskedBase, ArchitectureRefiner, base_outputs, coarse, pair, weighted_loss
from scripts.sensor_motion_pair import invariant
from q3_sensor.endpoint_models import EndpointFusion
from scripts.sensor_motion_architecture_formal_summary import summarize


def budget(cfg, stage):
    elapsed=time.monotonic()-START
    if elapsed>cfg['resources']['planned_main_budget_seconds']: raise RuntimeError('Main budget exceeded: '+stage)
    return elapsed


def weights_for(stats, kind):
    n=np.asarray(stats['class_counts'],float)
    w=np.ones(7) if kind=='FullUniform' else n.sum()/(7*n) if kind=='FullBalanced' else np.sqrt(n.sum()/(7*n))
    return torch.tensor(w,dtype=torch.float32,device='cuda')


def initial(parent, proto, seed):
    stats=get_data(parent,proto)['stats']
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);model=EndpointFusion('gru',stats['prior'])
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def make_base(parent,proto,seed,stats,kind):
    return FormalBase(stats['prior'],initial(parent,proto,seed),kind).cuda()


def make_ref(cfg,parent,proto,seed,stats,kind):
    base_kind=kind.removesuffix('Motion') if kind.endswith('Motion') else 'Full'
    mask=base_kind if base_kind in ['RecurrentOnly','CurrentOnly'] else 'Full'
    if base_kind=='Full': base=load_base(parent,proto,seed)
    else:
        directory=Path(cfg['output'])/proto/f'{base_kind}_seed{seed}'
        done=read(directory/'COMPLETE.json')
        assert sha(directory/'model.pt')==done['files']['model.pt']
        state=torch.load(directory/'model.pt',map_location='cpu',weights_only=False)['state_dict']
        base=FormalBase(stats['prior'],state,base_kind).cuda()
    return FormalRefiner(base,kind,seed+600000,mask).cuda()


def new_dir(cfg,proto,kind,seed):
    path=Path(cfg['output'])/proto/f'{kind}_seed{seed}'
    if path.exists(): raise RuntimeError('Fit restart prohibited')
    path.mkdir(parents=True);write(path/'STARTED.json',dict(protocol=proto,kind=kind,seed=seed,elapsed=budget(cfg,'fit_start')))
    return path


def save_eval(path, table, values, base_prediction=None):
    out=table[['cow_id','timestamp','label']].copy()
    for name,value in values.items(): out[name]=value
    if base_prediction is not None: out['base_prediction']=base_prediction
    out.to_csv(path/'predictions.csv',index=False)


def evaluate(model,d,motion=None):
    model.eval();predictions={};base=[];aux=[];logits=[]
    with torch.no_grad():
        for at in range(0,len(d['val']),256):
            idx=np.arange(at,min(at+256,len(d['val'])))
            if motion is None:
                b=model(d['vapool'].batch(idx));predictions.setdefault('prediction',[]).append(b.argmax(1).cpu().numpy()+1);logits.append(b.cpu().numpy())
            else:
                out=model(d['vapool'].batch(idx),torch.from_numpy(motion[idx]).cuda())
                for k,v in out['predictions'].items(): predictions.setdefault(k,[]).append(v.cpu().numpy())
                base.append(out['base_logits'].cpu().numpy());aux.append(out['auxiliary_logits'].cpu().numpy())
                if out['logits7'] is not None: logits.append(out['logits7'].cpu().numpy())
    return {k:np.concatenate(v) for k,v in predictions.items()},np.concatenate(base) if base else None,np.concatenate(aux) if aux else None,np.concatenate(logits) if logits else None


def sanity(cfg,parent):
    case='TS/fold_1';d=get_data(parent,case);stats=d['stats'];batch=d['trpool'].batch(np.arange(32))
    y=torch.tensor(d['train'].label.to_numpy()[:32]-1,device='cuda');w=weights_for(stats,'Full')
    rows=[];steps=0
    # Exact vanilla implementation/gradient update witness, not a new training run.
    a=make_base(parent,case,17,stats,'Full');b=EndpointFusion('gru',stats['prior']).cuda();b.load_state_dict(initial(parent,case,17))
    a.train();b.train();torch.manual_seed(17);torch.cuda.manual_seed_all(17)
    state=torch.cuda.get_rng_state();la=a(batch);torch.cuda.set_rng_state(state);lb=b(batch)
    torch.testing.assert_close(la,lb,atol=0,rtol=0)
    for model,logits in [(a,la),(b,lb)]:
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);torch.nn.functional.cross_entropy(logits,y,weight=w).backward();opt.step();steps+=1
    for k,v in a.state_dict().items(): torch.testing.assert_close(v,b.state_dict()[k],atol=0,rtol=0)
    del a,b
    for kind in [a['name'] for a in cfg['new_base_arms']]:
        model=make_base(parent,case,17,stats,kind).train();out=model(batch);loss=torch.nn.functional.cross_entropy(out,y,weight=weights_for(stats,kind));loss.backward()
        grad={k:float(p.grad.abs().max()) if p.grad is not None else None for k,p in model.named_parameters()}
        assert all(v is None or np.isfinite(v) for v in grad.values())
        head=model.head[0].weight.grad
        if kind=='CurrentOnly': assert torch.count_nonzero(head[:,:192])==0 and all(v==0 for k,v in grad.items() if k.startswith('encoders.'))
        if kind=='RecurrentOnly': assert torch.count_nonzero(head[:,192:])==0
        assert float(model.head[-1].weight.grad.abs().sum())>0
        torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01).step();steps+=1
        rows.append(dict(kind=kind,gradient_maxima=grad,nominal_parameters=sum(p.numel() for p in model.parameters()),inactive_capacity_disclosed=True));del model
    states={};motion=torch.from_numpy(d['native'][0][:32]).cuda()
    # Recurrent-only technical witness uses the original trained state, does not replace the actual new B-R fit.
    for kind in [r['name'] for r in cfg['new_neural_refiner_arms'] if r['name']!='GridIntervalPair']:
        mask=kind.removesuffix('Motion') if kind in ['RecurrentOnlyMotion','CurrentOnlyMotion'] else 'Full'
        model=FormalRefiner(load_base(parent,case,17),kind,600017,mask).cuda().train()
        assert not model.base.training
        frozen={k:v.cpu().clone() for k,v in model.base.state_dict().items()};states[kind]={k:v.cpu().clone() for k,v in model.residual.state_dict().items()}
        out=model(batch,motion);invariant(out['predictions']['pair3'].cpu().numpy(),out['base_logits'].argmax(1).cpu().numpy()+1)
        target_logits=out['logits7'] if kind=='R7Fine' else out['auxiliary_logits']
        rw=weights_for(stats,'FullUniform' if kind=='FullUniformMotion' else 'FullBalanced' if kind=='FullBalancedMotion' else 'Full')
        loss=weighted_loss(target_logits,y,rw,kind!='R7Fine');loss.backward()
        assert all(p.grad is None for p in model.base.parameters())
        params=[p for p in model.parameters() if p.requires_grad]
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        assert float(sum(p.grad.square().sum() for p in params))>0
        torch.optim.AdamW(params,lr=.001,weight_decay=.01).step();steps+=1
        for k,v in model.base.state_dict().items(): torch.testing.assert_close(v.cpu(),frozen[k],atol=0,rtol=0)
        if mask=='RecurrentOnly':
            x,_=base_outputs(model.base,batch,mask);assert torch.count_nonzero(x[:,192:])==0
        if mask=='CurrentOnly':
            x,_=base_outputs(model.base,batch,mask);assert torch.count_nonzero(x[:,:192])==0
        if kind in ['NoUWBMotion','NoHDMotion','NoAklMotion']:
            drop=model.drop;changed={m:{k:v.clone() for k,v in values.items()} for m,values in batch.items()}
            changed[drop]['x'].fill_(999)
            with torch.no_grad():
                aa=model(batch,motion)['auxiliary_logits'];bb=model(changed,motion*999 if drop=='uwb' else motion)['auxiliary_logits']
            torch.testing.assert_close(aa,bb,atol=0,rtol=0)
        rows.append(dict(kind=kind,trainable=sum(p.numel() for p in params),Base_frozen=True,protected_indicators=True));del model
    for k,v in states['R7Fine'].items(): torch.testing.assert_close(v,states['R7Coarse'][k],atol=0,rtol=0)
    write(Path(cfg['output'])/'SANITY.json',dict(status='PASS',records=rows,real_TRAIN_rows=32,sanity_optimizer_steps=steps,vanilla_update_exact=True))
    print('ARCHITECTURE_SANITY_PASS',steps,flush=True);gc.collect();torch.cuda.empty_cache()


def train_neural(cfg,parent,proto,kind,seed,pilot_state):
    path=new_dir(cfg,proto,kind,seed);d=get_data(parent,proto);stats=d['stats'];is_base=kind in [a['name'] for a in cfg['new_base_arms']]
    model=make_base(parent,proto,seed,stats,kind) if is_base else make_ref(cfg,parent,proto,seed,stats,kind)
    torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},path/'initial_state.pt')
    frozen=None if is_base else {k:v.detach().cpu().clone() for k,v in model.base.state_dict().items()}
    params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=.001,weight_decay=.01)
    w=weights_for(stats,kind if is_base else 'FullUniform' if kind=='FullUniformMotion' else 'FullBalanced' if kind=='FullBalancedMotion' else 'Full');y=torch.tensor(d['train'].label.to_numpy()-1,device='cuda')
    selected='grid' if kind=='GridIntervalPair' else 'native';motion=d[selected][0]
    write(path/'motion_scaler.json',d['grid_scaler'] if selected=='grid' else d['native_scaler'])
    torch.manual_seed(seed if is_base else seed+600000);torch.cuda.manual_seed_all(seed if is_base else seed+600000)
    rng=np.random.default_rng(seed);history=[];order_hashes=[];begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,21):
        e=time.monotonic();model.train();order=rng.permutation(len(d['train']));order_hashes.append(hashlib.sha256(order.astype('<i8').tobytes()).hexdigest());numerator=mass=0.
        for at in range(0,len(order),256):
            idx=order[at:at+256];opt.zero_grad(set_to_none=True)
            if is_base: loss=torch.nn.functional.cross_entropy(model(d['trpool'].batch(idx)),y[idx],weight=w)
            else:
                out=model(d['trpool'].batch(idx),torch.from_numpy(motion[idx]).cuda());loss=weighted_loss(out['logits7'] if kind=='R7Fine' else out['auxiliary_logits'],y[idx],w,kind!='R7Fine')
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            loss.backward()
            if not is_base: assert all(p.grad is None for p in model.base.parameters())
            opt.step();m=float(w[y[idx]].sum());numerator+=float(loss.detach())*m;mass+=m
        history.append(dict(epoch=epoch,train_CE=numerator/mass,epoch_seconds=time.monotonic()-e));write(path/'history.json',history);budget(cfg,'epoch')
        print('ARCHITECTURE_EPOCH',proto,kind,seed,epoch,history[-1],flush=True)
        family='Base' if is_base else kind
        if family not in pilot_state and epoch==5:
            pilot_state[family]=max(r['epoch_seconds'] for r in history[1:])*1.25
            measured_refs=[v for k,v in pilot_state.items() if k not in ['Base','neural_jobs']]
            if 'Base' not in pilot_state or not measured_refs:
                write(Path(cfg['output'])/'PILOT.json',dict(rates=pilot_state,forecast_deferred_until_Base_and_refiner_prefixes_measured=True))
                continue
            base_rate=pilot_state['Base'];ref_rate=max(measured_refs)
            completed=sum(1 for _ in Path(cfg['output']).glob('*/*/*/COMPLETE.json'))
            remaining_base=max(0,210-sum(1 for p in Path(cfg['output']).glob('*/*/*/COMPLETE.json') if read(p)['family']=='Base'))
            remaining_ref=max(0,pilot_state['neural_jobs']-completed-remaining_base)
            forecast=budget(cfg,'pilot')+remaining_base*20*base_rate+remaining_ref*20*ref_rate+5400+1200
            write(Path(cfg['output'])/'PILOT.json',dict(rates=pilot_state,forecast=forecast,remaining_base=remaining_base,remaining_ref=remaining_ref))
            if forecast>cfg['resources']['planned_main_budget_seconds']:
                torch.save(dict(state_dict=model.state_dict(),completed_epochs=epoch,resumable=False),path/'partial_model.pt');raise RuntimeError('Pilot budget insufficient')
    training_seconds=time.monotonic()-begin
    if frozen is not None:
        for k,v in model.base.state_dict().items(): torch.testing.assert_close(v.cpu(),frozen[k],atol=0,rtol=0)
    predictions,b,a,z=evaluate(model,d,None if is_base else d[selected][1])
    if is_base: np.save(path/'logits7.npy',z);basepred=None
    else:
        np.save(path/'base_logits.npy',b);np.save(path/'auxiliary_logits.npy',a);basepred=b.argmax(1)+1
        if z is not None: np.save(path/'logits7.npy',z)
        for decoder in ['pair3','pair7']:
            if decoder in predictions: invariant(predictions[decoder],basepred)
        if kind in ['R7Fine','R7Coarse','GridIntervalPair']:
            pp,ff=proto.split('/');old=pd.read_csv(Path(parent['original_formal'][pp])/ff/'predictions.csv');np.testing.assert_array_equal(basepred,old[f'NN_seed{seed}'].to_numpy())
    save_eval(path,d['val'],predictions,basepred)
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},kind=kind,protocol=proto,seed=seed,prior=stats['prior'],Base_mask=kind.removesuffix('Motion') if not is_base and kind.endswith('Motion') else kind if is_base else 'Full',fit='author_TRAIN_plus_VAL',motion_scaler=d['grid_scaler'] if selected=='grid' else d['native_scaler']),path/'model.pt')
    files={p.name:sha(p) for p in path.iterdir() if p.name not in ['COMPLETE.json','STARTED.json']}
    write(path/'COMPLETE.json',dict(family='Base' if is_base else 'Refiner',kind=kind,seed=seed,epochs=20,order_hashes=order_hashes,files=files,training_seconds=training_seconds,total_seconds=time.monotonic()-begin,
        parameters=sum(p.numel() for p in model.parameters()),trainable=sum(p.numel() for p in params),peak_allocated_bytes=torch.cuda.max_memory_allocated(),TEST_predictions_generated=True,TEST_scoring=False,Base_frozen_exact=not is_base))
    print('ARCHITECTURE_FIT_COMPLETE',proto,kind,seed,flush=True);del model,opt;gc.collect();torch.cuda.empty_cache()


def simple_features(parent,proto,seed,d):
    base=load_base(parent,proto,seed).eval();values=[]
    for role,pool in [('train',d['trpool']),('val',d['vapool'])]:
        x,b=frozen_features(base,pool,len(d[role]));values.append((np.column_stack((x,d['native'][0 if role=='train' else 1])),b))
    del base;return values


def train_linear(cfg,parent,proto,seed):
    path=new_dir(cfg,proto,'LinearPair',seed);begin=time.monotonic();d=get_data(parent,proto);(x,b),(v,bv)=simple_features(parent,proto,seed,d)
    x=torch.tensor(x,dtype=torch.float64,device='cuda');v=torch.tensor(v,dtype=torch.float64,device='cuda')
    b=coarse(torch.from_numpy(b).cuda()).double();bvc=coarse(torch.from_numpy(bv).cuda()).double()
    y=torch.tensor(d['train'].label.to_numpy()-1,device='cuda');w=weights_for(d['stats'],'Full').double()
    model=torch.nn.Linear(227,3,dtype=torch.float64,device='cuda');torch.nn.init.zeros_(model.weight);torch.nn.init.zeros_(model.bias)
    opt=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=500,max_eval=1000,tolerance_grad=1e-7,tolerance_change=1e-12,line_search_fn='strong_wolfe')
    evaluations=0;failure=None
    def objective(): return weighted_loss(b+model(x),y,w,True)+.005*model.weight.square().sum()
    def closure():
        nonlocal evaluations
        budget(cfg,'linear')
        if time.monotonic()-begin>180 or evaluations>=1000: raise RuntimeError('Linear fixed optimization limit')
        evaluations+=1;opt.zero_grad();loss=objective()
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite linear objective')
        loss.backward();return loss
    try:
        if time.monotonic()-begin>180: raise RuntimeError('Linear fixed optimization limit')
        opt.step(closure)
    except RuntimeError as e:
        if str(e)!='Linear fixed optimization limit': raise
        failure=str(e)
    if time.monotonic()-begin>180: failure='time_limit'
    opt.zero_grad();loss=objective();loss.backward();gradient=float(max(p.grad.abs().max() for p in model.parameters()))
    converged=gradient<=1e-7 and failure is None
    with torch.no_grad(): aux=bvc+model(v);basepred=torch.from_numpy(bv).cuda().argmax(1)+1;pred=pair(basepred,aux.argmax(1)+1)
    invariant(pred.cpu().numpy(),basepred.cpu().numpy());save_eval(path,d['val'],dict(prediction=pred.cpu().numpy()),basepred.cpu().numpy())
    np.save(path/'base_logits.npy',bv);np.save(path/'auxiliary_logits.npy',aux.cpu().numpy())
    torch.save(dict(state_dict={k:p.detach().cpu() for k,p in model.state_dict().items()},kind='LinearPair',seed=seed,protocol=proto,prior=d['stats']['prior'],motion_scaler=d['native_scaler'],fit='author_TRAIN_plus_VAL',frozen_feature_batch=256,dtype='float64'),path/'model.pt')
    if time.monotonic()-begin>180: failure='time_limit';converged=False
    write(path/'optimizer.json',dict(evaluations=evaluations,gradient_inf=gradient,objective=float(loss.detach()),converged=converged,failure=failure,L2_W=.01,max_iter=500,max_eval=1000,solver_state_iterations=int(opt.state[model.weight].get('n_iter',0)),fit_deadline_seconds=180,deadline_enforcement='closure_entry_and_post_step_and_final_qualification',process_hard_timeout=False))
    write(path/'COMPLETE.json',dict(family='LinearOptimization',kind='LinearPair',seed=seed,comparison_eligible=converged,seconds=time.monotonic()-begin,files={p.name:sha(p) for p in path.iterdir() if p.name!='STARTED.json'},TEST_predictions_generated=True,TEST_scoring=False))
    print('ARCHITECTURE_LINEAR_COMPLETE',proto,seed,'converged',converged,'gradient',gradient,flush=True);del model,opt,x,v;gc.collect();torch.cuda.empty_cache()


def threshold_predict(base,speed,valid,threshold):
    pred=base.copy();eligible=(base<=2)&valid;pred[eligible]=np.where(speed[eligible]>threshold,1,2);return pred


def train_threshold(cfg,parent,proto,seed):
    path=new_dir(cfg,proto,'SpeedThresholdPair',seed);begin=time.monotonic();d=get_data(parent,proto);base=load_base(parent,proto,seed).eval()
    _,bt=frozen_features(base,d['trpool'],len(d['train']));_,bv=frozen_features(base,d['vapool'],len(d['val']));del base
    p=bt.argmax(1)+1;y=d['train'].label.to_numpy();speed=d['raw'][0][:,4];valid=d['pairs'][0];eligible=(p<=2)&valid
    values=np.unique(speed[valid]);groups=np.searchsorted(values,speed[eligible])
    if len(values)==0: raise RuntimeError('No eligible TRAIN threshold cases')
    proposed=p.copy();proposed[eligible]=1;cm=np.asarray(metrics(y,proposed)['CM']);counts=np.bincount(groups*7+y[eligible]-1,minlength=len(values)*7).reshape(-1,7).cumsum(0)
    cms=np.broadcast_to(cm,(len(values)+1,7,7)).copy();cms[1:,:,0]-=counts;cms[1:,:,1]+=counts
    shifts=np.bincount(groups,weights=np.where(p[eligible]==1,1,-1),minlength=len(values)).cumsum()
    changes=np.r_[np.sum(p[eligible]!=1),np.sum(p[eligible]!=1)+shifts]
    scores=f1(cms).mean(1);best=np.flatnonzero(scores==scores.max());at=int(best[np.argmin(changes[best])])
    if at==0: threshold=-np.inf;mode='all_walking'
    elif at==len(values): threshold=np.inf;mode='all_standing'
    else:
        lo,hi=values[at-1],values[at];threshold=float(lo+(hi-lo)/2)
        if threshold>=hi: threshold=float(lo)
        mode='finite'
    fitted=threshold_predict(p,speed,valid,threshold);np.testing.assert_array_equal(np.asarray(metrics(y,fitted)['CM']),cms[at])
    q=bv.argmax(1)+1;pred=threshold_predict(q,d['raw'][1][:,4],d['pairs'][1],threshold);invariant(pred,q)
    save_eval(path,d['val'],dict(prediction=pred),q);np.save(path/'base_logits.npy',bv)
    write(path/'threshold.json',dict(mode=mode,threshold=float(threshold) if np.isfinite(threshold) else None,Fit_macro=float(scores[at]),Fit_changed_rows=int(changes[at]),unique_speed_values=len(values),eligible_Fit_rows=int(eligible.sum()),tie_rule='min_changes_then_min_threshold',midpoint_upper_rounding_uses_lower=True,fit='author_TRAIN_plus_VAL'))
    write(path/'COMPLETE.json',dict(family='ThresholdFit',kind='SpeedThresholdPair',seed=seed,seconds=time.monotonic()-begin,files={p.name:sha(p) for p in path.iterdir() if p.name!='STARTED.json'},TEST_predictions_generated=True,TEST_scoring=False))
    print('ARCHITECTURE_THRESHOLD_COMPLETE',proto,seed,flush=True);gc.collect();torch.cuda.empty_cache()


def main():
    global START,WALL
    setup();cfg=read(CONFIG);parent=cfg;root=Path(cfg['output'])
    amendment=cfg.get('resource_amendment')
    if amendment:
        original_start=read(amendment['original_budget_start'])['unix'];START=time.monotonic()-(time.time()-original_start);WALL=original_start
    if not cfg['execution_enabled'] or root.exists(): raise RuntimeError('Disabled or existing attempt preserved')
    root.mkdir(parents=True);write(root/'BUDGET_START.json',dict(unix=WALL,limit=cfg['resources']['planned_main_budget_seconds']))
    try:
        receipt=read(Path(cfg['preflight_output'])/'COMPLETE.json');assert receipt['status']=='PASS' and len(receipt['records'])==150
        identity=receipt['inputs'].copy()
        if amendment:
            previous=read(amendment['prior_configuration']);assert sha(amendment['prior_configuration'])==identity[str(CONFIG)]
            old_science={k:v for k,v in previous.items() if k not in ['output','resources']};new_science={k:v for k,v in cfg.items() if k not in ['output','resources','resource_amendment']}
            assert old_science==new_science,'Resource amendment must not alter scientific recipe'
            assert {k:v for k,v in previous['resources'].items() if k!='planned_main_budget_seconds'}=={k:v for k,v in cfg['resources'].items() if k!='planned_main_budget_seconds'}
            del identity[str(CONFIG)];identity[amendment['prior_configuration']]=sha(amendment['prior_configuration']);identity[str(CONFIG)]=sha(CONFIG)
            prior=Path(amendment['prior_output']);assert read(prior/'LAUNCHER_EXIT.json')['exit_code']!=0
            first=prior/amendment['preserved_complete_case'];done=read(first/'COMPLETE.json')
            for name,h in done['files'].items():assert sha(first/name)==h;identity[str(first/name)]=h
            dest=root/amendment['preserved_complete_case'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(first,dest)
            write(root/'RESOURCE_AMENDMENT.json',dict(original_start=original_start,old_budget=28800,cumulative_budget=36000,prior_failed_refiner_epochs=5,prior_complete_Base_reused=True,new_scientific_recipe=False,budget_reset=False))
        for p,h in identity.items(): assert sha(p)==h,'Frozen preflight input changed'
        for p in [Path(__file__),Path(cfg['preflight_output'])/'COMPLETE.json',Path('scripts/run_sensor_motion_architecture_formal.sh'),Path('scripts/sensor_motion_architecture_formal_summary.py')]: identity[str(p)]=sha(p)
        write(root/'IDENTITY.json',dict(inputs=identity));sanity(cfg,parent)
        jobs=[]
        for proto in [p+'/'+f for p in cfg['protocols'] for f in cfg['folds']]:
            for seed in cfg['seeds']:
                for arm in cfg['new_base_arms']: jobs.append(dict(protocol=proto,seed=seed,kind=arm['name'],family='Base'))
        for proto in [p+'/'+f for p in cfg['protocols'] for f in cfg['folds']]:
            for seed in cfg['seeds']:
                for arm in cfg['new_neural_refiner_arms']:
                    if arm['name']=='GridIntervalPair' and receipt['grid'][proto]['no_op']: continue
                    jobs.append(dict(protocol=proto,seed=seed,kind=arm['name'],family='Refiner'))
        neural_count=len(jobs)
        # Two predetermined real fits measure both compute families before a full forecast; no duplicate pilot fit.
        first_base=next(j for j in jobs if j['protocol']=='TS/fold_1' and j['seed']==17 and j['kind']=='RecurrentOnly')
        first_ref=next(j for j in jobs if j['protocol']=='TS/fold_1' and j['seed']==17 and j['kind']=='R7Fine')
        jobs=[first_base,first_ref]+[j for j in jobs if j is not first_base and j is not first_ref]
        for proto in [p+'/'+f for p in cfg['protocols'] for f in cfg['folds']]:
            for seed in cfg['seeds']:
                for kind,family in [('LinearPair','LinearOptimization'),('SpeedThresholdPair','ThresholdFit')]: jobs.append(dict(protocol=proto,seed=seed,kind=kind,family=family))
        write(root/'MANIFEST.json',dict(jobs=jobs,neural_jobs=neural_count,total_classifier_fits=len(jobs),Grid_skipped=receipt['Grid_skipped_protocols']))
        pilot={'neural_jobs':neural_count}
        if amendment:
            old_history=read(Path(amendment['prior_output'])/amendment['preserved_complete_case']/'history.json')
            pilot['Base']=max(r['epoch_seconds'] for r in old_history[1:5])*1.25
        state=[dict(j,status='pending') for j in jobs]
        for at,job in enumerate(jobs):
            existing=root/job['protocol']/f"{job['kind']}_seed{job['seed']}"/'COMPLETE.json'
            if existing.exists():state[at]['status']='complete_reused_from_preserved_first_attempt';write(root/'QUEUE_STATE.json',state);continue
            state[at]['status']='running';write(root/'QUEUE_STATE.json',state)
            if job['family'] in ['Base','Refiner']: train_neural(cfg,parent,job['protocol'],job['kind'],job['seed'],pilot)
            elif job['kind']=='LinearPair': train_linear(cfg,parent,job['protocol'],job['seed'])
            else: train_threshold(cfg,parent,job['protocol'],job['seed'])
            state[at]['status']='complete';write(root/'QUEUE_STATE.json',state);budget(cfg,'fit_end')
        summarize(cfg,jobs,receipt)
        for p,h in identity.items(): assert sha(p)==h,'Frozen inputs changed'
        write(root/'COMPLETE.json',dict(status='full_official_ablation_matrix_complete_pending_audit',new_Base_fits=210,new_neural_refiner_fits=neural_count-210,new_linear_fits=30,new_threshold_fits=30,new_classifier_fits=len(jobs),classifier_fit_attempts=len(jobs)+(1 if amendment else 0),neural_epochs=neural_count*20,total_neural_epochs_including_failed=neural_count*20+(5 if amendment else 0),failed_partial_refiner_epochs=5 if amendment else 0,reused_cases=150,sanity_optimizer_steps=read(root/'SANITY.json')['sanity_optimizer_steps'],prior_sanity_optimizer_steps=read(Path(amendment['prior_output'])/'SANITY.json')['sanity_optimizer_steps'] if amendment else 0,seconds=budget(cfg,'complete'),TEST_scoring=True,new_blindness=False))
        print('ARCHITECTURE_COMPLETE',len(jobs),flush=True)
    except Exception as e:
        write(root/'FAILURE.json',dict(error=repr(e),seconds=time.monotonic()-START,complete=False,TEST_scoring=False));raise


if __name__=='__main__': main()
