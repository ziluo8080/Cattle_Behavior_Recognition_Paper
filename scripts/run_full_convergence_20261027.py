"""Cloud-only adaptive convergence experiment with actual per-case stopping checks."""
from pathlib import Path
import argparse,copy,gc,json,math,os,shutil,subprocess,sys,time,traceback
import numpy as np
import torch
from scripts import run_b_low_lr_20261027 as source
from scripts.sensor_bn_momentum_20261027 import configure_statistics
from scripts.sensor_state_average_20261027 import apply_window
from scripts.sensor_fusion_stability_20261027 import metric,improves
from scripts.sensor_xgb_b_fusion_20261022 import temper,fuse
from scripts.check_empirical_convergence_20261027 import strong_assess

base,core=source.base,source.core
read,save,sha=source.read,source.save,source.sha
CONFIG=Path('configs/sensor_full_convergence_20261027_v9.json')
MODULE='scripts.run_full_convergence_20261027'


def setup(cuda=False):
    assert Path.cwd()==core.ROOT and (torch.__version__,np.__version__)==('2.4.1+cu121','1.24.4')
    if cuda:core.configure()
    else:torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    cfg=read(CONFIG)
    assert all(not cfg[k] for k in ['Test_allowed','outer_VAL_allowed','formal_refit_allowed','new_paid_resources'])
    return cfg,read(cfg['source_config'])


def implementation(cfg):
    files=dict(read(cfg['source_seal'])['code'])
    for p,d in files.items():assert sha(p)==d,p
    for name in [CONFIG, 'scripts/run_full_convergence_20261027.py','scripts/check_empirical_convergence_20261027.py',
                 'scripts/sensor_state_average_20261027.py','.aris/compute/bitahub-root-full-convergence-20261027.md']:
        files[str(name)]=sha(name)
    return files


def new(cfg,parent,data,seed,device):
    state=source.new(parent,data,'Constant_3e4',seed,device)
    configure_statistics(state,cfg['BN_momentum'])
    for group in state['opt'].param_groups:group.update(lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    state['scheduler']=torch.optim.lr_scheduler.ReduceLROnPlateau(state['opt'],**cfg['scheduler'])
    return state


def update(state,data,ids):
    result=base.update(state['model'],state['opt'],state['ema'],data,ids)
    state['step']+=1
    return result


def snapshot(state,epoch):
    return dict(epoch=epoch,model={n:v.detach().cpu().clone() for n,v in state['ema'].model.state_dict().items()})


def clone(state,history,epoch):
    model=copy.deepcopy(state['ema'].model).eval().requires_grad_(False)
    apply_window(model,history,epoch)
    return model


def checkpoint(path,state,**extra):
    source.checkpoint(path,state,scheduler=state['scheduler'].state_dict(),BN_momentum=.01,**extra)


def restore(cfg,parent,data,payload,device):
    state=new(cfg,parent,data,17,device)
    state['model'].load_state_dict(payload['model']);state['ema'].model.load_state_dict(payload['ema'])
    state['opt'].load_state_dict(copy.deepcopy(payload['optimizer']))
    state['ema'].updates=payload['ema_updates'];state['step']=payload['global_step']
    state['scheduler'].load_state_dict(copy.deepcopy(payload['scheduler']))
    core.restore_rng(payload['rng'])
    return state


def witness(cfg,parent,cpu):
    lock=None if cpu else core.acquire_gpu();device='cpu' if cpu else 'cuda'
    core.seed_all(17);n=17;dynamic=torch.randn(n,100,8,device=device);dynamic[:,:,6]=1
    s=torch.randn(n,60,26,device=device)
    data=dict(prior=np.full(7,1/7),train=np.arange(n),evaluation=np.arange(n),weights=torch.arange(1,8,dtype=torch.float32,device=device),
        tensors=dict(dynamic=dynamic,state=s,state_valid=torch.ones(n,60,dtype=torch.bool,device=device),context=torch.randn(n,136,device=device),current=s[:,-1],labels=torch.arange(n,device=device)%7))
    a=source.new(parent,data,'Constant_3e4',17,device);expected=source.step(parent,a,data,data['train']);rng=core.rng()
    b=new(cfg,parent,data,17,device);actual=update(b,data,data['train'])
    base.equal_tree(expected,actual);base.equal_tree(rng,core.rng())
    from scripts.sensor_bn_momentum_20261027 import equal_learned_state
    equal_learned_state(base,a,b)
    # Force scheduler reductions on synthetic measurements, then verify resume.
    rates=[]
    for score in [1.0]*15:
        b['scheduler'].step(score);rates.append(b['opt'].param_groups[0]['lr'])
    assert rates[-1]<rates[0] and rates[-1]>=cfg['scheduler']['min_lr']
    folder=Path(cfg['output'])/('cpu_work' if cpu else 'cuda_work');assert not folder.exists();folder.mkdir(parents=True)
    checkpoint(folder/'state.pt',b,epoch=0)
    payload=torch.load(folder/'state.pt',map_location='cpu',weights_only=False)
    frozen=copy.deepcopy(payload);h=[snapshot(b,i) for i in range(9,12)];live=copy.deepcopy(b['ema'].model.state_dict());rng=core.rng()
    evaluation=clone(b,h,11);base.infer(evaluation,data,data['evaluation']);base.equal_tree(live,b['ema'].model.state_dict());base.equal_tree(rng,core.rng())
    expected=update(b,data,data['train']);b['scheduler'].step(1.0);rng=core.rng()
    c=restore(cfg,parent,data,payload,device);actual=update(c,data,data['train']);c['scheduler'].step(1.0)
    base.equal_tree(expected,actual);base.equal_tree(rng,core.rng());equal_learned_state(base,b,c)
    base.equal_tree(b['scheduler'].state_dict(),c['scheduler'].state_dict());base.equal_tree(frozen,payload)
    record=dict(status='PASS',CUDA_validated=not cpu,real_data_rows_read=0,parameters=213119,
        native_update_exact=True,scheduler_reduction_positive_floor=True,resume_model_Adam_EMA_scheduler_RNG_exact=True,
        evaluation_preserves_live_state_RNG=True,implementation_files=implementation(cfg))
    save(Path(cfg['output'])/('INDEPENDENT_RUNTIME_WITNESS.json' if cpu else 'RUNTIME_WITNESS.json'),record)
    if lock:lock.close()
    print('FULL_CONVERGENCE_RUNTIME_WITNESS_PASS',flush=True)


def prepare(cfg):
    out=Path(cfg['output']);assert not (out/'SEAL.json').exists()
    files=read(cfg['source_seal'])['files']
    for p,d in files.items():assert sha(p)==d,p
    implementation_files=implementation(cfg)
    for name in ['INDEPENDENT_RUNTIME_WITNESS.json','RUNTIME_WITNESS.json']:
        receipt=read(out/name);assert receipt['status']=='PASS' and receipt['implementation_files']==implementation_files
    disk=shutil.disk_usage('.').free/1024**3;assert disk>=cfg['budget']['minimum_free_disk_GiB']
    save(out/'SEAL.json',dict(config=cfg,code=implementation_files,files=files,free_disk_GiB=disk,utc=time.time()))


def check(cfg):
    seal=read(Path(cfg['output'])/'SEAL.json');assert seal['config']==cfg and seal['code']==implementation(cfg)


def relative_change(model,before):
    numerator=sum((v.detach().double()-before[n].double()).square().sum() for n,v in model.named_parameters())
    denominator=sum(v.detach().double().square().sum() for v in model.parameters())
    return float((numerator/denominator).sqrt())


def worker(cfg,parent,p,k,seed):
    check(cfg);lock=core.acquire_gpu();started=time.monotonic()
    folder=Path(cfg['output'])/'trajectories'/p/f'inner{k}'/cfg['candidate']/f'seed{seed}'
    assert not (folder/'STARTED.json').exists();save(folder/'STARTED.json',dict(pid=os.getpid(),unix=time.time()))
    data=source.load(parent,p,k);d=Path(parent['data_parent'])/'data'/p/f'inner{k}'
    target=d/'DEV_TARGETS.jsonl';tree=Path(parent['data_parent'])/'baseline'/p/f'inner{k}'/'XGB.npy'
    files=read(Path(cfg['output'])/'SEAL.json')['files']
    for f in [target,tree]:assert sha(f)==files[str(f)]
    y=np.array([json.loads(line)['label']-1 for line in target.read_text().splitlines()]);px=np.load(tree)
    fy=data['tensors']['labels'][torch.as_tensor(data['train'],device='cuda')].cpu().numpy();w=data['weights'].double().cpu().numpy()
    state=new(cfg,parent,data,seed,'cuda');history=[];snaps=[];best=None;best_epoch=None;last_prob=None
    for epoch in range(1,cfg['maximum_epochs']+1):
        tick=time.monotonic();rate=state['opt'].param_groups[0]['lr'];before={n:v.detach().clone() for n,v in state['model'].named_parameters()};ebefore={n:v.detach().clone() for n,v in state['ema'].model.named_parameters()}
        order=np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(data['train']);num=den=0.
        for i in range(0,len(order),256):
            loss,mass=update(state,data,order[i:i+256]);num+=loss*mass;den+=mass
        raw_delta=relative_change(state['model'],before);ema_delta=relative_change(state['ema'].model,ebefore)
        snaps.append(snapshot(state,epoch));snaps=snaps[-3:];rng=core.rng();evaluation=clone(state,snaps,epoch)
        fit=base.infer(evaluation,data,data['train']);dev=base.infer(evaluation,data,data['evaluation'])
        fm,_=metric(fit,fy,w);dm,exact=metric(dev,y,w);fp=fuse(temper(px,2.),temper(dev,1.),'margin_average',.7);gm,_=metric(fp,y,w)
        ep=folder/f'epoch{epoch}';ep.mkdir();torch.save(dict(model=evaluation.state_dict(),epoch=epoch),ep/'ema.pt')
        evaluation.load_state_dict(torch.load(ep/'ema.pt',map_location='cpu',weights_only=False)['model'])
        np.testing.assert_array_equal(dev,base.infer(evaluation,data,data['evaluation']))
        np.save(ep/'held_ema.npy',dev);np.save(ep/'fusion_ema.npy',fp);core.restore_rng(rng)
        row=dict(epoch=epoch,learning_rate=rate,train_ce=num/den,fit_ema=fm,dev_ema=dm,fusion_ema=gm,
            raw_relative_parameter_change=raw_delta,EMA_relative_parameter_change=ema_delta,
            neural_prediction_flip_fraction=None if last_prob is None else float((dev.argmax(1)!=last_prob.argmax(1)).mean()),
            global_step=state['step'],seconds=time.monotonic()-tick)
        history.append(row);last_prob=dev
        if improves(exact,best):
            best=exact;best_epoch=epoch;shutil.copyfile(ep/'ema.pt',folder/'best_ema.pt');save(folder/'BEST.json',dict(epoch=epoch,criterion='exact_neural_DEV_macro_F1',ties='earlier'))
        state['scheduler'].step(dm['weighted_CE']);row['next_learning_rate']=state['opt'].param_groups[0]['lr']
        save(folder/'history.json',history);checkpoint(folder/'latest.pt',state,epoch=epoch,causal_ema_snapshots=snaps)
        assessment=strong_assess(history)
        save(folder/'PROGRESS.json',dict(protocol=p,inner=k,seed=seed,epoch=epoch,convergence=assessment,elapsed_seconds=time.monotonic()-started))
        print(dict(protocol=p,inner=k,seed=seed,epoch=epoch,LR=rate,TRAIN_CE=row['train_ce'],DEV_CE=dm['weighted_CE'],F1=dm['macro_f1_percent'],Fusion_F1=gm['macro_f1_percent'],converged=assessment['passed']),flush=True)
        del evaluation,before,ebefore
        assert time.monotonic()-started<cfg['budget']['job_seconds']
        if assessment['passed']:break
    save(folder/'CONVERGENCE.json',assessment)
    save(folder/'COMPLETE.json',dict(status='COMPLETE',empirical_convergence=assessment['passed'],epochs=epoch,best_epoch=best_epoch,
        official_Test_rows_read=0,elapsed_seconds=time.monotonic()-started,
        files={str(folder/n):sha(folder/n) for n in ['history.json','latest.pt','best_ema.pt','BEST.json','CONVERGENCE.json']}))
    lock.close()


def pilot(cfg):
    check(cfg);out=Path(cfg['output']);assert not (out/'PILOT_LAUNCH.json').exists();save(out/'PILOT_LAUNCH.json',dict(pid=os.getpid(),unix=time.time()))
    rows=[]
    for case in cfg['pilot_cases']:
        p,k,s=case['protocol'],case['inner'],case['seed'];log=out/f'pilot_{p}_{k}_{s}.log'
        with log.open('wb') as stream:
            result=subprocess.run([sys.executable,'-s','-u','-m',MODULE,'--mode','worker','--protocol',p,'--inner',str(k),'--seed',str(s)],stdout=stream,stderr=subprocess.STDOUT,timeout=cfg['budget']['job_seconds']+60)
        assert result.returncode==0,f'Inspect{log}; no retry'
        path=out/'trajectories'/p/f'inner{k}'/cfg['candidate']/f'seed{s}';h=read(path/'history.json');done=read(path/'COMPLETE.json')
        reference=read(Path(cfg['accuracy_guard']['reference'])/'trajectories'/p/f'inner{k}'/'CausalStateAverage3'/f'seed{s}'/'history.json')[-1]['fusion']['StateAverage3_MarginAverage']['macro_f1_percent']
        delta=h[-1]['fusion_ema']['macro_f1_percent']-reference
        rows.append(dict(**case,empirical_convergence=done['empirical_convergence'],epochs=done['epochs'],endpoint_fusion_delta_pp=delta,accuracy_guard_passed=delta>=-.2))
        save(out/'PILOT_PROGRESS.json',dict(records=rows))
    save(out/'PILOT_COMPLETE.json',dict(status='COMPLETE_PILOT',records=rows,full_run_trigger=all(r['empirical_convergence'] and r['accuracy_guard_passed'] for r in rows)))
    print('FULL_CONVERGENCE_PILOT_COMPLETE',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',required=True,choices=['cpu_witness','witness','prepare','worker','pilot'])
    parser.add_argument('--protocol',choices=['TS','OS']);parser.add_argument('--inner',type=int,choices=[0,1,2]);parser.add_argument('--seed',type=int,choices=[17,23,29]);args=parser.parse_args()
    cfg,parent=setup(args.mode in ['witness','worker'])
    if args.mode in ['cpu_witness','witness']:witness(cfg,parent,args.mode=='cpu_witness')
    elif args.mode=='prepare':prepare(cfg)
    elif args.mode=='worker':worker(cfg,parent,args.protocol,args.inner,args.seed)
    else:pilot(cfg)


if __name__=='__main__':main()
