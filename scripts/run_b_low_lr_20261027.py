"""Bounded lower-LR experiment, reusing the sealed ConvTran-ERI and XGB inputs."""
import argparse,copy,gc,math,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from scripts import run_b_early_schedule_20261024 as old
from scripts.sensor_fusion_stability_20261027 import metric,fusion_metric,improves

base=old.base;core=old.core
read,save,sha=old.read,old.save,old.sha
CONFIG=Path('configs/sensor_b_low_lr_20261027_v1.json')
MODULE='scripts.run_b_low_lr_20261027'

def configure(cuda=False):
    assert Path.cwd()==core.ROOT,'Scientific execution remains cloud-only'
    assert (torch.__version__,np.__version__)==('2.4.1+cu121','1.24.4')
    if cuda:core.configure()
    else:torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    cfg=read(CONFIG)
    assert cfg['arms']=={'Constant_3e4':.0003,'Constant_1e4':.0001} and cfg['epochs']==20
    assert all(cfg[k] is False for k in ['Test_allowed','outer_VAL_allowed','formal_refit_allowed','automatic_expansion','new_paid_resources'])
    return cfg

def code(cfg):
    files=dict(read(cfg['source_seal'])['code'])
    for p,d in files.items():assert sha(p)==d,p
    names=[CONFIG,Path(__file__).relative_to(core.ROOT),Path('scripts/sensor_fusion_stability_20261027.py'),Path('scripts/report_b_low_lr_20261027.py'),Path('tests/test_fusion_stability_20261027.py'),Path('.aris/compute/bitahub-root-b-low-lr-20261027.md')]
    files.update({str(p):sha(p) for p in names});return files

def new(cfg,data,arm,seed,device):
    core.seed_all(seed);model=base.native(cfg,data['prior'],zero=False,device=device)
    bcfg=dict(read(cfg['B_config']),learning_rate=cfg['arms'][arm])
    opt=base.optimizer_for(model,bcfg);ema=base.ParameterEMA(model,.998,200)
    return dict(model=model,opt=opt,ema=ema,step=0,arm=arm)

def step(cfg,state,data,ids):
    assert all(g['lr']==cfg['arms'][state['arm']] for g in state['opt'].param_groups)
    result=base.update(state['model'],state['opt'],state['ema'],data,ids);state['step']+=1;return result

def restore(cfg,data,payload,device):
    state=new(cfg,data,payload['arm'],17,device);state['model'].load_state_dict(payload['model'])
    state['opt'].load_state_dict(copy.deepcopy(payload['optimizer']))
    state['ema'].model.load_state_dict(payload['ema']);state['ema'].updates=payload['ema_updates'];state['step']=payload['global_step']
    core.restore_rng(payload['rng']);return state

def checkpoint(path,state,**extra):
    base.checkpoint(path,state['model'],state['opt'],state['ema'],global_step=state['step'],arm=state['arm'],**extra)

def observe(state,data):
    rng=core.rng()
    try:
        fit=base.infer(state['ema'].model,data,data['train'])
        dev=base.infer(state['ema'].model,data,data['evaluation'])
        return fit,dev
    finally:core.restore_rng(rng)

def synthetic(cfg,output,cpu=False):
    device='cpu' if cpu else 'cuda';lock=None if cpu else core.acquire_gpu()
    core.seed_all(17);n=67;dyn=torch.randn(n,100,8,device=device);dyn[:,:,6]=1;dyn[:,:,7]=.1
    sta=torch.randn(n,60,26,device=device)
    data=dict(prior=np.full(7,1/7),train=np.arange(n),evaluation=np.arange(n),weights=torch.ones(7,device=device),
        tensors=dict(dynamic=dyn,state=sta,state_valid=torch.ones(n,60,dtype=torch.bool,device=device),context=torch.randn(n,136,device=device),current=sta[:,-1],labels=torch.arange(n,device=device)%7))
    folder=Path(output).parent/(Path(output).stem+'_work_'+str(time.time_ns()));folder.mkdir(parents=True)
    initial=None;records=[]
    for arm in cfg['arms']:
        s=new(cfg,data,arm,17,device)
        if initial is None:initial=copy.deepcopy(s['model'].state_dict())
        else:base.equal_tree(initial,s['model'].state_dict())
        step(cfg,s,data,data['train']);checkpoint(folder/(arm+'.pt'),s,epoch=1)
        payload=torch.load(folder/(arm+'.pt'),map_location='cpu',weights_only=False);original=copy.deepcopy(payload)
        fit,dev=observe(s,data);expected=step(cfg,s,data,data['train']);expected_rng=core.rng()
        r=restore(cfg,data,payload,device);actual=step(cfg,r,data,data['train'])
        base.equal_tree(expected,actual);base.equal_tree(expected_rng,core.rng());old.extension.equal_bundles((s['model'],s['opt'],s['ema']),(r['model'],r['opt'],r['ema']))
        base.equal_tree(original,payload)
        a=restore(cfg,data,payload,device);np.testing.assert_array_equal(dev,base.infer(a['ema'].model,data,data['evaluation']))
        labels=np.arange(n)%7;m,score=metric(dev,labels,np.ones(7));fm,_,_=fusion_metric(np.full_like(dev,1/7),dev,labels,np.ones(7))
        assert m['weighted_CE'] is not None and fm['macro_f1_percent']>=0
        records.append(dict(arm=arm,rate=cfg['arms'][arm],model_optimizer_EMA_RNG_next_update_exact=True,observation_does_not_perturb_training=True,source_payload_unchanged=True,EMA_reload_tail67_exact=True))
    save(output,dict(status='PASS',CUDA_validated=not cpu,parameters=213119,real_data_rows_read=0,initialization_matched=True,records=records,implementation_files=code(cfg)))
    if lock:lock.close()
    print('B_LOW_LR_WITNESS_PASS',flush=True)

def prepare(cfg):
    out=Path(cfg['output']);assert not (out/'PREPARED.json').exists();source=read(cfg['source_seal']);files={cfg['source_seal']:sha(cfg['source_seal'])}
    code(cfg)
    for p in cfg['protocols']:
        for k in cfg['inners']:
            d=Path(cfg['data_parent'])/'data'/p/f'inner{k}';receipt=d/'RECEIPT.json'
            assert sha(receipt)==source['files'][str(receipt)];files[str(receipt)]=sha(receipt)
            for name,digest in read(receipt)['files'].items():
                path=d/name;assert sha(path)==digest;files[str(path)]=digest
            for name in ['XGB.npy']:
                path=Path(cfg['data_parent'])/'baseline'/p/f'inner{k}'/name
                assert sha(path)==source['files'][str(path)];files[str(path)]=sha(path)
    assert shutil.disk_usage('.').free/1024**3>=cfg['budget']['minimum_free_disk_GiB']
    save(out/'PREPARED.json',dict(status='PASS',files=files,new_scientific_fits=0))

def load(cfg,p,k):return old.data_for(cfg,p,k)

def preflight(cfg):
    out=Path(cfg['output']);assert not (out/'PREFLIGHT.json').exists();lock=core.acquire_gpu();records=[]
    for p in cfg['protocols']:
        for k in cfg['inners']:
            data=load(cfg,p,k);s=new(cfg,data,'Constant_3e4',17,'cuda');order=np.random.default_rng(np.random.SeedSequence([17,1])).permutation(data['train'])
            times=[]
            for j in range(3):
                tick=time.monotonic();step(cfg,s,data,order[j*256:(j+1)*256]);torch.cuda.synchronize();times.append(time.monotonic()-tick)
            tick=time.monotonic();observe(s,data);torch.cuda.synchronize();inference=time.monotonic()-tick
            estimate=(max(times[1:])*math.ceil(len(data['train'])/256)+inference+.4)*20*1.5+30
            assert estimate<cfg['budget']['job_seconds'];records.append(dict(protocol=p,inner=k,step_seconds=times,EMA_fit_DEV_inference_seconds=inference,estimated_job_seconds=estimate))
            del data,s;gc.collect();torch.cuda.empty_cache()
    lock.close();total=sum(r['estimated_job_seconds']*6 for r in records)+300
    assert total<cfg['budget']['total_seconds']
    save(out/'PREFLIGHT.json',dict(status='PASS',records=records,estimated_total_seconds=total,new_candidate_scores=0,implementation_files=code(cfg)))

def seal(cfg):
    out=Path(cfg['output']);assert not (out/'SEAL.json').exists();implementation=code(cfg);evidence={}
    for name in ['RUNTIME_WITNESS.json','INDEPENDENT_RUNTIME_WITNESS.json','PREFLIGHT.json']:
        r=read(out/name);assert r['status']=='PASS' and r['implementation_files']==implementation;evidence[name]=sha(out/name)
    files=read(out/'PREPARED.json')['files']
    for p,d in files.items():assert sha(p)==d,p
    save(out/'SEAL.json',dict(config=cfg,code=implementation,files=files,evidence=evidence,created_unix=time.time(),new_candidate_scores=0))

def check(cfg,full=False):
    r=read(Path(cfg['output'])/'SEAL.json');assert r['config']==cfg and r['code']==code(cfg)
    for n,d in r['evidence'].items():assert sha(Path(cfg['output'])/n)==d
    if full:
        for n,d in r['files'].items():assert sha(n)==d,n

def worker(cfg,p,k,arm,seed):
    check(cfg);lock=core.acquire_gpu();started=time.monotonic();folder=Path(cfg['output'])/'trajectories'/p/f'inner{k}'/arm/f'seed{seed}'
    assert not (folder/'STARTED.json').exists();save(folder/'STARTED.json',dict(pid=os.getpid(),unix=time.time()))
    data=load(cfg,p,k);d=Path(cfg['data_parent'])/'data'/p/f'inner{k}'
    truth=np.asarray([readrow['label']-1 for readrow in [__import__('json').loads(s) for s in (d/'DEV_TARGETS.jsonl').read_text().splitlines()]])
    fittruth=data['tensors']['labels'][torch.as_tensor(data['train'],device='cuda')].cpu().numpy()
    sealed=read(Path(cfg['output'])/'SEAL.json')['files']
    xgb_path=Path(cfg['data_parent'])/'baseline'/p/f'inner{k}'/'XGB.npy'
    for path in [d/'DEV_TARGETS.jsonl',xgb_path]:assert sha(path)==sealed[str(path)]
    weights=data['weights'].double().cpu().numpy();px=np.load(xgb_path)
    s=new(cfg,data,arm,seed,'cuda');history=[];best=None;best_epoch=None;previous=None
    for epoch in range(1,21):
        tick=time.monotonic();order=np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(data['train']);num=den=0.
        for start in range(0,len(order),256):
            loss,mass=step(cfg,s,data,order[start:start+256]);num+=loss*mass;den+=mass
        norm=float(torch.stack([v.grad.detach().double().square().sum() for v in s['model'].parameters()]).sum().sqrt())
        assert math.isfinite(norm)
        fit,dev=observe(s,data);fitm,_=metric(fit,fittruth,weights);devm,exact=metric(dev,truth,weights)
        fusion,previous,gate=fusion_metric(px,dev,truth,weights,previous)
        ep=folder/f'epoch{epoch}';ep.mkdir();torch.save(dict(model=s['ema'].model.state_dict(),epoch=epoch,global_step=s['step'],arm=arm),ep/'ema.pt')
        np.save(ep/'held_ema.npy',dev);np.save(ep/'fusion_ema.npy',gate)
        # Reload check uses a matched model and restores RNG to preserve the next update.
        rng=core.rng();net=base.native(cfg,data['prior'],zero=False,device='cuda').eval().requires_grad_(False);net.load_state_dict(torch.load(ep/'ema.pt',map_location='cpu',weights_only=False)['model'])
        np.testing.assert_array_equal(dev,base.infer(net,data,data['evaluation']));del net;core.restore_rng(rng)
        save(ep/'COMPLETE.json',dict(epoch=epoch,EMA_reload_exact=True,files={str(ep/n):sha(ep/n) for n in ['ema.pt','held_ema.npy','fusion_ema.npy']}))
        row=dict(epoch=epoch,train_ce=num/den,learning_rate=cfg['arms'][arm],last_batch_gradient_l2=norm,fit_ema=fitm,dev_ema=devm,fusion_ema=fusion,global_step=s['step'],seconds=time.monotonic()-tick)
        if improves(exact,best):
            best=exact;best_epoch=epoch;shutil.copyfile(ep/'ema.pt',folder/'best_ema.pt');save(folder/'BEST.json',dict(epoch=epoch,criterion='neural_DEV_macro_F1',score_percent=float(best),checkpoint_sha256=sha(folder/'best_ema.pt')))
        history.append(row);save(folder/'history.json',history);checkpoint(folder/'latest.pt',s,epoch=epoch)
        save(folder/'PROGRESS.json',dict(epoch=epoch,protocol=p,inner=k,arm=arm,seed=seed,best_epoch=best_epoch,elapsed_seconds=time.monotonic()-started))
        print(dict(protocol=p,inner=k,arm=arm,seed=seed,epoch=epoch,train_ce=row['train_ce'],EMA_DEV_CE=devm['weighted_CE'],EMA_DEV_F1=devm['macro_f1_percent'],fusion_F1=fusion['macro_f1_percent']),flush=True)
        assert time.monotonic()-started<cfg['budget']['job_seconds'],'Job budget exceeded'
    check(cfg);save(folder/'COMPLETE.json',dict(status='COMPLETE',protocol=p,inner=k,arm=arm,seed=seed,epochs=20,best_epoch=best_epoch,elapsed_seconds=time.monotonic()-started,files={str(folder/n):sha(folder/n) for n in ['history.json','latest.pt','best_ema.pt','BEST.json']}));lock.close()

def queue(cfg):
    import signal
    out=Path(cfg['output']);check(cfg,True);assert not (out/'LAUNCH.json').exists();started=time.monotonic();jobs=[];active=None
    for p in cfg['protocols']:
        for k in cfg['inners']:
            for arm in cfg['arms']:
                for seed in cfg['seeds']:jobs.append(dict(protocol=p,inner=k,arm=arm,seed=seed,status='pending'))
    save(out/'LAUNCH.json',dict(pid=os.getpid(),unix=time.time(),jobs=36,maximum_epochs=720))
    def snapshot(stage):save(out/'PUBLIC_PROGRESS.json',dict(stage=stage,completed=sum(j['status']=='completed' for j in jobs),total=len(jobs),elapsed_seconds=time.monotonic()-started,jobs=jobs))
    try:
        for j in jobs:
            assert time.monotonic()-started<cfg['budget']['total_seconds']
            log=out/'logs'/f"{j['protocol']}_{j['inner']}_{j['arm']}_{j['seed']}.log";log.parent.mkdir(exist_ok=True)
            with log.open('wb') as f:active=subprocess.Popen([sys.executable,'-s','-u','-m',MODULE,'--mode','worker','--protocol',j['protocol'],'--inner',str(j['inner']),'--arm',j['arm'],'--seed',str(j['seed'])],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            job_started=time.monotonic();j.update(status='running',pid=active.pid);snapshot('RUNNING')
            while active.poll() is None:
                snapshot('RUNNING')
                if time.monotonic()-started>cfg['budget']['total_seconds']:raise RuntimeError('Total budget exceeded')
                if time.monotonic()-job_started>cfg['budget']['job_seconds']:raise RuntimeError('Job watchdog budget exceeded')
                time.sleep(5)
            assert active.returncode==0,f'Worker failed; inspect {log}; no automatic retry'
            j['status']='completed';snapshot('RUNNING')
        check(cfg,True)
        from scripts.report_b_low_lr_20261027 import report
        save(out/'COMPLETE.json',report(cfg));snapshot('COMPLETE_DEVELOPMENT_COMPARISON');print('LOW_LR_TRAINING_COMPLETE',flush=True)
    except BaseException:
        if active is not None and active.poll() is None:os.killpg(active.pid,signal.SIGTERM);active.wait(timeout=30)
        snapshot('FAILED_STOP_FOR_INSPECTION');save(out/'FAILURE.json',dict(traceback=traceback.format_exc()));raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',required=True,choices=['cpu_witness','witness','prepare','preflight','seal','worker','queue','launch']);ap.add_argument('--witness-output');ap.add_argument('--protocol',choices=['TS','OS']);ap.add_argument('--inner',type=int,choices=[0,1,2]);ap.add_argument('--arm',choices=['Constant_3e4','Constant_1e4']);ap.add_argument('--seed',type=int,choices=[17,23,29]);a=ap.parse_args()
    cfg=configure(a.mode in ['witness','preflight','worker']);out=Path(cfg['output'])
    if a.mode in ['cpu_witness','witness']:synthetic(cfg,a.witness_output or out/'RUNTIME_WITNESS.json',a.mode=='cpu_witness')
    elif a.mode=='prepare':prepare(cfg)
    elif a.mode=='preflight':preflight(cfg)
    elif a.mode=='seal':seal(cfg)
    elif a.mode=='worker':
        assert a.protocol in cfg['protocols'] and a.inner in cfg['inners'] and a.arm in cfg['arms'] and a.seed in cfg['seeds'];worker(cfg,a.protocol,a.inner,a.arm,a.seed)
    elif a.mode=='queue':queue(cfg)
    else:
        check(cfg);assert not any((out/n).exists() for n in ['DISPATCH.json','LAUNCH.json'])
        with (out/'queue.log').open('wb') as f:child=subprocess.Popen([sys.executable,'-s','-u','-m',MODULE,'--mode','queue'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        save(out/'DISPATCH.json',dict(pid=child.pid,unix=time.time()));print(child.pid)
if __name__=='__main__':main()
