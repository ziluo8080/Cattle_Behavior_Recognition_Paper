"""From-scratch dense early curves with matched constant/cosine schedules."""
import argparse
import copy
import gc
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import torch

from scripts import run_b_epoch_extension_20261024 as extension
from scripts.sensor_b_lr_schedule_20261024 import ARMS,learning_rate,checks

base=extension.base;core=base.core
read,save,sha=extension.read,extension.save,extension.sha
CONFIG=Path('configs/sensor_b_early_schedule_20261024_v1.json')
MODULE='scripts.run_b_early_schedule_20261024'


def configure(cuda=False):
    assert Path.cwd()==core.ROOT
    assert (torch.__version__,np.__version__)==('2.4.1+cu121','1.24.4')
    if cuda:core.configure()
    else:torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    cfg=read(CONFIG)
    assert cfg['arms']==list(ARMS) and cfg['epochs']==cfg['schedule_horizon_epochs']==40
    assert cfg['protocols']==['TS','OS'] and cfg['inners']==[0,1,2] and cfg['seeds']==[17,23,29]
    assert cfg['trajectories']==36 and cfg['learning_rate']==.001 and cfg['minimum_learning_rate']==.00001
    assert all(cfg[n] is False for n in ['Test_allowed','outer_VAL_allowed','formal_refit_allowed','automatic_expansion','dynamic_early_stopping'])
    return cfg


def implementation(cfg):
    files=dict(read(cfg['source_seal'])['code'])
    for p,d in files.items():assert sha(p)==d,p
    paths=[CONFIG,Path(cfg['plan']),Path('.aris/compute/bitahub-root-b-early-schedule-20261024.md')]
    paths += list(Path('scripts').glob('*b_early_schedule*20261024.py'))+[Path('scripts/sensor_b_lr_schedule_20261024.py')]
    files.update({str(p):sha(p) for p in paths});return files


def data_for(cfg,p,k):
    parent_cfg=read(Path(cfg['data_parent'])/'SEAL.json')['config']
    return base.load_data(parent_cfg,p,k)


def create(cfg,data,arm,seed,device):
    core.seed_all(seed)
    model=base.native(cfg,data['prior'],zero=False,device=device)
    opt=base.optimizer_for(model,read(cfg['B_config']))
    ema=base.ParameterEMA(model,.998,200)
    return dict(model=model,opt=opt,ema=ema,arm=arm,step=0,total=40*math.ceil(len(data['train'])/256))


def update(cfg,state,data,ids):
    rate=learning_rate(state['arm'],state['step'],state['total'],cfg['learning_rate'],cfg['minimum_learning_rate'])
    for group in state['opt'].param_groups:group['lr']=rate
    loss,mass=base.update(state['model'],state['opt'],state['ema'],data,ids)
    state['step']+=1
    return loss,mass,rate


def checkpoint(path,state,**metadata):
    assert metadata.get('arm',state['arm'])==state['arm']
    metadata=dict(metadata,arm=state['arm'])
    base.checkpoint(path,state['model'],state['opt'],state['ema'],global_step=state['step'],total_steps=state['total'],**metadata)


def restore(cfg,data,payload,device):
    model,opt,ema=extension.bundle(cfg,data,payload,device)
    assert payload['total_steps']==40*math.ceil(len(data['train'])/256)
    return dict(model=model,opt=opt,ema=ema,arm=payload['arm'],step=payload['global_step'],total=payload['total_steps'])


def equal_states(a,b):
    extension.equal_bundles((a['model'],a['opt'],a['ema']),(b['model'],b['opt'],b['ema']))
    assert (a['step'],a['total'],a['arm'])==(b['step'],b['total'],b['arm'])


def capture(cfg,state,data,folder,epoch,metadata):
    folder=Path(folder);assert not (folder/'COMPLETE.json').exists();folder.mkdir(parents=True,exist_ok=True)
    before=core.rng();files={};raw=epoch in cfg['raw_diagnostic_epochs']
    if raw:checkpoint(folder/'full.pt',state,epoch=epoch,**metadata);files[str(folder/'full.pt')]=sha(folder/'full.pt')
    torch.save(dict(model=state['ema'].model.state_dict(),epoch=epoch,global_step=state['step'],schedule_horizon_epochs=40,**metadata),folder/'ema.pt')
    files[str(folder/'ema.pt')]=sha(folder/'ema.pt')
    try:
        prob=base.infer(state['ema'].model,data,data['evaluation'])
        other=base.native(cfg,data['prior'],zero=False,device=data['tensors']['dynamic'].device).eval().requires_grad_(False)
        other.load_state_dict(torch.load(folder/'ema.pt',map_location='cpu',weights_only=False)['model'])
        np.testing.assert_array_equal(prob,base.infer(other,data,data['evaluation']));del other
        np.save(folder/'held_ema.npy',prob);files[str(folder/'held_ema.npy')]=sha(folder/'held_ema.npy')
        if raw:
            prob=base.infer(state['model'],data,data['evaluation'])
            other=base.native(cfg,data['prior'],zero=False,device=data['tensors']['dynamic'].device).eval().requires_grad_(True)
            other.load_state_dict(torch.load(folder/'full.pt',map_location='cpu',weights_only=False)['model'])
            np.testing.assert_array_equal(prob,base.infer(other,data,data['evaluation']));del other
            np.save(folder/'held_raw.npy',prob);files[str(folder/'held_raw.npy')]=sha(folder/'held_raw.npy')
    finally:core.restore_rng(before)
    base.equal_tree(before,core.rng())
    record=dict(status='COMPLETE',epoch=epoch,raw_saved=raw,EMA_reload_exact=True,raw_reload_exact=True if raw else None,
                rng_preserved=True,global_step=state['step'],files=files,**metadata)
    save(folder/'COMPLETE.json',record);return record


def restore_check(cfg,data,arm,folder,device):
    state=create(cfg,data,arm,17,device);ids=data['train'][:67]
    update(cfg,state,data,ids);checkpoint(Path(folder)/'partial.pt',state,epoch=0)
    payload=torch.load(Path(folder)/'partial.pt',map_location='cpu',weights_only=False)
    original=copy.deepcopy(payload['optimizer'])
    capture(cfg,state,data,Path(folder)/'capture',5,dict(scope='synthetic/restore diagnostic only'))
    expected=update(cfg,state,data,ids);expected_rng=core.rng()
    other=restore(cfg,data,payload,device);actual=update(cfg,other,data,ids)
    base.equal_tree(expected,actual);base.equal_tree(expected_rng,core.rng());equal_states(state,other)
    base.equal_tree(original,payload['optimizer'])
    for x,y in [(state['model'],other['model']),(state['ema'].model,other['ema'].model)]:base.equal_tree(base.infer(x,data,ids),base.infer(y,data,ids))
    return dict(model_Adam_EMA_RNG_schedule_next_update_exact=True,source_optimizer_unchanged=True,
                dense_prediction_does_not_change_next_update=True,raw_EMA_tail67_exact=True)


def witness(cfg,output,cpu=False):
    device='cpu' if cpu else 'cuda';lock=None if cpu else core.acquire_gpu();core.seed_all(17)
    n=67;dynamic=torch.randn(n,100,8,device=device);dynamic[:,:,6]=1.;dynamic[:,:,7]=.1
    state=torch.randn(n,60,26,device=device)
    data=dict(prior=np.full(7,1/7),train=np.arange(n),evaluation=np.arange(n),weights=torch.ones(7,device=device),
       tensors=dict(dynamic=dynamic,state=state,state_valid=torch.ones(n,60,dtype=torch.bool,device=device),context=torch.randn(n,136,device=device),current=state[:,-1],labels=torch.arange(n,device=device)%7))
    folder=Path(output).parent/(Path(output).stem+'_work_'+str(time.time_ns()));records=[]
    for arm in ARMS:records.append(dict(arm=arm,**restore_check(cfg,data,arm,folder/arm,device)))
    a=create(cfg,data,'Constant',17,device);rng=core.rng();b=create(cfg,data,'Cosine',17,device)
    base.equal_tree(a['model'].state_dict(),b['model'].state_dict());base.equal_tree(rng,core.rng())
    save(output,dict(status='PASS',CUDA_validated=not cpu,real_data_rows_read=0,parameters=213119,initial_model_and_RNG_matched=True,
        schedule_checks=checks(),restore=records,implementation_files=implementation(cfg)))
    if lock:lock.close()
    print('B_EARLY_SCHEDULE_WITNESS_PASS',flush=True)


def prepare(cfg):
    out=Path(cfg['output']);assert not (out/'PREPARE_STARTED.json').exists()
    save(out/'PREPARE_STARTED.json',dict(pid=os.getpid(),unix=time.time()));old=read(cfg['source_seal'])
    files={cfg['source_seal']:sha(cfg['source_seal'])}
    for p,d in {**old['code'],**old['files']}.items():assert sha(p)==d,p;files[p]=d
    for p in cfg['protocols']:
        for k in cfg['inners']:
            for s in cfg['seeds']:
                for e in [20,30,40]:
                    d=Path(cfg['extension_parent'])/'trajectories'/p/f'inner{k}'/f'seed{s}'/f'epoch{e}'
                    r=read(d/'COMPLETE.json');assert r['serialized_raw_EMA_exact'] and r['rng_preserved']
                    files[str(d/'COMPLETE.json')]=sha(d/'COMPLETE.json')
                    for name,digest in r['files'].items():assert sha(name)==digest,name;files[name]=digest
    free=shutil.disk_usage('.').free/1024**3;assert free>=cfg['budget']['minimum_free_disk_GiB']
    save(out/'PREPARED.json',dict(status='PASS',files=files,free_disk_GiB=free,new_scientific_fits=0))


def preflight(cfg):
    out=Path(cfg['output']);assert not (out/'PREFLIGHT_STARTED.json').exists()
    save(out/'PREFLIGHT_STARTED.json',dict(pid=os.getpid(),unix=time.time()));lock=core.acquire_gpu();records=[]
    for p in cfg['protocols']:
        for k in cfg['inners']:
            data=data_for(cfg,p,k);original=extension.original_index_data(cfg,p,k,data)
            order=np.random.default_rng(np.random.SeedSequence([17,1])).permutation(data['train'])
            original_order=np.random.default_rng(np.random.SeedSequence([17,1])).permutation(original['train'])
            base.equal_tree(base.minibatch(data,order[:256]),base.minibatch(original,original_order[:256]))
            a=create(cfg,data,'Constant',17,'cuda');expected=update(cfg,a,original,original_order[:256]);expected_rng=core.rng()
            b=create(cfg,data,'Constant',17,'cuda');actual=update(cfg,b,data,order[:256])
            base.equal_tree(expected,actual);base.equal_tree(expected_rng,core.rng());equal_states(a,b)
            for s in cfg['seeds']:
                path=Path(cfg['original_experts'])/p/f'inner{k}'/f'B_seed{s}_latest.pt'
                payload=torch.load(path,map_location='cpu',weights_only=False);extension.validate_payload(cfg,data,payload,p,k,s)
                model,opt,ema=extension.bundle(cfg,data,payload)
                np.testing.assert_array_equal(base.infer(ema.model,data,data['evaluation']),np.load(Path(cfg['data_parent'])/'baseline'/p/f'inner{k}'/f'B_{s}.npy'))
                del model,opt,ema,payload
            for arm in ARMS:
                restore=restore_check(cfg,data,arm,out/'preflight'/f'{p}_{k}_{arm}','cuda')
                st=create(cfg,data,arm,17,'cuda');times=[]
                for step in range(3):
                    tick=time.monotonic();update(cfg,st,data,order[step*256:(step+1)*256]);torch.cuda.synchronize();times.append(time.monotonic()-tick)
                tick=time.monotonic();capture(cfg,st,data,out/'preflight'/f'{p}_{k}_{arm}'/'timing',5,dict(scope='Fit timing only'));capture_time=time.monotonic()-tick
                estimate=(max(times[1:])*math.ceil(len(data['train'])/256)*40+capture_time*40+45)*1.4
                assert estimate<cfg['budget']['job_seconds']
                records.append(dict(protocol=p,inner=k,arm=arm,step_seconds=times,capture_seconds=capture_time,estimated_job_seconds=estimate,restore=restore))
                del st;gc.collect();torch.cuda.empty_cache()
            del data,original,a,b;gc.collect();torch.cuda.empty_cache()
    lock.close();estimate=sum(r['estimated_job_seconds']*3 for r in records)+300;assert estimate<cfg['budget']['total_seconds']
    save(out/'PREFLIGHT.json',dict(status='PASS',records=records,original_ema20_replays=18,original_packed_initial_next_update_exact_cases=6,
        estimated_total_seconds=estimate,implementation_files=implementation(cfg),new_held_scores=0))


def seal(cfg):
    out=Path(cfg['output']);assert not (out/'SEAL.json').exists();code=implementation(cfg);evidence={}
    for n in ['RUNTIME_WITNESS.json','INDEPENDENT_RUNTIME_WITNESS.json','PREFLIGHT.json']:
        r=read(out/n);assert r['status']=='PASS' and r['implementation_files']==code,n;evidence[n]=sha(out/n)
    prepared=read(out/'PREPARED.json')
    for p,d in prepared['files'].items():assert sha(p)==d,p
    evidence['PREPARED.json']=sha(out/'PREPARED.json')
    save(out/'SEAL.json',dict(config=cfg,code=code,files=prepared['files'],evidence=evidence,unix=time.time(),new_curve_scores=0))


def check(cfg,full=False):
    out=Path(cfg['output']);r=read(out/'SEAL.json');assert r['config']==cfg and r['code']==implementation(cfg)
    for n,d in r['evidence'].items():assert sha(out/n)==d,n
    if full:
        for p,d in r['files'].items():assert sha(p)==d,p


def compare_constant(cfg,state,data,folder,p,k,seed,epoch):
    if epoch not in [20,30,40]:return
    for view in ['ema','raw']:
        old=Path(cfg['extension_parent'])/'trajectories'/p/f'inner{k}'/f'seed{seed}'/f'epoch{epoch}'/f'held_{view}.npy'
        np.testing.assert_array_equal(np.load(folder/f'held_{view}.npy'),np.load(old))
    if epoch==20:
        before=core.rng();payload=torch.load(Path(cfg['original_experts'])/p/f'inner{k}'/f'B_seed{seed}_latest.pt',map_location='cpu',weights_only=False)
        base.equal_tree(before,payload['rng'])
        other=extension.bundle(cfg,data,payload)
        extension.equal_bundles((state['model'],state['opt'],state['ema']),other);core.restore_rng(before)
    save(folder/'ORIGINAL_REPLAY.json',dict(status='PASS',epoch=epoch,predictions_exact=True,full_source20_state_exact=epoch==20))


def worker(cfg,p,k,arm,seed):
    check(cfg);lock=core.acquire_gpu();out=Path(cfg['output']);folder=out/'trajectories'/p/f'inner{k}'/arm/f'seed{seed}'
    assert not (folder/'STARTED.json').exists();started=time.monotonic();save(folder/'STARTED.json',dict(pid=os.getpid(),unix=time.time()))
    data=data_for(cfg,p,k);state=create(cfg,data,arm,seed,'cuda');history=[]
    for epoch in range(1,41):
        tick=time.monotonic();order=np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(data['train']);num=den=0.;rates=[]
        for start in range(0,len(order),256):
            loss,mass,rate=update(cfg,state,data,order[start:start+256]);num+=loss*mass;den+=mass;rates.append(rate)
        row=dict(epoch=epoch,train_ce=num/den,lr_first=rates[0],lr_last=rates[-1],global_step=state['step'],seconds=time.monotonic()-tick)
        assert np.isfinite(row['train_ce']);history.append(row);save(folder/'history.json',history)
        d=folder/f'epoch{epoch}';capture(cfg,state,data,d,epoch,dict(protocol=p,inner=k,arm=arm,seed=seed))
        if arm=='Constant':compare_constant(cfg,state,data,d,p,k,seed,epoch)
        checkpoint(folder/'latest.pt',state,epoch=epoch,protocol=p,inner=k,seed=seed)
        save(folder/'PROGRESS.json',dict(protocol=p,inner=k,arm=arm,seed=seed,epoch=epoch,elapsed_seconds=time.monotonic()-started,updated_unix=time.time()))
        print(row,flush=True)
    check(cfg);save(folder/'COMPLETE.json',dict(status='COMPLETE',protocol=p,inner=k,arm=arm,seed=seed,epoch=40,
        epoch_receipts={str(e):sha(folder/f'epoch{e}'/'COMPLETE.json') for e in range(1,41)},
        files={str(folder/n):sha(folder/n) for n in ['latest.pt','history.json']},constant_old_trajectory_exact=arm=='Constant',
        elapsed_seconds=time.monotonic()-started,held_scores_computed=0))
    lock.close()


def queue(cfg):
    import signal
    out=Path(cfg['output']);check(cfg,True);assert not (out/'LAUNCH.json').exists();started=time.monotonic();active=None
    jobs=[dict(protocol=p,inner=k,arm=a,seed=s,status='pending') for s in cfg['seeds'] for k in cfg['inners'] for p in cfg['protocols'] for a in ARMS]
    save(out/'LAUNCH.json',dict(pid=os.getpid(),started_unix=time.time(),trajectories=36,epochs=1440,budget=cfg['budget']))
    def snapshot(phase):save(out/'PUBLIC_PROGRESS.json',dict(phase=phase,pid=os.getpid(),updated_unix=time.time(),elapsed_seconds=time.monotonic()-started,completed=sum(j['status']=='completed' for j in jobs),total=36,jobs=jobs))
    def stop():
        if active is not None and active.poll() is None:
            os.killpg(active.pid,signal.SIGTERM)
            try:active.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
    try:
        for j in jobs:
            if time.monotonic()-started>cfg['budget']['total_seconds']:raise RuntimeError('Cumulative budget exhausted')
            p,k,a,s=(j[n] for n in ['protocol','inner','arm','seed']);log=out/'logs'/f'{p}_{k}_{a}_{s}.log';log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('wb') as handle:
                active=subprocess.Popen([sys.executable,'-s','-u','-m',MODULE,'--mode','worker','--protocol',p,'--inner',str(k),'--arm',a,'--seed',str(s)],stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                j.update(status='running',pid=active.pid,started_unix=time.time());tick=time.monotonic()
                while active.poll() is None:
                    snapshot('DENSE_EARLY_CURVES')
                    if time.monotonic()-tick>cfg['budget']['job_seconds'] or time.monotonic()-started>cfg['budget']['total_seconds']:stop();raise RuntimeError('Scientific budget exhausted')
                    time.sleep(3)
            marker=out/'trajectories'/p/f'inner{k}'/a/f'seed{s}'/'COMPLETE.json'
            if active.returncode!=0 or not marker.exists():j['status']='failed';raise RuntimeError('Dense curve fit/replay failed; no automatic retry')
            assert read(marker)['status']=='COMPLETE';j.update(status='completed',finished_unix=time.time());snapshot('DENSE_EARLY_CURVES')
        check(cfg,True);snapshot('FREEZING_AND_SCORING')
        from scripts.report_b_early_schedule_20261024 import report
        save(out/'COMPLETE.json',report(cfg));snapshot('COMPLETE_STOP_FOR_REVIEW')
    except BaseException:
        stop()
        for j in jobs:
            if j['status'] in ['pending','running']:j['status']='blocked'
        snapshot('FAILED_STOP_FOR_INSPECTION');save(out/'QUEUE_FAILURE.json',dict(status='FAILED',traceback=traceback.format_exc(),unix=time.time()));raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['cpu_witness','witness','prepare','preflight','seal','launch','queue','worker']);p.add_argument('--witness-output')
    p.add_argument('--protocol',choices=['TS','OS']);p.add_argument('--inner',type=int,choices=[0,1,2]);p.add_argument('--arm',choices=ARMS);p.add_argument('--seed',type=int,choices=[17,23,29])
    a=p.parse_args();cfg=configure(a.mode in ['witness','preflight','worker']);out=Path(cfg['output'])
    if a.mode in ['cpu_witness','witness']:witness(cfg,a.witness_output or out/'RUNTIME_WITNESS.json',a.mode=='cpu_witness')
    elif a.mode=='prepare':prepare(cfg)
    elif a.mode=='preflight':preflight(cfg)
    elif a.mode=='seal':seal(cfg)
    elif a.mode=='worker':
        assert a.protocol in cfg['protocols'] and a.inner in cfg['inners'] and a.arm in ARMS and a.seed in cfg['seeds'];worker(cfg,a.protocol,a.inner,a.arm,a.seed)
    elif a.mode=='queue':queue(cfg)
    else:
        check(cfg,True);assert not any((out/n).exists() for n in ['DISPATCH.json','LAUNCH.json'])
        with (out/'queue.log').open('wb') as log:child=subprocess.Popen([sys.executable,'-s','-u','-m',MODULE,'--mode','queue'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(out/'DISPATCH.json',dict(pid=child.pid,unix=time.time()));print(child.pid)


if __name__=='__main__':main()
