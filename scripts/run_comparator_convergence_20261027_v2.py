"""Cloud-only convergence development for the four retained neural comparators."""
from pathlib import Path
import argparse, copy, gc, json, math, os, shutil, subprocess, sys, time, traceback
import numpy as np
import torch
from scripts import run_comparators_fivefold_20261025 as old
from scripts.comparator_convergence_criteria_20261027_v2 import assess
from scripts.sensor_state_average_20261027 import apply_window
from scripts.sensor_fusion_stability_20261027 import metric

base, core = old.base, old.core
read, save, sha = old.read, old.save, old.sha
CONFIG=Path('configs/sensor_comparator_convergence_20261027_v2.json')
MODULE='scripts.run_comparator_convergence_20261027_v2'


def setup(cuda=False):
    assert Path.cwd()==core.ROOT, 'Real workloads remain cloud-only'
    assert (torch.__version__,np.__version__)==('2.4.1+cu121','1.24.4')
    if cuda: core.configure()
    else:
        torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    cfg=read(CONFIG)
    assert all(not cfg[k] for k in ['Test_allowed','outer_VAL_allowed','formal_refit_allowed','new_paid_resources'])
    return cfg


def code(cfg):
    files=old.implementation(read(old.CONFIG))
    for p in [CONFIG,'scripts/run_comparator_convergence_20261027_v2.py',
              'scripts/comparator_convergence_criteria_20261027_v2.py',
              'scripts/comparator_convergence_criteria_20261027.py',
              'scripts/sensor_state_average_20261027.py','scripts/sensor_fusion_stability_20261027.py',
              'tests/test_comparator_convergence_20261027.py',
              'tests/test_comparator_convergence_v2_20261027.py',
              '.aris/compute/bitahub-root-comparator-convergence-20261027-v2.md']:
        files[str(p)]=sha(p)
    return files


def model(cfg, arm, prior, device):
    m=old.model_for(cfg,arm,prior,device)
    for module in m.modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm): module.momentum=cfg['BN_momentum']
    return m


def new(cfg, arm, prior, seed, device, rate=None):
    core.seed_all(seed); m=model(cfg,arm,prior,device)
    opt=base.optimizer_for(m,dict(read(cfg['model_config']),learning_rate=rate or cfg['learning_rate'],weight_decay=cfg['weight_decay']))
    ema=base.ParameterEMA(m,cfg['EMA_decay'],cfg['EMA_warmup_steps'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,**cfg['scheduler'])
    return dict(model=m,opt=opt,ema=ema,scheduler=scheduler,step=0)


def snapshot(s,epoch):
    return dict(epoch=epoch,model={k:v.detach().cpu().clone() for k,v in s['ema'].model.state_dict().items()})


def evaluation_model(s,snaps,epoch):
    m=copy.deepcopy(s['ema'].model).eval().requires_grad_(False)
    apply_window(m,snaps,epoch)
    return m


def checkpoint(path,s,**extra):
    base.checkpoint(path,s['model'],s['opt'],s['ema'],scheduler=s['scheduler'].state_dict(),global_step=s['step'],**extra)


def restore(cfg,arm,prior,seed,payload,device):
    s=new(cfg,arm,prior,seed,device)
    s['model'].load_state_dict(payload['model']);s['ema'].model.load_state_dict(payload['ema'])
    s['opt'].load_state_dict(copy.deepcopy(payload['optimizer']));s['scheduler'].load_state_dict(copy.deepcopy(payload['scheduler']))
    s['ema'].updates=payload['ema_updates'];s['step']=payload['global_step'];core.restore_rng(payload['rng'])
    return s


def update(s,data,ids):
    values=base.update(s['model'],s['opt'],s['ema'],data,ids);s['step']+=1
    return values


def witness(cfg,cpu=False):
    lock=None if cpu else core.acquire_gpu(); device='cpu' if cpu else 'cuda'; records=[]
    folder=Path(cfg['output'])/('cpu_witness_final_work' if cpu else 'cuda_witness_work')
    assert not folder.exists(); folder.mkdir(parents=True)
    for arm in cfg['models']:
        batch=old.synthetic_batch(device,21)
        data=dict(tensors=dict(**batch,labels=torch.arange(21,device=device)%7),prior=np.ones(7)/7,
                  weights=torch.ones(7,device=device),train=np.arange(21))
        s=new(cfg,arm,data['prior'],17,device);update(s,data,data['train'])
        rates=[]
        for v in [1.0]*28:
            s['scheduler'].step(v);rates.append(s['opt'].param_groups[0]['lr'])
        assert min(rates)>=cfg['scheduler']['min_lr'] and rates[-1]<rates[0]
        checkpoint(folder/(arm+'.pt'),s,epoch=1)
        p=torch.load(folder/(arm+'.pt'),map_location='cpu',weights_only=False)
        raw=copy.deepcopy(s['ema'].model.state_dict());rng=core.rng()
        m=evaluation_model(s,[snapshot(s,i) for i in [9,10,11]],11)
        expected_prob=base.infer(m,data,data['train'])
        base.equal_tree(raw,s['ema'].model.state_dict());base.equal_tree(rng,core.rng())
        m.load_state_dict(copy.deepcopy(m.state_dict()))
        np.testing.assert_array_equal(expected_prob,base.infer(m,data,data['train']))
        expected=update(s,data,data['train']);expected_rng=core.rng()
        r=restore(cfg,arm,data['prior'],17,p,device);actual=update(r,data,data['train'])
        base.equal_tree(expected,actual);base.equal_tree(expected_rng,core.rng())
        for a,b in [(s['model'].state_dict(),r['model'].state_dict()),(s['opt'].state_dict(),r['opt'].state_dict()),
                    (s['ema'].model.state_dict(),r['ema'].model.state_dict()),(s['scheduler'].state_dict(),r['scheduler'].state_dict())]:base.equal_tree(a,b)
        records.append(dict(model=arm,parameters=sum(p.numel() for p in m.parameters()),
                            resume_model_Adam_EMA_scheduler_RNG_exact=True,averaging_preserves_live_state_RNG=True,positive_LR_floor=True))
        del s,r,m,batch,data;gc.collect()
        if not cpu:torch.cuda.empty_cache()
    save(Path(cfg['output'])/('CPU_WITNESS_FINAL.json' if cpu else 'CUDA_WITNESS.json'),
         dict(status='PASS',real_data_rows_read=0,CUDA_validated=not cpu,models=records,code=code(cfg)))
    if lock:lock.close()
    print('COMPARATOR_CONVERGENCE_CPU_PASS' if cpu else 'COMPARATOR_CONVERGENCE_CUDA_PASS',flush=True)


def prepare(cfg):
    out=Path(cfg['output']);assert not (out/'SEAL.json').exists()
    expected=code(cfg)
    for name in ['CPU_WITNESS_FINAL.json','CUDA_WITNESS.json','PREFLIGHT.json']:
        r=read(out/name);assert r['status']=='PASS' and r['code']==expected
    assert shutil.disk_usage('.').free/2**30>=cfg['budget']['minimum_free_disk_GiB']
    seal=read(Path(cfg['data_parent'])/'SEAL.json');files={}
    for protocol in cfg['protocols']:
        for inner in cfg['inners']:
            d=Path(cfg['data_parent'])/'data'/protocol/f'inner{inner}'
            receipt=d/'RECEIPT.json';assert sha(receipt)==seal['files'][str(receipt)]
            files[str(receipt)]=sha(receipt)
            for name,h in read(receipt)['files'].items():
                assert sha(d/name)==h;files[str(d/name)]=h
    save(out/'SEAL.json',dict(config=cfg,code=expected,files=files,created_unix=time.time(),new_Test_scores=0))


def preflight(cfg):
    out=Path(cfg['output']);assert not (out/'PREFLIGHT.json').exists()
    lock=core.acquire_gpu();data=base.load_data(read(Path(cfg['data_parent'])/'SEAL.json')['config'],'TS',0);records=[]
    for arm in cfg['models']:
        s=new(cfg,arm,data['prior'],17,'cuda');times=[]
        torch.cuda.reset_peak_memory_stats()
        for i in range(3):
            tick=time.monotonic();update(s,data,data['train'][i*256:(i+1)*256]);torch.cuda.synchronize();times.append(time.monotonic()-tick)
        tick=time.monotonic()
        for ids in [data['train'],data['evaluation']]:base.infer(s['ema'].model,data,ids)
        torch.cuda.synchronize();infer_seconds=time.monotonic()-tick
        epoch_seconds=max(times[1:])*math.ceil(len(data['train'])/256)+infer_seconds*2+0.5
        maximum=epoch_seconds*(cfg['source_max_epochs']+cfg['recovery_max_epochs'])*1.5+60
        assert maximum<cfg['budget']['worker_seconds'],(arm,maximum)
        records.append(dict(model=arm,step_seconds=times,full_fit_DEV_inference_seconds=infer_seconds,
                            conservative_maximum_worker_seconds=maximum,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
        del s;gc.collect();torch.cuda.empty_cache()
    save(out/'PREFLIGHT.json',dict(status='PASS',records=records,code=code(cfg),official_Test_rows_read=0,
                                  maximum72_worker_hours=sum(r['conservative_maximum_worker_seconds']*18 for r in records)/3600))
    lock.close();print('COMPARATOR_CONVERGENCE_PREFLIGHT_PASS',flush=True)


def check(cfg):
    seal=read(Path(cfg['output'])/'SEAL.json');assert seal['config']==cfg and seal['code']==code(cfg)
    return seal


def relative_delta(net,before):
    numerator=sum((p.detach().double()-before[n].double()).square().sum() for n,p in net.named_parameters())
    denominator=sum(p.detach().double().square().sum() for p in net.parameters())
    return float((numerator/denominator).sqrt())


def case_dir(cfg,arm,p,k,seed):return Path(cfg['output'])/'trajectories'/p/f'inner{k}'/arm/f'seed{seed}'


def worker(cfg,arm,p,k,seed):
    seal=check(cfg);lock=core.acquire_gpu();started=time.monotonic();folder=case_dir(cfg,arm,p,k,seed)
    prior=Path(cfg['prior_output'])/'trajectories'/p/f'inner{k}'/arm/f'seed{seed}'
    if (prior/'COMPLETE.json').exists():
        previous=read(prior/'COMPLETE.json'); oldhistory=read(prior/'history.json')
        stricter=assess(oldhistory,cfg['convergence'])
        if stricter['passed']:
            assert previous['empirical_convergence']
            for path,digest in previous['files'].items():assert sha(path)==digest
            folder.mkdir(parents=True,exist_ok=False)
            for name in ['history.json','terminal_ema.pt','terminal_DEV_EMA.npy','best_DEV_CE_ema.pt']:
                shutil.copyfile(prior/name,folder/name)
            save(folder/'CONVERGENCE.json',stricter)
            save(folder/'COMPLETE.json',dict(status='COMPLETE',model=arm,protocol=p,inner=k,seed=seed,
                 empirical_convergence=True,epochs=len(oldhistory),new_training_epochs=0,reused_source=str(prior),
                 reused_source_sha256=sha(prior/'COMPLETE.json'),stricter_criteria_replayed=True,
                 official_Test_rows_read=0,outer_VAL_rows_read=0,
                 files={str(folder/n):sha(folder/n) for n in ['history.json','terminal_ema.pt','terminal_DEV_EMA.npy','best_DEV_CE_ema.pt','CONVERGENCE.json']}))
            lock.close();return
    assert not (folder/'STARTED.json').exists();save(folder/'STARTED.json',dict(pid=os.getpid(),unix=started,wall_unix=time.time()))
    data=base.load_data(read(Path(cfg['data_parent'])/'SEAL.json')['config'],p,k);targets=Path(cfg['data_parent'])/'data'/p/f'inner{k}'/'DEV_TARGETS.jsonl'
    assert sha(targets)==seal['files'][str(targets)]
    y=np.array([json.loads(line)['label']-1 for line in targets.read_text().splitlines()]);assert len(y)==len(data['evaluation'])
    fy=data['tensors']['labels'][torch.as_tensor(data['train'],device='cuda')].cpu().numpy();weights=data['weights'].double().cpu().numpy()
    history=[];s=new(cfg,arm,data['prior'],seed,'cuda');best_ce=float('inf');assessment={'passed':False}
    for stage in ['source','recovery']:
        if stage=='recovery':
            payload=torch.load(folder/'best_DEV_CE_ema.pt',map_location='cpu',weights_only=False)
            save(folder/'RECOVERY.json',dict(source_epoch=payload['epoch'],source_DEV_CE=best_ce,reason='source failed original convergence criteria',
                 Adam_and_EMA_clock_reset=True,source_history_and_minimum_preserved=True,learning_rate=cfg['scheduler']['min_lr']))
            s=new(cfg,arm,data['prior'],seed,'cuda',cfg['scheduler']['min_lr'])
            s['model'].load_state_dict(payload['model']);s['ema'].model.load_state_dict(payload['model'])
        snaps=[]
        limit=cfg['source_max_epochs'] if stage=='source' else cfg['recovery_max_epochs']
        for epoch in range(1,limit+1):
            tick=time.monotonic();rate=s['opt'].param_groups[0]['lr']
            before={n:v.detach().clone() for n,v in s['model'].named_parameters()};ebefore={n:v.detach().clone() for n,v in s['ema'].model.named_parameters()}
            order=np.random.default_rng(np.random.SeedSequence([seed,epoch,0 if stage=='source' else 1])).permutation(data['train']);num=den=0.
            for offset in range(0,len(order),cfg['batch_size']):
                loss,mass=update(s,data,order[offset:offset+cfg['batch_size']]);num+=loss*mass;den+=mass
            raw_delta=relative_delta(s['model'],before);ema_delta=relative_delta(s['ema'].model,ebefore)
            snaps.append(snapshot(s,epoch));snaps=snaps[-3:];rng=core.rng();evaluation=evaluation_model(s,snaps,epoch)
            fit=base.infer(evaluation,data,data['train']);dev=base.infer(evaluation,data,data['evaluation']);rawdev=base.infer(s['model'],data,data['evaluation'])
            fm,_=metric(fit,fy,weights);dm,_=metric(dev,y,weights);rm,_=metric(rawdev,y,weights)
            ep=folder/f'epoch{len(history)+1}';ep.mkdir()
            torch.save(dict(model=evaluation.state_dict(),epoch=len(history)+1,stage=stage,local_epoch=epoch,arm=arm),ep/'ema.pt')
            evaluation.load_state_dict(torch.load(ep/'ema.pt',map_location='cpu',weights_only=False)['model'])
            np.testing.assert_array_equal(dev,base.infer(evaluation,data,data['evaluation']))
            np.save(ep/'DEV_EMA.npy',dev);core.restore_rng(rng)
            row=dict(epoch=len(history)+1,stage=stage,stage_epoch=epoch,learning_rate=rate,train_ce=num/den,
                     fit_ema=fm,dev_ema=dm,dev_raw=rm,raw_relative_parameter_change=raw_delta,EMA_relative_parameter_change=ema_delta,
                     seconds=time.monotonic()-tick,global_step=s['step'])
            history.append(row)
            if dm['weighted_CE']<best_ce:
                best_ce=dm['weighted_CE'];shutil.copyfile(ep/'ema.pt',folder/'best_DEV_CE_ema.pt')
            if stage=='source':s['scheduler'].step(dm['weighted_CE'])
            row['next_learning_rate']=s['opt'].param_groups[0]['lr']
            save(folder/'history.json',history);checkpoint(folder/'latest.pt',s,epoch=len(history),stage=stage,stage_epoch=epoch,causal_ema_snapshots=snaps)
            assessment=assess(history,cfg['convergence'])
            save(folder/'PROGRESS.json',dict(model=arm,protocol=p,inner=k,seed=seed,epoch=len(history),stage=stage,stage_epoch=epoch,
                                           DEV_CE=dm['weighted_CE'],DEV_F1=dm['macro_f1_percent'],convergence=assessment,elapsed_seconds=time.monotonic()-started))
            print(dict(model=arm,protocol=p,inner=k,seed=seed,epoch=len(history),stage=stage,LR=rate,
                       train_CE=num/den,DEV_CE=dm['weighted_CE'],DEV_F1=dm['macro_f1_percent'],converged=assessment['passed']),flush=True)
            del evaluation,before,ebefore
            assert time.monotonic()-started<cfg['budget']['worker_seconds'],'Registered worker budget exceeded'
            if assessment['passed']:break
        if assessment['passed']:break
    terminal=folder/f'epoch{len(history)}'
    shutil.copyfile(terminal/'ema.pt',folder/'terminal_ema.pt');shutil.copyfile(terminal/'DEV_EMA.npy',folder/'terminal_DEV_EMA.npy')
    save(folder/'CONVERGENCE.json',assessment)
    check(cfg)
    save(folder/'COMPLETE.json',dict(status='COMPLETE',model=arm,protocol=p,inner=k,seed=seed,
          empirical_convergence=assessment['passed'],epochs=len(history),terminal_is_best_checkpoint_substitution=False,
          official_Test_rows_read=0,outer_VAL_rows_read=0,elapsed_seconds=time.monotonic()-started,
          files={str(folder/n):sha(folder/n) for n in ['history.json','latest.pt','terminal_ema.pt','terminal_DEV_EMA.npy','best_DEV_CE_ema.pt','CONVERGENCE.json']}))
    lock.close()


def queue(cfg):
    check(cfg);out=Path(cfg['output']);assert not (out/'LAUNCH.json').exists()
    save(out/'LAUNCH.json',dict(pid=os.getpid(),unix=time.time(),models=cfg['models'],maximum_fits=72))
    jobs=[];started=time.monotonic()
    # Two pilots per architecture precede any expansion for that architecture.
    plan=[(a,p,0,17) for a in cfg['models'] for p in cfg['protocols']]
    expanded=set()
    while plan:
        arm,p,k,seed=plan.pop(0);folder=case_dir(cfg,arm,p,k,seed);log=out/'logs'/f'{arm}_{p}_{k}_{seed}.log';log.parent.mkdir(exist_ok=True)
        job=dict(model=arm,protocol=p,inner=k,seed=seed,status='RUNNING',log=str(log));jobs.append(job)
        save(out/'QUEUE_STATE.json',dict(status='RUNNING',jobs=jobs,pending=len(plan),elapsed_seconds=time.monotonic()-started))
        while int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()[0])>=500:
            if time.monotonic()-started>=cfg['budget']['queue_seconds']:
                job.update(status='FAILED_RESOURCE_WAIT_TIMEOUT')
                save(out/'QUEUE_STATE.json',dict(status='FAILED_RESOURCE_WAIT_TIMEOUT',jobs=jobs,pending=len(plan)))
                raise RuntimeError('Registered queue budget exhausted while waiting for GPU')
            time.sleep(10)
        remaining=cfg['budget']['queue_seconds']-(time.monotonic()-started)
        if remaining<=0:
            job.update(status='FAILED_QUEUE_BUDGET')
            save(out/'QUEUE_STATE.json',dict(status='FAILED_QUEUE_BUDGET',jobs=jobs,pending=len(plan)))
            raise RuntimeError('Registered queue budget exhausted before worker launch')
        with log.open('wb') as stream:
            try:
                result=subprocess.run([sys.executable,'-s','-u','-m',MODULE,'--mode','worker','--model',arm,'--protocol',p,'--inner',str(k),'--seed',str(seed)],stdout=stream,stderr=subprocess.STDOUT,timeout=min(cfg['budget']['worker_seconds']+60,remaining))
            except subprocess.TimeoutExpired:
                job.update(status='FAILED_TIMEOUT_NO_RETRY')
                save(out/'QUEUE_STATE.json',dict(status='FAILED_TIMEOUT_NO_RETRY',jobs=jobs,pending=len(plan)))
                raise
        if result.returncode!=0:
            job.update(status='FAILED_RUNTIME',exit_code=result.returncode)
            save(out/'QUEUE_STATE.json',dict(status='FAILED_RUNTIME_NO_RETRY',jobs=jobs,pending=len(plan)));raise RuntimeError(f'Inspect {log}; no blind retry')
        done=read(folder/'COMPLETE.json');job.update(status='CONVERGED' if done['empirical_convergence'] else 'NOT_CONVERGED',epochs=done['epochs'])
        if k==0 and seed==17 and arm not in expanded:
            pilots=[j for j in jobs if j['model']==arm and j['inner']==0 and j['seed']==17]
            if len(pilots)==2:
                expanded.add(arm)
                if all(j['status']=='CONVERGED' for j in pilots):
                    plan += [(arm,pp,kk,ss) for pp in cfg['protocols'] for kk in cfg['inners'] for ss in cfg['seeds'] if (kk,ss)!=(0,17)]
        if time.monotonic()-started>=cfg['budget']['queue_seconds']:
            save(out/'QUEUE_STATE.json',dict(status='FAILED_QUEUE_BUDGET',jobs=jobs,pending=len(plan)))
            raise RuntimeError('Registered queue budget exhausted after worker completion')
        save(out/'QUEUE_STATE.json',dict(status='RUNNING',jobs=jobs,pending=len(plan),elapsed_seconds=time.monotonic()-started))
    passed=len(jobs)==72 and all(j['status']=='CONVERGED' for j in jobs)
    report=dict(status='ALL72_DEVELOPMENT_CONVERGED' if passed else 'DEVELOPMENT_NOT_FULLY_CONVERGED',jobs=jobs,
                converged=sum(j['status']=='CONVERGED' for j in jobs),total_completed=len(jobs),required=72,
                official_Test_rows_read=0,formal_fivefold_replaced=False,manuscript_changed=False)
    save(out/'COMPLETE.json',report);save(out/'QUEUE_STATE.json',report)
    print(report['status'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['cpu_witness','cuda_witness','preflight','prepare','worker','queue'])
    p.add_argument('--model');p.add_argument('--protocol',choices=['TS','OS']);p.add_argument('--inner',type=int,choices=[0,1,2]);p.add_argument('--seed',type=int,choices=[17,23,29]);a=p.parse_args()
    cfg=setup(a.mode in ['cuda_witness','preflight','worker'])
    if a.mode in ['cpu_witness','cuda_witness']:witness(cfg,a.mode=='cpu_witness')
    elif a.mode=='prepare':prepare(cfg)
    elif a.mode=='preflight':preflight(cfg)
    elif a.mode=='worker':
        assert a.model in cfg['models'] and a.protocol in cfg['protocols'] and a.inner in cfg['inners'] and a.seed in cfg['seeds']
        worker(cfg,a.model,a.protocol,a.inner,a.seed)
    else:queue(cfg)


if __name__=='__main__':main()
