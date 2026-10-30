"""Bounded neural DS repair: moderate L2, validation plateau, actual best checkpoint."""
from pathlib import Path
import argparse
import copy
import gc
import json
import os
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from scripts import run_comparator_convergence_20261027_v2 as parent
from scripts.comparator_repair_policy_20261028 import selected, stopping, compare, allow_confirmation
from scripts.comparator_convergence_criteria_20261027_v2 import assess as optimization
from scripts.comparator_generalization_gate_20261027 import assess as generalization
from scripts.sensor_fusion_stability_20261027 import metric
from scripts.resume_comparator_queue_20261028 import resource_conflict

CONFIG = Path('configs/sensor_comparator_repair_20261028_v7.json')
MODULE = 'scripts.run_comparator_repair_20261028_v7'
read, save, sha = parent.read, parent.save, parent.sha
base, core = parent.base, parent.core


def setup(cuda=False):
    parent.setup(cuda)
    cfg = read(CONFIG)
    assert cfg['weight_decay'] == .03 and cfg['seeds'] == [17, 23, 29]
    assert all(not cfg[k] for k in ['Test_allowed', 'outer_VAL_allowed', 'formal_refit_allowed', 'new_paid_resources', 'automatic_fivefold_expansion'])
    return cfg


def code(cfg):
    result = parent.code(read(parent.CONFIG))
    for file in [CONFIG, Path(__file__), cfg['ledger'], 'scripts/comparator_repair_policy_20261028.py',
                 'tests/test_comparator_repair_policy_20261028.py', 'scripts/comparator_generalization_gate_20261027.py',
                 'configs/sensor_comparator_joint_acceptance_20261027_v1.json', 'scripts/resume_comparator_queue_20261028.py']:
        result[str(file)] = sha(file)
    return result


def witness(cfg, cpu):
    lock = None if cpu else core.acquire_gpu()
    device = 'cpu' if cpu else 'cuda'
    out = Path(cfg['output'])
    folder = out / ('cpu_work' if cpu else 'cuda_work')
    assert not folder.exists(); folder.mkdir(parents=True)
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_comparator_repair_policy_20261028.py'], check=True)
    records = []
    for model in cfg['models']:
        batch = parent.old.synthetic_batch(device, 21)
        data = dict(prior=np.ones(7)/7, tensors=dict(**batch, labels=torch.arange(21, device=device)%7),
                    weights=torch.arange(1, 8, device=device).float(), train=np.arange(21))
        original = parent.new(dict(cfg, weight_decay=.01), model, data['prior'], 17, device)
        state = parent.new(cfg, model, data['prior'], 17, device)
        base.equal_tree(original['model'].state_dict(), state['model'].state_dict())
        assert all(g['weight_decay'] in [0., .03] for g in state['opt'].param_groups)
        assert any(g['weight_decay'] == .03 for g in state['opt'].param_groups)
        del original
        parent.update(state, data, data['train']); state['scheduler'].step(.5)
        parent.checkpoint(folder / (model+'.pt'), state, epoch=1)
        payload = torch.load(folder / (model+'.pt'), map_location='cpu', weights_only=False)
        expected = parent.update(state, data, data['train']); rng = core.rng()
        restored = parent.restore(cfg, model, data['prior'], 17, payload, device)
        actual = parent.update(restored, data, data['train'])
        base.equal_tree(expected, actual); base.equal_tree(rng, core.rng())
        for key in ['model', 'opt', 'scheduler']:
            base.equal_tree(state[key].state_dict(), restored[key].state_dict())
        base.equal_tree(state['ema'].model.state_dict(), restored['ema'].model.state_dict())
        assert state['ema'].updates == restored['ema'].updates and state['step'] == restored['step']
        records.append(dict(model=model, initialization_matches_original=True, complete_next_update_exact=True))
        del state, restored, payload, batch, data; gc.collect()
        if not cpu: torch.cuda.empty_cache()
    save(out / ('CPU_WITNESS.json' if cpu else 'CUDA_WITNESS.json'),
         dict(status='PASS', real_data_rows_read=0, CUDA_validated=not cpu, records=records, code=code(cfg)))
    if lock: lock.close()
    print('DS_REPAIR_CPU_PASS' if cpu else 'DS_REPAIR_CUDA_PASS', flush=True)


def baseline_path(cfg, model, protocol, inner, seed):
    return Path(cfg['reference_output'])/'trajectories'/protocol/f'inner{inner}'/model/f'seed{seed}'/'history.json'


def prepare(cfg):
    out = Path(cfg['output']); assert not (out/'SEAL.json').exists()
    c = code(cfg)
    for name in ['CPU_WITNESS.json', 'CUDA_WITNESS.json']:
        r = read(out/name); assert r['status']=='PASS' and r['code']==c
    capacity = Path(read(parent.CONFIG)['output'])/'PREFLIGHT.json'
    profile = read(capacity); assert profile['status']=='PASS'
    for item in profile['records']:
        assert item['conservative_maximum_worker_seconds'] < cfg['budget']['worker_seconds']
    files = {str(capacity): sha(capacity)}
    data_seal = read(Path(cfg['data_parent'])/'SEAL.json')
    for protocol in cfg['protocols']:
        for inner in cfg['inners']:
            d = Path(cfg['data_parent'])/'data'/protocol/f'inner{inner}'
            receipt = d/'RECEIPT.json'
            assert sha(receipt)==data_seal['files'][str(receipt)]
            files[str(receipt)] = sha(receipt)
            for name, expected in read(receipt)['files'].items():
                assert sha(d/name)==expected; files[str(d/name)] = expected
            for model in cfg['models']:
                for seed in cfg['seeds']:
                    hp = baseline_path(cfg,model,protocol,inner,seed)
                    if hp.exists(): files[str(hp)]=sha(hp)
    save(out/'SEAL.json', dict(config=cfg, code=c, files=files, original_capacity_reference=str(capacity),
        capacity_scope='Same model/batch/dtype and update; only AdamW scalar wd differs, maximum120 vs profiled200 epochs', Test_rows=0))


def check(cfg):
    seal = read(Path(cfg['output'])/'SEAL.json')
    assert seal['config']==cfg and seal['code']==code(cfg)
    return seal


def directory(cfg, model, protocol, inner, seed):
    return Path(cfg['output'])/'trajectories'/protocol/f'inner{inner}'/model/f'seed{seed}'


def worker(cfg, model, protocol, inner, seed):
    seal = check(cfg); lock = core.acquire_gpu()
    out = directory(cfg,model,protocol,inner,seed)
    assert not out.exists(); save(out/'STARTED.json', dict(pid=os.getpid(),unix=time.time()))
    started = time.monotonic()
    data = base.load_data(read(Path(cfg['data_parent'])/'SEAL.json')['config'],protocol,inner)
    target = Path(cfg['data_parent'])/'data'/protocol/f'inner{inner}'/'DEV_TARGETS.jsonl'
    assert sha(target)==seal['files'][str(target)]
    y = np.asarray([json.loads(line)['label']-1 for line in target.read_text().splitlines()])
    fy = data['tensors']['labels'][torch.as_tensor(data['train'],device='cuda')].cpu().numpy()
    weights = data['weights'].double().cpu().numpy()
    hp = baseline_path(cfg,model,protocol,inner,seed)
    baseline = None
    if hp.exists():
        assert sha(hp)==seal['files'][str(hp)]
        baseline = selected([r for r in read(hp) if r['stage']=='source'])
    state = parent.new(cfg,model,data['prior'],seed,'cuda')
    history, snaps = [], []
    best = float('inf')
    for epoch in range(1,cfg['stopping']['maximum_epochs']+1):
        tick = time.monotonic(); rate=state['opt'].param_groups[0]['lr']
        before = {n:v.detach().clone() for n,v in state['model'].named_parameters()}
        ebefore = {n:v.detach().clone() for n,v in state['ema'].model.named_parameters()}
        order = np.random.default_rng(np.random.SeedSequence([seed,epoch,0])).permutation(data['train'])
        num=den=0.
        for offset in range(0,len(order),cfg['batch_size']):
            loss,mass = parent.update(state,data,order[offset:offset+cfg['batch_size']]); num+=loss*mass; den+=mass
        raw_delta = parent.relative_delta(state['model'],before)
        ema_delta = parent.relative_delta(state['ema'].model,ebefore)
        snaps.append(parent.snapshot(state,epoch)); snaps=snaps[-3:]
        rng=core.rng(); evaluation=parent.evaluation_model(state,snaps,epoch)
        fit=base.infer(evaluation,data,data['train']); dev=base.infer(evaluation,data,data['evaluation'])
        fm,_=metric(fit,fy,weights); dm,_=metric(dev,y,weights)
        row=dict(epoch=epoch,stage='source',stage_epoch=epoch,learning_rate=rate,train_ce=num/den,
            fit_ema=fm,dev_ema=dm,raw_relative_parameter_change=raw_delta,EMA_relative_parameter_change=ema_delta,
            seconds=time.monotonic()-tick)
        history.append(row)
        if dm['weighted_CE'] < best:
            best=dm['weighted_CE']
            torch.save(dict(model=evaluation.state_dict(),epoch=epoch,model_name=model),out/'selected_best_DEV_CE.pt')
            np.save(out/'selected_DEV_EMA.npy',dev)
        state['scheduler'].step(dm['weighted_CE']); row['next_learning_rate']=state['opt'].param_groups[0]['lr']
        core.restore_rng(rng)
        save(out/'history.json',history)
        parent.checkpoint(out/'latest.pt',state,epoch=epoch,causal_ema_snapshots=snaps)
        stop=stopping(history,cfg['stopping'])
        save(out/'PROGRESS.json',dict(model=model,protocol=protocol,inner=inner,seed=seed,epoch=epoch,
            selected_epoch=stop['selected_epoch'],DEV_CE=dm['weighted_CE'],DEV_F1=dm['macro_f1_percent'],
            stopping=stop,elapsed_seconds=time.monotonic()-started))
        print(dict(model=model,protocol=protocol,inner=inner,seed=seed,epoch=epoch,DEV=dm,stop=stop),flush=True)
        del evaluation, before, ebefore
        assert time.monotonic()-started<cfg['budget']['worker_seconds']
        if stop['stop']: break
    chosen=selected(history)
    payload=torch.load(out/'selected_best_DEV_CE.pt',map_location='cpu',weights_only=False)
    assert payload['epoch']==chosen['epoch']
    state['model'].load_state_dict(payload['model'])
    np.testing.assert_array_equal(base.infer(state['model'],data,data['evaluation']),np.load(out/'selected_DEV_EMA.npy'))
    pair=compare(chosen,baseline,cfg['paired_guard']) if baseline else dict(paired_guard_pass=False,status='NO_PRIOR_MATCHED_BASELINE')
    opt=optimization(history,cfg['convergence'])
    gen=generalization(history,read('configs/sensor_comparator_joint_acceptance_20261027_v1.json')['neural'])
    check(cfg)
    save(out/'COMPLETE.json',dict(status='COMPLETE_DEVELOPMENT',model=model,protocol=protocol,inner=inner,seed=seed,
        epochs=len(history),stopping=stop,selected_checkpoint=chosen,baseline_selected_checkpoint=baseline,
        comparison=pair,legacy_terminal_optimization=opt,legacy_terminal_generalization=gen,
        legacy_terminal_joint_pass=opt['passed'] and gen['passed'],selected_checkpoint_replay_exact=True,
        independent_generalization=False,official_Test_rows_read=0,formal_adoption_allowed=False,
        elapsed_seconds=time.monotonic()-started,
        files={str(out/n):sha(out/n) for n in ['history.json','selected_best_DEV_CE.pt','selected_DEV_EMA.npy','latest.pt']}))
    lock.close()


def queue(cfg):
    check(cfg); out=Path(cfg['output']); assert not (out/'LAUNCH.json').exists()
    save(out/'LAUNCH.json',dict(pid=os.getpid(),unix=time.time(),maximum_cases=72))
    started=time.monotonic(); jobs=[]
    plan=[(m,p,0,17) for m in cfg['models'] for p in cfg['protocols']]
    expanded=set()
    while plan:
        model,protocol,inner,seed=plan.pop(0)
        folder=directory(cfg,model,protocol,inner,seed)
        job=dict(model=model,protocol=protocol,inner=inner,seed=seed,status='RUNNING'); jobs.append(job)
        for admission in range(4):
            while int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()[0])>=500:
                assert time.monotonic()-started<cfg['budget']['queue_seconds']
                save(out/'QUEUE_STATE.json',dict(status='WAITING_RESOURCE',jobs=jobs,pending=len(plan))); time.sleep(15)
            log=out/f'{model}_{protocol}_{inner}_{seed}_admission{admission}.log'
            save(out/'QUEUE_STATE.json',dict(status='RUNNING',jobs=jobs,pending=len(plan),log=str(log)))
            with log.open('xb') as stream:
                result=subprocess.run([sys.executable,'-s','-u','-m',MODULE,'--mode','worker','--model',model,
                    '--protocol',protocol,'--inner',str(inner),'--seed',str(seed)],stdout=stream,stderr=subprocess.STDOUT,
                    timeout=cfg['budget']['worker_seconds']+60)
            if result.returncode==0: break
            if admission<3 and resource_conflict(result.returncode,log,folder): continue
            raise RuntimeError('Scientific/runtime failure; preserve '+str(log))
        r=read(folder/'COMPLETE.json'); job.update(status='COMPLETE',validation_plateau=r['stopping']['validation_plateau'],
            paired_guard_pass=r['comparison']['paired_guard_pass'],selected_epoch=r['stopping']['selected_epoch'],
            selected_DEV_F1=r['selected_checkpoint']['dev_ema']['macro_f1_percent'])
        if inner==0 and seed==17 and model not in expanded:
            pilots=[read(directory(cfg,model,p,0,17)/'COMPLETE.json') for p in cfg['protocols']
                    if (directory(cfg,model,p,0,17)/'COMPLETE.json').exists()]
            if len(pilots)==2:
                expanded.add(model)
                if allow_confirmation(pilots):
                    plan += [(model,p,k,s) for p in cfg['protocols'] for k in cfg['inners'] for s in cfg['seeds'] if (k,s)!=(0,17)]
        assert time.monotonic()-started<cfg['budget']['queue_seconds']
    summary=dict(status='COMPLETE_REVIEW_REQUIRED',jobs=jobs,completed=len(jobs),maximum_cases=72,
        legacy_gates_changed=False,independent_generalization=False,formal_fivefold_complete=False)
    save(out/'COMPLETE.json',summary); save(out/'QUEUE_STATE.json',summary)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['cpu_witness','cuda_witness','prepare','worker','queue'],required=True)
    p.add_argument('--model');p.add_argument('--protocol',choices=['TS','OS']);p.add_argument('--inner',type=int);p.add_argument('--seed',type=int)
    a=p.parse_args();cfg=setup(a.mode in ['cuda_witness','worker'])
    try:
        if a.mode in ['cpu_witness','cuda_witness']:witness(cfg,a.mode=='cpu_witness')
        elif a.mode=='prepare':prepare(cfg)
        elif a.mode=='worker':
            assert a.model in cfg['models'] and a.protocol in cfg['protocols'] and a.inner in cfg['inners'] and a.seed in cfg['seeds']
            worker(cfg,a.model,a.protocol,a.inner,a.seed)
        else:queue(cfg)
    except Exception as exc:
        if a.mode=='queue':save(Path(cfg['output'])/'QUEUE_STATE.json',dict(status='FAILED_NO_BLIND_RETRY',error=repr(exc)))
        raise


if __name__=='__main__':main()
