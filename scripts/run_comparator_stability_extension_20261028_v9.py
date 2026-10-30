"""Bounded continuation of immutable V7 terminal states; frozen selected models."""
import argparse
import copy
import gc
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

CONFIG = Path('configs/sensor_comparator_stability_extension_20261028_v9.json')
MODULE = 'scripts.run_comparator_stability_extension_20261028_v9'
KEYS = ('model', 'protocol', 'inner', 'seed')


def training_components(result, cfg):
    """Extract original tolerances, excluding DEV-dependent mixed-gate clauses."""
    windows = result.get('windows', [])
    cross = result.get('cross_window_terminal_checks', {})
    checks = {}
    if len(windows) != 2 or not cross:
        return dict(passed=False, checks={}, reason='MISSING_ORIGINAL_COMPONENTS')
    for i, row in enumerate(windows):
        for name, limit in [('train_ce', cfg['max_abs_train_CE_relative_block_drift']),
                            ('fit_ema', cfg['max_abs_fit_EMA_CE_relative_block_drift'])]:
            value = row[name + '_relative_block_drift']
            checks[f'window{i}_{name}'] = math.isfinite(value) and abs(value) <= limit
        checks[f'window{i}_positive_LR'] = bool(row['positive_LR'])
        for key in ['maximum_raw_parameter_update', 'maximum_EMA_parameter_update']:
            value = row[key]
            checks[f'window{i}_{key}'] = math.isfinite(value) and value <= cfg['max_relative_parameter_update']
    for name, limit in [('train_ce', cfg['max_abs_train_CE_relative_block_drift']),
                        ('fit_ema', cfg['max_abs_fit_EMA_CE_relative_block_drift'])]:
        value = cross[name + '_cross_window_relative_drift']
        checks[name + '_cross_window'] = math.isfinite(value) and abs(value) <= limit
    return dict(passed=all(checks.values()), checks=checks,
                boundary='Original training-side empirical stability components only; not a global optimum or generalization certificate')


def extension_stop(history, source_end, stable, stopping_cfg, extension):
    """The extra budget is independent of the old maximum global epoch."""
    assert history and all(r['stage'] == 'source' for r in history)
    assert [r['epoch'] for r in history] == list(range(1, history[-1]['epoch'] + 1))
    meaningful = history[0]['dev_ema']['weighted_CE']; last = history[0]['epoch']
    for row in history[1:]:
        value = row['dev_ema']['weighted_CE']
        assert math.isfinite(value)
        if value < meaningful * (1 - stopping_cfg['relative_improvement']):
            meaningful, last = value, row['epoch']
    end = history[-1]['epoch']; extra = end - source_end
    assert 0 <= extra <= extension['maximum_extra_epochs']
    plateau = end >= stopping_cfg['minimum_epochs'] and end-last >= stopping_cfg['patience_epochs']
    accepted = extra >= extension['minimum_extra_epochs'] and stable and plateau
    exhausted = extra >= extension['maximum_extra_epochs']
    return dict(stop=accepted or exhausted, extra_epochs=extra, training_components_pass=bool(stable),
                validation_plateau=plateau, last_material_improvement_epoch=last,
                reason='TRAINING_STABLE_AND_DEV_PLATEAU' if accepted else 'BOUNDED_BUDGET_HOLD' if exhausted else 'CONTINUE')


def runtime_imports():
    global np, torch, v7, parent, base, core, read, save, sha, optimization, generalization, metric, slots
    import numpy as np
    import torch
    from scripts import run_comparator_repair_20261028_v7 as v7
    from scripts import accelerate_comparator_queue_20261028_v8b as slots
    parent, base, core = v7.parent, v7.base, v7.core
    read, save, sha = v7.read, v7.save, v7.sha
    optimization, generalization, metric = v7.optimization, v7.generalization, v7.metric


def setup(cuda=False):
    runtime_imports(); science = v7.setup(cuda); cfg = read(CONFIG)
    assert cfg['source_config'] == str(v7.CONFIG)
    assert cfg['minimum_extra_epochs'] == 20 and cfg['maximum_extra_epochs'] == 40
    assert cfg['GPU_processes'] == 3 and cfg['Torch_memory_fraction_per_process'] == .20
    assert cfg['minimum_free_GPU_MiB'] == 6144
    assert not any(cfg[k] for k in ['Test_allowed', 'outer_VAL_allowed', 'formal_adoption_allowed'])
    return science, cfg


def code(science, cfg):
    result = v7.code(science)
    for file in [CONFIG, Path(__file__), cfg['ledger'],
                 'scripts/accelerate_comparator_queue_20261028_v8b.py',
                 'scripts/run_convtran_endpoint_20261022.py',
                 'tests/test_comparator_stability_extension_20261028.py']:
        result[str(file)] = sha(file)
    return result


def folder(cfg, job):
    return Path(cfg['output'])/'trajectories'/job['protocol']/f"inner{job['inner']}"/job['model']/f"seed{job['seed']}"


def verify(files):
    for file, digest in files.items():
        assert sha(file) == digest, 'Immutable source changed: ' + file


def witness(science, cfg, cpu):
    lock = None if cpu else core.acquire_gpu()
    if cpu: assert not torch.cuda.is_available(), 'CPU witness must hide CUDA'
    device = 'cpu' if cpu else 'cuda'; out = Path(cfg['output'])
    work = out/('cpu_work' if cpu else 'cuda_work'); work.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p',
                    'test_comparator_stability_extension_20261028.py'], check=True)
    records = []
    for model in science['models']:
        batch = parent.old.synthetic_batch(device, 21)
        data = dict(prior=np.ones(7)/7, tensors=dict(**batch, labels=torch.arange(21, device=device)%7),
                    weights=torch.arange(1, 8, device=device).float(), train=np.arange(21))
        state = parent.new(science, model, data['prior'], 17, device); snaps = []
        initial = copy.deepcopy(state['model'].state_dict())
        for epoch in range(1, 4):
            parent.update(state, data, data['train']); state['scheduler'].step(.5)
            snaps.append(parent.snapshot(state, epoch))
        assert any(not torch.equal(initial[n], t) for n,t in state['model'].state_dict().items())
        parent.checkpoint(work/(model+'.pt'), state, epoch=3, causal_ema_snapshots=snaps)
        payload = torch.load(work/(model+'.pt'), map_location='cpu', weights_only=False)
        expected = parent.update(state, data, data['train']); state['scheduler'].step(.51)
        snaps.append(parent.snapshot(state, 4)); snaps = snaps[-3:]
        rng = core.rng()
        restored = parent.restore(science, model, data['prior'], 17, payload, device)
        restored_snaps = copy.deepcopy(payload['causal_ema_snapshots'])
        assert len(restored_snaps) == 3 and payload['epoch'] == 3 and restored['step'] == 3
        actual = parent.update(restored, data, data['train']); restored['scheduler'].step(.51)
        restored_snaps.append(parent.snapshot(restored, 4)); restored_snaps = restored_snaps[-3:]
        base.equal_tree(expected, actual); base.equal_tree(rng, core.rng()); base.equal_tree(snaps, restored_snaps)
        for key in ['model', 'opt', 'scheduler']:
            base.equal_tree(state[key].state_dict(), restored[key].state_dict())
        base.equal_tree(state['ema'].model.state_dict(), restored['ema'].model.state_dict())
        assert state['step'] == restored['step'] and state['ema'].updates == restored['ema'].updates
        records.append(dict(model=model, full_terminal_continuation_bitwise_exact=True,
                            causal_three_snapshots_exact=True, optimizer_not_reset=True))
        del state, restored, payload, data, batch, snaps, restored_snaps; gc.collect()
        if not cpu: torch.cuda.empty_cache()
    save(out/('CPU_WITNESS.json' if cpu else 'CUDA_WITNESS.json'),
         dict(status='PASS', code=code(science,cfg), records=records, real_data_rows_read=0, CUDA_validated=not cpu))
    if lock: lock.close()
    print('COMPARATOR_STABILITY_EXTENSION_CPU_PASS' if cpu else 'COMPARATOR_STABILITY_EXTENSION_CUDA_PASS', flush=True)


def prepare(science, cfg):
    out = Path(cfg['output']); assert not (out/'SEAL.json').exists()
    codes = code(science,cfg)
    for name in ['CPU_WITNESS.json','CUDA_WITNESS.json']:
        receipt = read(out/name); assert receipt['status']=='PASS' and receipt['code']==codes
    source_seal = v7.check(science)
    files = dict(source_seal['files'])
    capacity_path=Path('reports/sensor_comparator_convergence_20261027_v2/PREFLIGHT.json')
    capacity=read(capacity_path)
    assert capacity['status']=='PASS'
    assert {r['model'] for r in capacity['records']}==set(science['models'])
    total_mib=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True).splitlines()[0])
    assert max(r['peak_cuda_bytes'] for r in capacity['records']) < total_mib*2**20*cfg['Torch_memory_fraction_per_process']
    files[str(capacity_path)]=sha(capacity_path)
    for file in [Path(science['output'])/'SEAL.json', Path(science['data_parent'])/'SEAL.json', Path(cfg['source_review'])]:
        files[str(file)] = sha(file)
    verify(files); jobs=[]; all_cases=[]
    for job in slots.all_jobs(science):
        receipt = slots.completed(science,job)
        cp=slots.folder(science,job)/'COMPLETE.json'
        source_files=dict(receipt['files']);source_files[str(cp)]=sha(cp)
        history=read(slots.folder(science,job)/'history.json')
        replay=optimization(history,science['convergence'])
        assert replay==receipt['legacy_terminal_optimization'], str(cp)
        gate=training_components(receipt['legacy_terminal_optimization'],science['convergence'])
        entry={k:job[k] for k in KEYS}
        entry.update(source_directory=str(slots.folder(science,job)), source_end=history[-1]['epoch'],
                     source_files=source_files, training_components=gate)
        all_cases.append(entry)
        if not gate['passed']:jobs.append(entry)
    counts={m:sum(j['model']==m for j in jobs) for m in science['models']}
    assert len(jobs)==36 and counts==cfg['expected_failed_counts'], counts
    # All 72 receipts bind the cohort decision; only the failed 36 are continued.
    for case in all_cases: files.update(case['source_files'])
    verify(files)
    save(out/'SEAL.json',dict(config=cfg,science=science,code=codes,files=files,jobs=jobs,
         capacity_reference=str(capacity_path),unchanged_model_batch_dtype_and_data_shape=True,
         all_case_training_checks=all_cases,counts=counts,scope='Bounded continuation diagnostics only'))


def check(science,cfg,verify_all=False):
    seal=read(Path(cfg['output'])/'SEAL.json')
    assert seal['config']==cfg and seal['science']==science and seal['code']==code(science,cfg)
    if verify_all:verify(seal['files'])
    return seal


def worker(science,cfg,job,slot):
    seal=check(science,cfg); assert job in seal['jobs']
    source_inputs=[Path(science['output'])/'SEAL.json',Path(science['data_parent'])/'SEAL.json',
                   Path(science['data_parent'])/'data'/job['protocol']/f"inner{job['inner']}"/'RECEIPT.json']
    bound_sources={**job['source_files'],**{str(p):seal['files'][str(p)] for p in source_inputs}}
    verify(bound_sources)
    lock=slots.slot_gpu(cfg,slot); out=folder(cfg,job)
    out.mkdir(parents=True,exist_ok=False)
    save(out/'STARTED.json',dict(pid=os.getpid(),unix=time.time(),job=job,slot=slot))
    started=time.monotonic(); source=Path(job['source_directory'])
    data=base.load_data(read(Path(science['data_parent'])/'SEAL.json')['config'],job['protocol'],job['inner'])
    target=Path(science['data_parent'])/'data'/job['protocol']/f"inner{job['inner']}"/'DEV_TARGETS.jsonl'
    y=np.asarray([json.loads(line)['label']-1 for line in target.read_text().splitlines()])
    fy=data['tensors']['labels'][torch.as_tensor(data['train'],device='cuda')].cpu().numpy()
    weights=data['weights'].double().cpu().numpy()
    history=read(source/'history.json'); original_history=copy.deepcopy(history)
    payload=torch.load(source/'latest.pt',map_location='cpu',weights_only=False)
    assert payload['epoch']==job['source_end']==history[-1]['epoch']
    snaps=copy.deepcopy(payload['causal_ema_snapshots'])
    assert [x['epoch'] for x in snaps]==list(range(job['source_end']-2,job['source_end']+1))
    state=parent.restore(science,job['model'],data['prior'],job['seed'],payload,'cuda')
    del payload
    frozen=read(source/'COMPLETE.json')['selected_checkpoint']
    diagnostic_best=frozen['dev_ema']['weighted_CE']
    for epoch in range(job['source_end']+1,job['source_end']+cfg['maximum_extra_epochs']+1):
        tick=time.monotonic();rate=state['opt'].param_groups[0]['lr']
        before={n:v.detach().clone() for n,v in state['model'].named_parameters()}
        ebefore={n:v.detach().clone() for n,v in state['ema'].model.named_parameters()}
        order=np.random.default_rng(np.random.SeedSequence([job['seed'],epoch,0])).permutation(data['train'])
        num=den=0.
        for offset in range(0,len(order),science['batch_size']):
            loss,mass=parent.update(state,data,order[offset:offset+science['batch_size']]);num+=loss*mass;den+=mass
        raw=parent.relative_delta(state['model'],before);ema=parent.relative_delta(state['ema'].model,ebefore)
        snaps.append(parent.snapshot(state,epoch));snaps=snaps[-3:]
        rng=core.rng();evaluation=parent.evaluation_model(state,snaps,epoch)
        fit=base.infer(evaluation,data,data['train']);dev=base.infer(evaluation,data,data['evaluation'])
        fm,_=metric(fit,fy,weights);dm,_=metric(dev,y,weights)
        row=dict(epoch=epoch,stage='source',stage_epoch=epoch,learning_rate=rate,train_ce=num/den,
                 fit_ema=fm,dev_ema=dm,raw_relative_parameter_change=raw,EMA_relative_parameter_change=ema,
                 seconds=time.monotonic()-tick)
        history.append(row)
        if dm['weighted_CE']<diagnostic_best:
            diagnostic_best=dm['weighted_CE']
            torch.save(dict(model=evaluation.state_dict(),epoch=epoch,diagnostic_only=True),out/'diagnostic_best_DEV_CE.pt')
            np.save(out/'diagnostic_DEV_EMA.npy',dev)
        state['scheduler'].step(dm['weighted_CE']);row['next_learning_rate']=state['opt'].param_groups[0]['lr']
        core.restore_rng(rng)
        save(out/'history.json',history);parent.checkpoint(out/'latest.pt',state,epoch=epoch,causal_ema_snapshots=snaps)
        opt=optimization(history,science['convergence']);training=training_components(opt,science['convergence'])
        stop=extension_stop(history,job['source_end'],training['passed'],science['stopping'],cfg)
        save(out/'PROGRESS.json',dict(**{k:job[k] for k in KEYS},epoch=epoch,source_end=job['source_end'],
             stopping=stop,DEV_CE=dm['weighted_CE'],DEV_F1=dm['macro_f1_percent'],elapsed_seconds=time.monotonic()-started))
        print(dict(epoch=epoch,stopping=stop,DEV=dm),flush=True)
        del evaluation,before,ebefore
        assert time.monotonic()-started<cfg['worker_seconds']
        if stop['stop']:break
    assert history[:len(original_history)]==original_history
    extension_rows=history[len(original_history):]
    baseline_ce=min(r['dev_ema']['weighted_CE'] for r in original_history)
    baseline_f1=max(r['dev_ema']['macro_f1_percent'] for r in original_history)
    extra_ce=min(r['dev_ema']['weighted_CE'] for r in extension_rows)
    extra_f1=max(r['dev_ema']['macro_f1_percent'] for r in extension_rows)
    gen=generalization(history,read('configs/sensor_comparator_joint_acceptance_20261027_v1.json')['neural'])
    check(science,cfg);verify(bound_sources)
    artifacts={str(p):sha(p) for p in out.iterdir() if p.name in ['history.json','latest.pt','diagnostic_best_DEV_CE.pt','diagnostic_DEV_EMA.npy']}
    save(out/'COMPLETE.json',dict(status='COMPLETE_DIAGNOSTIC',**{k:job[k] for k in KEYS},
         source_end=job['source_end'],terminal_epoch=history[-1]['epoch'],stopping=stop,training_components=training,
         frozen_V7_selected_checkpoint=frozen,frozen_V7_selected_files={str(source/n):sha(source/n) for n in ['selected_best_DEV_CE.pt','selected_DEV_EMA.npy']},
         validation_extra=dict(CE_material_improvement=extra_ce<baseline_ce*.999,
             F1_material_improvement=extra_f1>baseline_f1+.5,best_CE_relative_change=extra_ce/baseline_ce-1,
             peak_F1_change_pp=extra_f1-baseline_f1),
         legacy_terminal_optimization=opt,legacy_terminal_generalization=gen,
         legacy_terminal_joint_pass=opt['passed'] and gen['passed'],source_history_unchanged=True,
         formal_adoption_allowed=False,official_Test_rows_read=0,independent_generalization=False,
         files=artifacts,source_files=job['source_files'],elapsed_seconds=time.monotonic()-started))
    lock.close()


def run(science,cfg):
    seal=check(science,cfg,True);out=Path(cfg['output']);shared=core.acquire_gpu()
    started=time.monotonic();jobs=copy.deepcopy(seal['jobs']);pending=[];active={};errors=[]
    for job in jobs:
        path=folder(cfg,job)/'COMPLETE.json'
        if path.exists():
            receipt=read(path);verify(receipt['files']);verify(receipt['source_files'])
            assert receipt['status']=='COMPLETE_DIAGNOSTIC';job['status']='COMPLETE'
        else:
            assert not folder(cfg,job).exists(),'Incomplete scientific case: no blind retry'
            job['status']='PENDING';pending.append(job)
    launch=out/'LAUNCH.json'
    assert not launch.exists(),'One bounded launch only; inspect any interrupted controller explicitly'
    save(launch,dict(pid=os.getpid(),unix=time.time(),cases=len(jobs)))
    while pending or active:
        if time.monotonic()-started>=cfg['queue_seconds'] and not errors:errors.append('QUEUE_BUDGET')
        for slot,a in list(active.items()):
            proc=a['process'];rc=proc.poll()
            if rc is None and (time.monotonic()-a['started']>cfg['worker_seconds']+120
                               or time.monotonic()-started>=cfg['queue_seconds']):
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait();rc=proc.returncode
            if rc is not None:
                a['stream'].close();job=a['job']
                try:
                    assert rc==0, f'Worker return code {rc}'
                    receipt=read(folder(cfg,job)/'COMPLETE.json');verify(receipt['files']);verify(receipt['source_files'])
                    assert receipt['status']=='COMPLETE_DIAGNOSTIC'
                    job['status']='COMPLETE'
                except Exception as exc:
                    job['status']='FAILED';errors.append(dict(job={k:job[k] for k in KEYS},returncode=rc,error=repr(exc)))
                del active[slot]
        if not errors:
            for slot in range(cfg['GPU_processes']):
                if slot in active or not pending:continue
                job=pending.pop(0);job['status']='RUNNING'
                log=out/('_'.join(str(job[k]) for k in KEYS)+'.log');stream=log.open('xb')
                cmd=[sys.executable,'-s','-u','-m',MODULE,'--mode','worker','--slot',str(slot)]
                for key in KEYS:cmd += ['--'+key,str(job[key])]
                try:
                    proc=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                except Exception as exc:
                    stream.close();job['status']='FAILED';errors.append(dict(error=repr(exc),job={k:job[k] for k in KEYS}));break
                active[slot]=dict(process=proc,stream=stream,job=job,started=time.monotonic())
        save(out/'QUEUE_STATE.json',dict(status='DRAINING_AFTER_FAILURE' if errors else 'RUNNING',jobs=jobs,
             completed=sum(j['status']=='COMPLETE' for j in jobs),active=len(active),pending=len(pending),errors=errors))
        if errors and not active:break
        if pending or active:time.sleep(5)
    if not errors:
        try:check(science,cfg,True)
        except Exception as exc:errors.append(dict(error='FINAL_SOURCE_VERIFICATION',details=repr(exc)))
    summary=dict(status='FAILED_NO_BLIND_RETRY' if errors else 'COMPLETE_REVIEW_REQUIRED',jobs=jobs,
                 completed=sum(j['status']=='COMPLETE' for j in jobs),errors=errors,formal_adoption_allowed=False)
    save(out/'QUEUE_STATE.json',summary)
    if not errors:save(out/'COMPLETE.json',summary)
    shared.close()
    if errors:raise RuntimeError(str(errors))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',required=True,choices=['cpu_witness','cuda_witness','prepare','worker','run'])
    parser.add_argument('--model');parser.add_argument('--protocol');parser.add_argument('--inner',type=int)
    parser.add_argument('--seed',type=int);parser.add_argument('--slot',type=int)
    args=parser.parse_args();science,cfg=setup(args.mode in ['cuda_witness','worker'])
    if args.mode.endswith('witness'):witness(science,cfg,args.mode=='cpu_witness')
    elif args.mode=='prepare':prepare(science,cfg)
    elif args.mode=='run':run(science,cfg)
    else:
        seal=check(science,cfg)
        jobs=[j for j in seal['jobs'] if all(j[k]==getattr(args,k) for k in KEYS)]
        assert len(jobs)==1;worker(science,cfg,jobs[0],args.slot)


if __name__=='__main__':main()
