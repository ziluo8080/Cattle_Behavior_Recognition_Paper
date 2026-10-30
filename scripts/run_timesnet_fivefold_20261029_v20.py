"""Fresh TimesNet calibration/refit on five folds, then frozen prediction/scoring."""
import argparse
import copy
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from scripts import timesnet_fivefold_common_20261029_v20 as c

MODULE='scripts.run_timesnet_fivefold_20261029_v20'
KEYS=('stage','protocol','fold','model','seed')


def identity(job):return '/'.join(str(job[k]) for k in KEYS)
def folder(cfg,job):return c.case_dir(cfg,job['stage'],job['protocol'],job['fold'],job['model'],job['seed'])


def manifest(cfg):
    return [dict(stage=stage,protocol=p,fold=f,model=m,seed=s,status='pending')
            for s in cfg['seeds'] for f in cfg['folds'] for p in cfg['protocols']
            for m in cfg['models'] for stage in ('calibration','refit')]


def prediction_manifest(cfg,models):
    return [dict(stage='prediction',protocol=p,fold=f,model=m,seed=s,status='pending')
            for s in cfg['seeds'] for f in cfg['folds'] for p in cfg['protocols'] for m in models]


def eligible(job,lookup):
    if job['status']!='pending':return False
    if job['stage']!='refit':return True
    return lookup[identity(dict(job,stage='calibration'))]['status']=='completed'


def receipt(cfg,job):
    d=folder(cfg,job);r=c.read(d/'COMPLETE.json')
    assert all(r[k]==job[k] for k in ('protocol','fold','model','seed'))
    allowed={'calibration':('COMPLETE_CALIBRATION','COMPLETE_HOLD'),
             'refit':('COMPLETE_REFIT',),'prediction':('COMPLETE_PREDICTION',)}
    assert r['status'] in allowed[job['stage']] and r['Test_labels_parsed'] is False
    for name,digest in r['files'].items():
        p=Path(name)
        if not p.is_absolute():p=d/p if len(p.parts)==1 else p
        assert p.resolve().is_relative_to(d.resolve())
        assert c.sha(p)==digest,name
    return r


def witness(cfg,cpu):
    import torch
    from scripts import timesnet_fivefold_neural_20261029_v20 as nn
    from scripts.run_convtran_context_20261022 import configure
    out=Path(cfg['output']);name='CPU_WITNESS.json' if cpu else 'CUDA_WITNESS.json'
    assert not (out/name).exists()
    if cpu:assert not torch.cuda.is_available()
    guard=None if cpu else c.controller_gpu_lock(cfg)
    try:
        for pattern in ('test_timesnet_fivefold_neural_20261029_v20.py','test_timesnet_fivefold_queue_20261029_v20.py'):
            subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p',pattern],check=True)
        if not cpu:
            configure();torch.cuda.set_per_process_memory_fraction(cfg['runtime']['neural_fraction'])
        records=nn.witness(cfg,'cpu' if cpu else 'cuda',out/('cpu_work' if cpu else 'cuda_work'))
        c.save(out/name,dict(status='PASS',code=c.code(cfg),records=records,real_data_rows_read=0,CUDA_validated=not cpu))
        print('TIMESNET_FIVEFOLD_CPU_PASS' if cpu else 'TIMESNET_FIVEFOLD_CUDA_PASS',flush=True)
    finally:
        if guard:guard.close()


def preflight(cfg):
    from scripts import timesnet_fivefold_neural_20261029_v20 as nn
    from scripts.run_convtran_context_20261022 import configure
    out=Path(cfg['output']);assert not (out/'PREFLIGHT.json').exists();guard=c.controller_gpu_lock(cfg)
    try:
        configure();records=nn.preflight(cfg)
        c.save(out/'PREFLIGHT.json',dict(status='PASS',code=c.code(cfg),records=records,Test_labels_parsed=False))
        print('TIMESNET_FIVEFOLD_PREFLIGHT_PASS',flush=True)
    finally:guard.close()


def prepare(cfg):
    import shutil
    out=Path(cfg['output']);assert not (out/'SEAL.json').exists();code=c.code(cfg)
    for name in ('CPU_WITNESS.json','CUDA_WITNESS.json','PREFLIGHT.json'):
        r=c.read(out/name);assert r['status']=='PASS' and r['code']==code
    assert shutil.disk_usage('.').free>=cfg['budget']['minimum_free_disk_GiB']*2**30
    sources,cases=c.source_manifest(cfg);evidence=c.evidence_gate(cfg)
    assert evidence['status']=='PASS';sources.update(evidence['files'])
    jobs=manifest(cfg);assert len(jobs)==60
    c.save(out/'JOB_MANIFEST.json',dict(training_jobs=jobs,calibration=30,final_neural=30,predictions_after_training=30))
    c.save(out/'SEAL.json',dict(config=cfg,code=code,sources=sources,cases=cases,prior_training_evidence=evidence['records'],
        manifest_sha256=c.sha(out/'JOB_MANIFEST.json'),gates={n:c.sha(out/n) for n in ('CPU_WITNESS.json','CUDA_WITNESS.json','PREFLIGHT.json')},
        Test_targets_opened=False,calibration_refit_distinct=True,selected_refit_endpoint_not_stationarity_claim=True,
        user_released_previous_fivefold_hold=True))
    print('TIMESNET_FIVEFOLD_SEALED',flush=True)


def dispatch(cfg,job,slot,phase):
    assert not (folder(cfg,job)/'STARTED.json').exists(),'No scientific retry'
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
    if free<cfg['runtime']['neural_minimum_free_MiB']:return None
    log=Path(cfg['output'])/'logs'/(identity(job).replace('/','_')+'.log');log.parent.mkdir(exist_ok=True)
    args=[sys.executable,'-s','-u','-m',MODULE,'--mode','worker','--slot',str(slot)]
    for k in KEYS:args+=['--'+k,str(job[k])]
    limit=cfg['budget']['prediction_worker_seconds'] if phase=='prediction' else cfg['budget']['worker_seconds']
    stream=log.open('xb')
    try:proc=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    except BaseException:stream.close();raise
    job.update(status='running',pid=proc.pid,log=str(log),started_unix=time.time())
    return dict(process=proc,stream=stream,job=job,tick=time.monotonic(),limit=limit)


def kill_own(proc):
    if proc.poll() is None:
        try:os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            proc.wait(timeout=10)


def run_jobs(cfg,jobs,phase,start):
    out=Path(cfg['output']);lookup={identity(j):j for j in jobs};active={};errors=[]
    assert len(lookup)==len(jobs)
    try:
        while True:
            exhausted=time.monotonic()-start>=cfg['budget']['queue_seconds']
            if exhausted and not errors:errors.append('QUEUE_TIMEOUT')
            for slot,a in list(active.items()):
                proc=a['process'];rc=proc.poll()
                if rc is None and (time.monotonic()-a['tick']>a['limit']+120 or exhausted):
                    kill_own(proc);rc=proc.returncode
                if rc is not None:
                    a['stream'].close();job=a['job']
                    try:
                        assert rc==0,f'exit {rc}'
                        r=receipt(cfg,job);job['status']='hold' if r['status']=='COMPLETE_HOLD' else 'completed'
                        job['seconds']=time.monotonic()-a['tick']
                    except Exception as exc:job['status']='failed';errors.append(dict(job=identity(job),error=repr(exc)))
                    del active[slot]
            for j in jobs:
                if j['stage']=='refit' and j['status']=='pending':
                    if lookup[identity(dict(j,stage='calibration'))]['status'] in ('hold','blocked','failed'):j['status']='blocked'
            if not errors:
                for slot in range(3):
                    if slot in active:continue
                    candidates=[j for j in jobs if eligible(j,lookup)]
                    if not candidates:continue
                    job=candidates[0]
                    try:item=dispatch(cfg,job,slot,phase)
                    except BaseException as exc:
                        job['status']='failed';errors.append(dict(job=identity(job),error=repr(exc)));break
                    if item:active[slot]=item
            counts={state:sum(j['status']==state for j in jobs) for state in ('pending','running','completed','hold','blocked','failed')}
            c.save(out/'QUEUE_STATE.json',dict(status='DRAINING_FAILURE' if errors else 'RUNNING',phase=phase,counts=counts,jobs=jobs,errors=errors,updated_unix=time.time()))
            if errors and not active:raise RuntimeError(str(errors))
            if not active and not any(eligible(j,lookup) for j in jobs):break
            time.sleep(5)
        return jobs
    finally:
        # Unexpected exceptions also retire only our own verified process groups.
        for item in active.values():
            try:kill_own(item['process'])
            finally:item['stream'].close()


def training_lock(cfg,jobs):
    assert all(j['status'] in ('completed','hold','blocked') for j in jobs)
    files={};models=[];out=Path(cfg['output'])
    for model in cfg['models']:
        finals=[j for j in jobs if j['stage']=='refit' and j['model']==model]
        if len(finals)==30 and all(j['status']=='completed' for j in finals):
            models.append(model)
            # Freeze both calibration decisions and the resulting final fits.
            for job in [j for j in jobs if j['model']==model]:
                r=receipt(cfg,job);d=folder(cfg,job);files[str(d/'COMPLETE.json')]=c.sha(d/'COMPLETE.json')
                for name,digest in r['files'].items():
                    p=Path(name);p=d/p if not p.is_absolute() and len(p.parts)==1 else p
                    files[str(p)]=digest
    c.save(out/'TRAINING_LOCK.json',dict(status='FROZEN_TRAINING',seal_sha256=c.sha(out/'SEAL.json'),eligible_models=models,
        held_models=[m for m in cfg['models'] if m not in models],files=files,training_jobs=jobs,no_more_training_dispatch=True,Test_labels_parsed=False))
    return models


def run(cfg):
    seal=c.check(cfg);out=Path(cfg['output']);assert not (out/'LAUNCH.json').exists()
    assert seal['manifest_sha256']==c.sha(out/'JOB_MANIFEST.json')
    guard=c.controller_gpu_lock(cfg);start=time.monotonic()
    def interrupted(signum,frame):raise RuntimeError('Controller interrupted: '+str(signum))
    previous={s:signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        with (out/'LAUNCH.json').open('x',encoding='utf-8') as stream:
            import json
            json.dump(dict(pid=os.getpid(),unix=time.time(),training_jobs=60,expected_final_models=30,prediction_jobs=30),stream)
        jobs=run_jobs(cfg,copy.deepcopy(c.read(out/'JOB_MANIFEST.json')['training_jobs']),'training',start)
        c.check(cfg)
        for p,digest in seal['sources'].items():assert c.sha(p)==digest,p
        models=training_lock(cfg,jobs)
        if models:
            run_jobs(cfg,prediction_manifest(cfg,models),'prediction',start)
            from scripts.report_timesnet_fivefold_20261029_v20 import report
            result=report(cfg)
        else:result=dict(status='TIMESNET_HOLD_NO_TEST_SCORING',completed_models=[],held_models=cfg['models'])
        c.save(out/'COMPLETE.json',result)
        c.save(out/'QUEUE_STATE.json',dict(status=result['status'],phase='complete',training_jobs=jobs,eligible_models=models,
            held_models=[m for m in cfg['models'] if m not in models],updated_unix=time.time()))
    except BaseException as exc:
        c.save(out/'FAILURE.json',dict(error=repr(exc),no_blind_retry=True,unix=time.time()));raise
    finally:
        for s,h in previous.items():signal.signal(s,h)
        guard.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['cpu_witness','cuda_witness','preflight','prepare','run','worker'])
    for k in KEYS:p.add_argument('--'+k,type=int if k=='seed' else str)
    p.add_argument('--slot',type=int);a=p.parse_args();cfg=c.config()
    if a.mode.endswith('witness'):witness(cfg,a.mode=='cpu_witness')
    elif a.mode=='preflight':preflight(cfg)
    elif a.mode=='prepare':prepare(cfg)
    elif a.mode=='run':run(cfg)
    else:
        from scripts import timesnet_fivefold_neural_20261029_v20 as worker
        assert a.model in cfg['models'] and a.protocol in cfg['protocols'] and a.fold in cfg['folds'] and a.seed in cfg['seeds']
        assert a.slot in range(3)
        {'calibration':worker.calibrate,'refit':worker.refit,'prediction':worker.predict}[a.stage](cfg,a.protocol,a.fold,a.model,a.seed,a.slot)


if __name__=='__main__':main()
