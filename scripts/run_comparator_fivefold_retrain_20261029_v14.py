"""One bounded three-slot queue, calibrated refits, frozen predictions and scoring."""
import argparse
import copy
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from scripts import comparator_fivefold_common_20261029_v14 as c

MODULE='scripts.run_comparator_fivefold_retrain_20261029_v14'
KEYS=('stage','protocol','fold','model','seed')
def identity(job):return '/'.join(str(job[k]) for k in KEYS)
def manifest(cfg):
    jobs=[]
    for seed in cfg['seeds']:
        for fold in cfg['folds']:
            for protocol in cfg['protocols']:
                for model in cfg['models']:
                    for stage in ['calibration','refit']:
                        jobs.append(dict(stage=stage,protocol=protocol,fold=fold,model=model,seed=seed,status='pending'))
                jobs.append(dict(stage='refit',protocol=protocol,fold=fold,model=cfg['rocket_model'],seed=seed,status='pending'))
    return jobs
def folder(cfg,job):return c.case_dir(cfg,job['stage'],job['protocol'],job['fold'],job['model'],job['seed'])
def eligible(job,lookup):
    if job['status']!='pending':return False
    if job['stage']!='refit' or job['model']=='MultiRocket_Fusion':return True
    dependency=dict(job,stage='calibration')
    return lookup[identity(dependency)]['status']=='completed'
def receipt(cfg,job):
    directory=folder(cfg,job);r=c.read(directory/'COMPLETE.json')
    assert all(r[k]==job[k] for k in ['protocol','fold','model','seed'])
    expected={'calibration':['COMPLETE_CALIBRATION','COMPLETE_HOLD'],'refit':['COMPLETE_REFIT'],'prediction':['COMPLETE_PREDICTION']}
    assert r['status'] in expected[job['stage']]
    assert r['Test_labels_parsed'] is False
    for name,digest in r['files'].items():
        path=Path(name)
        if not path.is_absolute():path=directory/path if len(path.parts)==1 else path
        assert path.resolve().is_relative_to(directory.resolve())
        assert c.sha(path)==digest,name
    return r
def convergence_evidence(cfg):
    from scripts.comparator_convergence_criteria_20261027_v2 import assess
    from scripts.run_comparator_stability_extension_20261028_v9 import training_components
    rows=[];files={}
    for m in cfg['models']:
        for p in cfg['protocols']:
            for inner in [0,1,2]:
                for seed in cfg['seeds']:
                    suffix=Path('trajectories')/p/f'inner{inner}'/m/f'seed{seed}'
                    base=Path('reports/sensor_comparator_repair_20261028_v7')/suffix
                    r=c.read(base/'COMPLETE.json')
                    training=training_components(r['legacy_terminal_optimization'],cfg['convergence'])
                    if not training['passed']:
                        base=Path('reports/sensor_comparator_stability_extension_20261028_v9')/suffix
                        r=c.read(base/'COMPLETE.json');training=r['training_components']
                    assert training['passed'],str(base)
                    history=base/'history.json'
                    assert c.sha(history)==r['files'][str(history)]
                    assert training==training_components(assess(c.read(history),cfg['convergence']),cfg['convergence'])
                    for path in [base/'COMPLETE.json',history]:files[str(path)]=c.sha(path)
                    rows.append(dict(model=m,protocol=p,inner=inner,seed=seed,source=str(base),training_stable=True))
    return rows,files
def witness(cfg,cpu):
    import torch
    from scripts import comparator_fivefold_neural_20261029_v14 as neural
    from scripts import comparator_fivefold_rocket_20261029_v14 as rocket
    from scripts.run_convtran_context_20261022 import acquire_gpu
    out=Path(cfg['output']);name='CPU_WITNESS.json' if cpu else 'CUDA_WITNESS.json'
    assert not (out/name).exists()
    if cpu:assert not torch.cuda.is_available()
    guard=None if cpu else acquire_gpu()
    try:
        for pattern in ['test_comparator_fivefold_neural_20261029_v14.py','test_comparator_fivefold_rocket_20261029_v14.py','test_comparator_fivefold_queue_20261029_v14.py']:
            subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p',pattern],check=True)
        device='cpu' if cpu else 'cuda'
        if not cpu:torch.cuda.set_per_process_memory_fraction(.15)
        nn=neural.witness(cfg,device,out/(device+'_neural_work_r3'))
        if not cpu:torch.cuda.empty_cache();torch.cuda.set_per_process_memory_fraction(.65)
        mr=rocket.witness(cfg,device,out/(device+'_rocket_work_r3'))
        c.save(out/name,dict(status='PASS',code=c.code(cfg),neural=nn,rocket=mr,real_data_rows_read=0,CUDA_validated=not cpu))
        print('V14_CPU_WITNESS_PASS' if cpu else 'V14_CUDA_WITNESS_PASS',flush=True)
    finally:
        if guard:guard.close()
def preflight(cfg):
    from scripts import comparator_fivefold_neural_20261029_v14 as neural
    from scripts import comparator_fivefold_rocket_20261029_v14 as rocket
    from scripts.run_convtran_context_20261022 import acquire_gpu
    guard=acquire_gpu();out=Path(cfg['output'])
    assert not (out/'PREFLIGHT.json').exists()
    try:
        nn=neural.preflight(cfg);mr=rocket.preflight(cfg)
        c.save(out/'PREFLIGHT.json',dict(status='PASS',code=c.code(cfg),neural=nn,rocket=mr,Test_labels_parsed=False))
        print('V14_PREFLIGHT_PASS',flush=True)
    finally:guard.close()
def prepare(cfg):
    import shutil
    out=Path(cfg['output']);assert not (out/'SEAL.json').exists()
    code=c.code(cfg)
    for name in ['CPU_WITNESS.json','CUDA_WITNESS.json','PREFLIGHT.json']:
        r=c.read(out/name);assert r['status']=='PASS' and r['code']==code
    assert shutil.disk_usage('.').free>=cfg['budget']['minimum_free_disk_GiB']*2**30
    sources,cases=c.source_manifest(cfg);evidence,prior_files=convergence_evidence(cfg)
    sources.update(prior_files)
    assert max(r['test_rows'] for r in cases)<=max(r['final_fit_rows'] for r in cases)
    jobs=manifest(cfg);assert len(jobs)==210
    c.save(out/'JOB_MANIFEST.json',dict(training_jobs=jobs,calibration=90,final_neural=90,final_rocket=30))
    c.save(out/'SEAL.json',dict(config=cfg,code=code,sources=sources,cases=cases,prior_training_evidence=evidence,
        manifest_sha256=c.sha(out/'JOB_MANIFEST.json'),gates={n:c.sha(out/n) for n in ['CPU_WITNESS.json','CUDA_WITNESS.json','PREFLIGHT.json']},
        Test_targets_opened=False,calibration_refit_distinct=True,selected_refit_endpoint_not_stationarity_claim=True))
    print('V14_SEALED',flush=True)
def dispatch(cfg,job,slot,phase):
    directory=folder(cfg,job)
    assert not (directory/'STARTED.json').exists(),'No scientific retry'
    kind='rocket' if job['model']==cfg['rocket_model'] else 'neural'
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
    if free<cfg['runtime'][kind+'_minimum_free_MiB']:return None
    log=Path(cfg['output'])/'logs'/(identity(job).replace('/','_')+'.log');log.parent.mkdir(exist_ok=True)
    args=[sys.executable,'-s','-u','-m',MODULE,'--mode','worker','--slot',str(slot)]
    for k in KEYS:args+=['--'+k,str(job[k])]
    limit=cfg['budget']['prediction_worker_seconds'] if phase=='prediction' else cfg['budget']['rocket_worker_seconds'] if kind=='rocket' else cfg['budget']['worker_seconds']
    stream=log.open('xb')
    try:proc=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    except BaseException:stream.close();raise
    job.update(status='running',pid=proc.pid,log=str(log),started_unix=time.time())
    return dict(process=proc,stream=stream,job=job,tick=time.monotonic(),limit=limit)

def run_jobs(cfg,jobs,phase,start):
    out=Path(cfg['output']);lookup={identity(j):j for j in jobs};active={};errors=[]
    while True:
        if time.monotonic()-start>=cfg['budget']['queue_seconds'] and not errors:errors.append('QUEUE_TIMEOUT')
        for slot,a in list(active.items()):
            proc=a['process'];rc=proc.poll()
            if rc is None and (time.monotonic()-a['tick']>a['limit']+120 or time.monotonic()-start>=cfg['budget']['queue_seconds']):
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait();rc=proc.returncode
            if rc is not None:
                a['stream'].close();job=a['job']
                try:
                    assert rc==0,f'exit {rc}'
                    r=receipt(cfg,job)
                    job['status']='hold' if r['status']=='COMPLETE_HOLD' else 'completed'
                    job['seconds']=time.monotonic()-a['tick']
                except Exception as exc:job['status']='failed';errors.append(dict(job=identity(job),error=repr(exc)))
                del active[slot]
        for j in jobs:
            if j['stage']=='refit' and j['model']!=cfg['rocket_model'] and j['status']=='pending':
                if lookup[identity(dict(j,stage='calibration'))]['status'] in ['hold','blocked','failed']:j['status']='blocked'
        if not errors:
            for slot in range(3):
                if slot in active:continue
                candidates=[j for j in jobs if eligible(j,lookup)]
                if slot==2:
                    rocket=[j for j in candidates if j['model']==cfg['rocket_model']]
                    candidates=rocket or [j for j in candidates if j['model']!=cfg['rocket_model']]
                else:candidates=[j for j in candidates if j['model']!=cfg['rocket_model']]
                if not candidates:continue
                job=candidates[0]
                try:a=dispatch(cfg,job,slot,phase)
                except BaseException as exc:
                    job['status']='failed';errors.append(dict(job=identity(job),error=repr(exc)));break
                if a:active[slot]=a
        counts={state:sum(j['status']==state for j in jobs) for state in ['pending','running','completed','hold','blocked','failed']}
        c.save(out/'QUEUE_STATE.json',dict(status='DRAINING_FAILURE' if errors else 'RUNNING',phase=phase,counts=counts,jobs=jobs,errors=errors,updated_unix=time.time()))
        if errors and not active:raise RuntimeError(str(errors))
        if not active and not any(eligible(j,lookup) for j in jobs):break
        time.sleep(5)
    return jobs
def training_lock(cfg,jobs):
    files={};eligible_models=[];all_models=cfg['models']+[cfg['rocket_model']]
    for model in all_models:
        finals=[j for j in jobs if j['stage']=='refit' and j['model']==model]
        if len(finals)==30 and all(j['status']=='completed' for j in finals):
            eligible_models.append(model)
            for j in finals:
                r=receipt(cfg,j);d=folder(cfg,j)
                files[str(d/'COMPLETE.json')]=c.sha(d/'COMPLETE.json')
                for name,digest in r['files'].items():
                    path=Path(name);path=d/path if not path.is_absolute() and len(path.parts)==1 else path
                    files[str(path)]=digest
    c.save(Path(cfg['output'])/'TRAINING_LOCK.json',dict(status='FROZEN_TRAINING',seal_sha256=c.sha(Path(cfg['output'])/'SEAL.json'),
        eligible_models=eligible_models,held_models=[m for m in all_models if m not in eligible_models],files=files,training_jobs=jobs,
        no_more_training_dispatch=True,Test_labels_parsed=False))
    return eligible_models
def run(cfg):
    from scripts.run_convtran_context_20261022 import acquire_gpu
    seal=c.check(cfg);out=Path(cfg['output']);assert not (out/'LAUNCH.json').exists()
    assert seal['manifest_sha256']==c.sha(out/'JOB_MANIFEST.json')
    guard=acquire_gpu();start=time.monotonic()
    c.save(out/'LAUNCH.json',dict(pid=os.getpid(),unix=time.time(),training_jobs=210,expected_final_models=120))
    try:
        jobs=run_jobs(cfg,copy.deepcopy(c.read(out/'JOB_MANIFEST.json')['training_jobs']),'training',start)
        c.check(cfg)
        for path,digest in seal['sources'].items():assert c.sha(path)==digest,path
        models=training_lock(cfg,jobs)
        if models:
            prediction=[dict(stage='prediction',protocol=p,fold=f,model=m,seed=s,status='pending')
                for s in cfg['seeds'] for f in cfg['folds'] for p in cfg['protocols'] for m in models]
            predictions=run_jobs(cfg,prediction,'prediction',start)
            from scripts.report_comparator_fivefold_retrain_20261029_v14 import report
            result=report(cfg)
        else:result=dict(status='ALL_MODELS_HOLD_NO_TEST_SCORING',models=[])
        c.save(out/'COMPLETE.json',result)
        c.save(out/'QUEUE_STATE.json',dict(status=result['status'],phase='complete',training_jobs=jobs,
            eligible_models=models,held_models=[m for m in cfg['models']+[cfg['rocket_model']] if m not in models],updated_unix=time.time()))
    except BaseException as exc:
        c.save(out/'FAILURE.json',dict(error=repr(exc),no_blind_retry=True,unix=time.time()));raise
    finally:guard.close()
def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['cpu_witness','cuda_witness','preflight','prepare','run','worker'])
    for k in KEYS:p.add_argument('--'+k,type=int if k=='seed' else str)
    p.add_argument('--slot',type=int);a=p.parse_args();cfg=c.config()
    if a.mode.endswith('witness'):witness(cfg,a.mode=='cpu_witness')
    elif a.mode=='preflight':preflight(cfg)
    elif a.mode=='prepare':prepare(cfg)
    elif a.mode=='run':run(cfg)
    else:
        assert a.stage in ['calibration','refit','prediction']
        if a.model==cfg['rocket_model']:
            from scripts import comparator_fivefold_rocket_20261029_v14 as worker
            assert a.stage!='calibration'
            fn=worker.predict if a.stage=='prediction' else worker.refit
            fn(cfg,a.protocol,a.fold,a.seed,a.slot)
        else:
            from scripts import comparator_fivefold_neural_20261029_v14 as worker
            fn={'calibration':worker.calibrate,'refit':worker.refit,'prediction':worker.predict}[a.stage]
            fn(cfg,a.protocol,a.fold,a.model,a.seed,a.slot)

if __name__=='__main__':main()
