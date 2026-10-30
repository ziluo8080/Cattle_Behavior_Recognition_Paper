"""Forty fixed refits; Test evaluation only after the entire queue completes."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from scripts.shared_fusion_fivefold_common_20261024 import configure,read,save,sha,check_method_lock
from scripts.freeze_shared_fusion_fivefold_20261024 import check_training_seal


def run():
    cfg=configure();check_training_seal(cfg);out=Path(cfg['output']);manifest=read(out/'JOB_MANIFEST.json')
    assert not (out/'LAUNCH.json').exists(),'No automatic queue restart'
    save(out/'LAUNCH.json',dict(pid=os.getpid(),started_unix=time.time(),jobs=40,neural_jobs=30,tree_jobs=10))
    state={j['id']:dict(j,status='pending',attempts=0) for j in manifest['jobs']};running={};start=time.monotonic()
    limits=dict(cpu=manifest['max_parallel_cpu'],gpu=manifest['max_parallel_gpu'])
    def stop(process):
        if process.poll() is not None:return
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    try:
        while True:
            for name,(process,handle,tick) in list(running.items()):
                j=state[name]
                if process.poll() is None and time.monotonic()-tick>j['timeout_seconds']:
                    stop(process);j['error']='JOB_BUDGET_EXHAUSTED'
                if process.poll() is not None:
                    handle.close();j['returncode']=process.returncode
                    valid=process.returncode==0 and Path(j['expected']).exists()
                    if valid:valid=read(j['expected']).get('status')=='COMPLETE'
                    j.update(status='completed' if valid else 'failed',finished_unix=time.time());del running[name]
            if time.monotonic()-start>manifest['total_seconds']:
                for name,(p,h,_) in running.items():stop(p);h.close();state[name].update(status='failed',error='GLOBAL_BUDGET_EXHAUSTED')
                running.clear()
                for j in state.values():
                    if j['status']=='pending':j.update(status='blocked',error='GLOBAL_BUDGET_EXHAUSTED')
            for j in state.values():
                if j['status']!='pending':continue
                if sum(state[n]['resource']==j['resource'] for n in running)>=limits[j['resource']]:continue
                if j['resource']=='gpu':
                    used=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()[0])
                    if used>=500:continue
                logs=out/'queue_logs';logs.mkdir(exist_ok=True);handle=(logs/(j['id']+'.log')).open('wb')
                command=[sys.executable,'-s','-u','-m',j['module']]+j['args']
                process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                j.update(status='running',pid=process.pid,attempts=1,started_unix=time.time());running[j['id']]=(process,handle,time.monotonic())
            snapshot=dict(updated_unix=time.time(),elapsed_seconds=time.monotonic()-start,jobs=list(state.values()))
            save(out/'QUEUE_STATE.json',snapshot)
            save(out/'PUBLIC_PROGRESS.json',dict(updated_unix=time.time(),elapsed_seconds=time.monotonic()-start,
                counts={s:sum(j['status']==s for j in state.values()) for s in ('pending','running','completed','failed','blocked')},running=list(running),Test_scores_computed=0))
            if not running and all(j['status'] in ('completed','failed','blocked') for j in state.values()):break
            time.sleep(2)
        check_training_seal(cfg);check_method_lock(cfg,full_sources=True)
        seal=read(out/'SEAL.json')
        for path,digest in seal['data_files'].items():assert sha(path)==digest,path
        from scripts.report_shared_fusion_fivefold_20261024 import report
        result=report(cfg,snapshot);save(out/'COMPLETE.json',result)
        progress=read(out/'PUBLIC_PROGRESS.json');progress['final_status']=result['status'];progress['Test_scores_computed']=result.get('Test_evaluated',False)
        save(out/'PUBLIC_PROGRESS.json',progress);print(result,flush=True)
    except BaseException:
        for p,h,_ in running.values():stop(p);h.close()
        save(out/'QUEUE_FAILURE.json',dict(status='FAILED',traceback=traceback.format_exc(),unix=time.time()));raise


if __name__=='__main__':run()
