"""Prepare inputs, wait for existing shared work, then verify and run all folds."""
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from scripts.shared_fusion_fivefold_common_20261024 import configure,read,save,check_method_lock


def main():
    cfg=configure();check_method_lock(cfg);out=Path(cfg['output'])
    assert not (out/'BOOTSTRAP.json').exists(),'Inspect existing pipeline before any restart'
    started=time.time();save(out/'BOOTSTRAP.json',dict(pid=os.getpid(),started_unix=started))
    def progress(phase,**extra):
        save(out/'PIPELINE_PROGRESS.json',dict(phase=phase,pid=os.getpid(),updated_unix=time.time(),elapsed_seconds=time.time()-started,**extra))
    def command(phase,module,args,timeout):
        progress(phase)
        with (out/(phase.lower()+'.log')).open('wb') as log:
            completed=subprocess.run([sys.executable,'-s','-u','-m',module]+args,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        assert completed.returncode==0,f'{phase} failed; read {phase.lower()}.log'
    try:
        command('DEVELOPMENT_INPUT_REPLAY','scripts.prepare_shared_fusion_fivefold_20261024',['--development-replay'],1800)
        command('PREPARING_FORMAL_INPUTS','scripts.prepare_shared_fusion_fivefold_20261024',[],3600)
        assert read(out/'FIVEFOLD_COVERAGE.json')['status']=='PASS'
        wait_started=time.monotonic();guard=cfg['resource_wait']
        while True:
            state_path=Path(guard['existing_queue']);pending=False;counts={}
            if state_path.exists():
                jobs=read(state_path).get('jobs',[])
                counts={s:sum(j.get('status')==s for j in jobs) for s in ('pending','running','completed','failed','blocked')}
                pending=any(j.get('status') not in ('completed','failed','blocked','cancelled') for j in jobs)
            process_lines=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
            live_other=any(guard['existing_process_token'] in line and 'python' in line for line in process_lines)
            used=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()[0])
            if not (pending and live_other) and used<500:break
            elapsed=time.monotonic()-wait_started
            progress('WAITING_FOR_SHARED_GPU',resource_wait_seconds=elapsed,existing_queue_counts=counts,gpu_memory_used_MiB=used)
            if elapsed>guard['max_wait_seconds']:raise RuntimeError('Bounded shared-GPU wait exhausted; no unrelated process stopped')
            time.sleep(5)
        command('CUDA_RUNTIME_WITNESS','scripts.run_shared_fusion_fivefold_20261024',['--mode','witness','--witness-output',str(out/'RUNTIME_WITNESS.json')],300)
        command('FORMAL_FIT_PREFLIGHT','scripts.run_shared_fusion_fivefold_20261024',['--mode','preflight'],2400)
        command('TRAINING_SEAL','scripts.freeze_shared_fusion_fivefold_20261024',[],600)
        command('FIVEFOLD_TRAINING','scripts.queue_shared_fusion_fivefold_20261024',[],cfg['budget']['total_seconds']+1800)
        result=read(out/'COMPLETE.json')
        progress(result['status'],final1pp_established=result.get('final1pp_established',False),formal_evaluation_finished=result.get('Test_evaluated',False))
        save(out/'PIPELINE_COMPLETE.json',dict(status=result['status'],started_unix=started,finished_unix=time.time(),method_automatically_changed=False))
    except BaseException:
        save(out/'BOOTSTRAP_FAILED.json',dict(status='FAILED',traceback=traceback.format_exc(),unix=time.time()))
        progress('FAILED_STOP_FOR_INSPECTION');raise


if __name__=='__main__':main()
