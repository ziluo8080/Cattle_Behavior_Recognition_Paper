"""Administrative continuation of the already sealed fair-selection group."""
from pathlib import Path
import argparse,fcntl,gc,json,os,subprocess,sys,time
import torch
from scripts.sensor_fair_selection_20261113 import main as fair

ROOT=Path('/q3/mmcows-sensor-behavior-20261007')
PLAN=ROOT/'configs/sensor_recovery_20261114.json'
def run():
 cfg=fair.setup();out=Path(cfg['output']);fair.check(cfg)
 lock=(out/'CONTINUATION.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (out/'QUEUE_STARTED.json').exists(),'Do not duplicate or restart scientific fits'
 old=fair.common.read(ROOT/'reports/sensor_ablation_20261113/QUEUE_STATE.json');assert old['status']=='STOPPED_ERROR'
 for job in old['jobs']:
  pid=job.get('pid')
  if pid:assert not (Path('/proc')/str(pid)).exists(),'Old worker still alive'
 fair.common.save(out/'CONTINUATION_AUTH.json',{'unix':time.time(),'pid':os.getpid(),'recovery_plan_sha256':fair.common.sha(PLAN),'old_failure_record_sha256':fair.common.sha(ROOT/'reports/sensor_ablation_20261113/QUEUE_STATE.json'),'scientific_config_changed':False,'dependency_order_override':'Old structural queue has stopped. Run the already-authorized matched reference first; subsequent recovery arms use this same protocol.'})
 fair.witness(cfg,'cuda');assert fair.common.read(out/'GPU_WITNESS.json')['code']==fair.common.read(out/'SEAL.json')['code'];gc.collect();torch.cuda.empty_cache()
 fair.execute(cfg)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--launch',action='store_true');a=ap.parse_args();assert Path.cwd()==ROOT
 if a.launch:
  cfg=fair.setup();out=Path(cfg['output']);assert not (out/'QUEUE_STARTED.json').exists();assert not (out/'CONTINUATION_DISPATCH.json').exists()
  stream=(out/'continuation.log').open('w');p=subprocess.Popen([sys.executable,'-s','-u',__file__],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  fair.common.save(out/'CONTINUATION_DISPATCH.json',{'pid':p.pid,'unix':time.time()});print('REFERENCE_PID',p.pid)
 else:run()
if __name__=='__main__':main()
