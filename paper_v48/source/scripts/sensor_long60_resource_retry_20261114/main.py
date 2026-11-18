"""Resource-only serial retry of the failed long-window arm; old results immutable."""
from pathlib import Path
import argparse,copy,fcntl,gc,json,os,subprocess,sys,time,traceback
import torch
from scripts.sensor_recovery_20261114 import core as c

ROOT=c.ROOT;OLD=ROOT/'reports/sensor_recovery_20261114/runs/long_60_60';OUT=ROOT/'reports/sensor_long60_resource_retry_20261114'
ORIGINAL_CODE=c.code
def config():
 cfg=c.setup('long_60_60');cfg=dict(cfg,output=str(OUT.relative_to(ROOT)));return cfg
def code(cfg):return {**ORIGINAL_CODE(cfg),'scripts/sensor_long60_resource_retry_20261114/main.py':c.sha(__file__)}
def check(cfg):
 s=c.read(OUT/'SEAL.json');assert s['config']==cfg and s['code']==code(cfg)
 for p,h in s['sources'].items():assert c.sha(p)==h,p
 return s
def gpu(cfg,slot,kind='neural'):
 assert slot==0 and kind=='neural'
 free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=20*1024,'Wait for other jobs; never stop them'
 h=(OUT/'SERIAL_GPU.lock').open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.cuda.set_per_process_memory_fraction(.85);c.fair.parent.core.configure();return h
def setup():
 cfg=config();c.check=check;c.fair.check=check;c.fair.code=code;c.fair.common.check=check;c.fair.common.gpu_lock=gpu
 return cfg
def prepare(cfg):
 assert not (OUT/'SEAL.json').exists();OUT.mkdir(parents=True,exist_ok=True)
 original=c.read(OLD/'SEAL.json');oldcfg=dict(cfg,output=str(OLD.relative_to(ROOT)));assert original['config']==oldcfg
 for p,h in original['code'].items():assert c.sha(p)==h,p
 for p,h in original['sources'].items():assert c.sha(p)==h,p
 (OUT/'INPUT_LOCK.json').write_bytes((OLD/'INPUT_LOCK.json').read_bytes())
 sources={str(OLD/'SEAL.json'):c.sha(OLD/'SEAL.json'),str(OUT/'INPUT_LOCK.json'):c.sha(OUT/'INPUT_LOCK.json'),**original['sources']}
 for p in cfg['protocols']:
  for f in cfg['folds']:
   src=OLD/'trees'/p/f;done=c.read(src/'COMPLETE.json');assert done['status']=='FIXED500_COMPLETE' and c.sha(src/'model.ubj')==done['model_sha256']
   dst=OUT/'trees'/p/f;dst.mkdir(parents=True,exist_ok=True)
   for name in ['model.ubj','COMPLETE.json']:
    if not (dst/name).exists():os.link(src/name,dst/name)
    assert c.sha(dst/name)==c.sha(src/name);sources[str(dst/name)]=c.sha(dst/name)
 c.save(OUT/'SEAL.json',{'config':cfg,'code':code(cfg),'sources':sources,'unix':time.time(),'resource_changes':{'old_fraction':.45,'new_fraction':.85,'gpu_workers':1},'scientific_changes':[],'old_oom_logs_preserved':True})
def witness(cfg):
 c.fair.worker.witness(cfg,'cpu',OUT/('cpu_witness_'+str(time.time_ns())))
 c.save(OUT/'CPU_WITNESS.json',{'status':'PASS','code':code(cfg),'real_data_rows':0,'unix':time.time()});print('LONG60_CPU_WITNESS_PASS',flush=True)
def preflight(cfg):
 check(cfg);lock=gpu(cfg,0);data=c.load(cfg,'TS','fold_1','calibration');state=c.fair.parent.new(cfg,'long_60_60',data['prior'],17,'cuda');ids=data['train'][:256]
 times=[];torch.cuda.reset_peak_memory_stats()
 for _ in range(2):
  start=time.monotonic();loss,mass=c.fair.parent.update(state,data,ids);torch.cuda.synchronize();times.append(time.monotonic()-start)
 assert torch.isfinite(torch.tensor(loss))
 c.save(OUT/'REAL_TRAIN_PREFLIGHT.json',{'status':'PASS','batch':256,'dynamic_steps':600,'memory_fraction':.85,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'update_seconds':times,'Test_labels_parsed':False,'no_scientific_checkpoint_adopted':True,'code':code(cfg),'unix':time.time()});del state,data;gc.collect();torch.cuda.empty_cache();lock.close()
def execute(cfg):
 handle=(OUT/'QUEUE.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'QUEUE_STARTED.json').exists();c.save(OUT/'QUEUE_STARTED.json',{'pid':os.getpid(),'unix':time.time()});jobs=[]
 for p in cfg['protocols']:
  for f in cfg['folds']:
   for seed in cfg['seeds']:
    for mode in ['calibration','refit']:jobs.append({'protocol':p,'fold':f,'seed':seed,'mode':mode,'status':'pending'})
 try:
  for j in jobs:
   log=OUT/'logs'/f'{j["protocol"]}_{j["fold"]}_{j["seed"]}_{j["mode"]}.log';log.parent.mkdir(exist_ok=True)
   with log.open('w') as stream:
    args=[sys.executable,'-s','-u',__file__,'--mode',j['mode'],'--protocol',j['protocol'],'--fold',j['fold'],'--seed',str(j['seed'])];proc=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    j.update(status='running',pid=proc.pid,log=str(log),started_unix=time.time());c.save(OUT/'QUEUE_STATE.json',{'status':'RUNNING','jobs':jobs,'unix':time.time()})
    try:rc=proc.wait(timeout=cfg['budget']['worker_seconds']+120)
    except subprocess.TimeoutExpired:
     proc.terminate()
     try:proc.wait(timeout=30)
     except subprocess.TimeoutExpired:proc.kill();proc.wait()
     j.update(status='failed',error='owned_worker_timeout');raise
   j.update(status='completed' if rc==0 else 'failed',exit_code=rc,ended_unix=time.time());assert rc==0,str(log)
   path=c.fair.common.case_dir(cfg,j['mode'],j['protocol'],j['fold'],'long_60_60',j['seed']);c.fair.worker._receipt(path,'COMPLETE_CALIBRATION' if j['mode']=='calibration' else 'COMPLETE_REFIT')
  c.training_lock(cfg)
  for p in cfg['protocols']:
   for f in cfg['folds']:
    for seed in cfg['seeds']:
     c.fair.worker.predict(cfg,p,f,'long_60_60',seed,0);gc.collect();torch.cuda.empty_cache()
  c.score(cfg);c.save(OUT/'QUEUE_STATE.json',{'status':'COMPLETE','jobs':jobs,'unix':time.time()})
 except BaseException as e:
  c.save(OUT/'QUEUE_STATE.json',{'status':'STOPPED_ERROR','jobs':jobs,'error':str(e),'traceback':traceback.format_exc(),'unix':time.time()});raise
def guardian(cfg):
 handle=(OUT/'GUARDIAN.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  began=time.time()
  while True:
   assert time.time()-began<cfg['dependency_wait_seconds'],'Dependency wait watchdog exceeded'
   state=c.read(ROOT/'reports/sensor_recovery_20261114/QUEUE_STATE.json');status=state['status'];free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
   if status in ['COMPLETE','COMPLETE_WITH_HOLDS'] and free>=20*1024:break
   if status=='STOPPED_ERROR':c.save(OUT/'QUEUE_STATE.json',{'status':'HELD_DEPENDENCY_ERROR','unix':time.time()});return
   c.save(OUT/'QUEUE_STATE.json',{'status':'WAITING_FOR_RECOVERY_QUEUE','dependency_status':status,'free_MiB':free,'unix':time.time()});time.sleep(60)
  c.save(OUT/'QUEUE_STATE.json',{'status':'REAL_TRAIN_MEMORY_PREFLIGHT','unix':time.time()});preflight(cfg);execute(cfg)
 except BaseException as e:
  path=OUT/'QUEUE_STATE.json';state=c.read(path) if path.exists() else {};state.update(status='STOPPED_ERROR',error=str(e),traceback=traceback.format_exc(),unix=time.time());c.save(path,state);raise
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['prepare','witness','launch','guardian','calibration','refit'],required=True);ap.add_argument('--protocol');ap.add_argument('--fold');ap.add_argument('--seed',type=int);a=ap.parse_args();assert Path.cwd()==ROOT;cfg=setup()
 if a.mode=='prepare':prepare(cfg)
 elif a.mode=='witness':witness(cfg)
 elif a.mode=='calibration':c.fair.worker.calibrate(cfg,a.protocol,a.fold,'long_60_60',a.seed,0)
 elif a.mode=='refit':c.fair.worker.refit(cfg,a.protocol,a.fold,'long_60_60',a.seed,0)
 elif a.mode=='guardian':guardian(cfg)
 elif a.mode=='launch':
  check(cfg);assert c.read(OUT/'CPU_WITNESS.json')['code']==code(cfg);assert not (OUT/'DISPATCH.json').exists();stream=(OUT/'guardian.log').open('w');p=subprocess.Popen([sys.executable,'-s','-u',__file__,'--mode','guardian'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);c.save(OUT/'DISPATCH.json',{'pid':p.pid,'unix':time.time()});print('LONG60_GUARDIAN_PID',p.pid)
if __name__=='__main__':main()
