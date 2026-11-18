"""Append-only matched selection study; reuse the actual DS calibration/refit code."""
from pathlib import Path
import argparse,copy,fcntl,gc,json,os,subprocess,sys,time,traceback
import numpy as np
import torch
from scripts import comparator_fivefold_neural_replay_20261029_v14r1 as worker
from scripts import comparator_fivefold_common_20261029_v14 as common
from scripts import run_comparator_convergence_20261027_v2 as parent
from scripts.sensor_convtran_endpoint_interaction_20261022 import EndpointInteractionConvTran
from scripts import converged_fivefold_common_20261027 as anchor

ROOT=Path('/q3/mmcows-sensor-behavior-20261007')
CONFIG=Path('configs/sensor_fair_selection_20261113.json')
MODEL='ConvTran_ERI'
OLD_MODEL=parent.model
def cfg():return common.read(CONFIG)
def model(config,arm,prior,device):
 assert arm==MODEL
 net=EndpointInteractionConvTran(common.read(config['architecture_config']),prior).to(device)
 for module in net.modules():
  if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.momentum=config['BN_momentum']
 return net
def code(config):
 paths=set(common.read(Path(config['selection_parent'])/'TRAINING_SEAL.json')['code'])
 paths.update([str(CONFIG),__file__,'scripts/comparator_fivefold_neural_replay_20261029_v14r1.py','scripts/comparator_fivefold_common_20261029_v14.py','scripts/run_comparator_convergence_20261027_v2.py','scripts/comparator_convergence_criteria_20261027_v2.py','scripts/comparator_convergence_criteria_20261027.py','scripts/comparator_repair_policy_20261028.py','scripts/run_comparator_stability_extension_20261028_v9.py','scripts/prediction_replay_verifier_20261029_v14r1.py','configs/sensor_comparator_fivefold_retrain_20261029_v14.json','configs/sensor_timesnet_fivefold_20261029_v20.json'])
 return {str(Path(p).resolve().relative_to(ROOT)):common.sha(p) for p in sorted(paths)}
def check(config):
 s=common.read(Path(config['output'])/'SEAL.json');assert s['config']==config and s['code']==code(config)
 for p,h in s['sources'].items():
  if p.endswith(('RECEIPT.json','REFIT_SOURCE.json','DATA_RECEIPT.json','TRAINING_SEAL.json','TRAINING_LOCK.json','PREDICTION_LOCK.json','/SEAL.json')):assert common.sha(p)==h,p
 return s
def gpu_lock(config,slot,kind='neural'):
 assert kind=='neural' and slot in [0,1]
 free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=11*1024
 handle=(Path(config['output'])/f'GPU_SLOT_{slot}.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.cuda.set_per_process_memory_fraction(.45)
 parent.core.configure()
 return handle
def setup():
 assert Path.cwd()==ROOT
 parent.setup(False);worker.runtime();worker.MODELS=(MODEL,);parent.model=model;common.check=check;common.gpu_lock=gpu_lock
 return cfg()
def witness(config,device='cpu'):
 out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
 if device=='cuda':
  free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=11*1024;torch.cuda.set_per_process_memory_fraction(.45);parent.core.configure()
 worker.witness(config,device,out/(device+'_witness_'+str(time.time_ns())))
 # The old DS worker verifies selection, EMA preservation, checkpoint reload and RNG.
 common.save(out/('GPU_WITNESS.json' if device=='cuda' else 'CPU_WITNESS.json'),{'status':'PASS','unix':time.time(),'real_data_rows_read':0,'device':device,'code':code(config)})
 print('FAIR_SELECTION_'+device.upper()+'_WITNESS_PASS',flush=True)
def seal(config):
 out=Path(config['output']);assert not (out/'SEAL.json').exists()
 proof=common.read(out/'CPU_WITNESS.json');assert proof['status']=='PASS' and proof['code']==code(config)
 files,cases=common.source_manifest(config)
 for name in ['TRAINING_LOCK.json','PREDICTION_LOCK.json']:
  path=Path(config['selection_parent'])/name;files[str(path)]=common.sha(path)
 assert common.read(Path(config['selection_parent'])/'PREDICTION_LOCK.json')['training_lock_sha256']==files[str(Path(config['selection_parent'])/'TRAINING_LOCK.json')]
 common.save(out/'SEAL.json',{'config':config,'code':code(config),'sources':files,'cases':cases,'unix':time.time(),'witness_sha256':common.sha(out/'CPU_WITNESS.json'),'automatic_manuscript_update':False})
def training_lock(config):
 files={};out=Path(config['output'])
 for p in config['protocols']:
  for f in config['folds']:
   for seed in config['seeds']:
    for stage,status in [('calibration','COMPLETE_CALIBRATION'),('refit','COMPLETE_REFIT')]:
     folder=common.case_dir(config,stage,p,f,MODEL,seed);receipt=worker._receipt(folder,status);files[str(folder/'COMPLETE.json')]=common.sha(folder/'COMPLETE.json')
 common.save(out/'TRAINING_LOCK.json',{'status':'FROZEN_TRAINING','eligible_models':[MODEL],'seal_sha256':common.sha(out/'SEAL.json'),'files':files,'unix':time.time()})
def score(config):
 out=Path(config['output']);assert not (out/'SCORING_STARTED.json').exists();common.check_training_lock(config);check(config);files={}
 old=common.read(Path(config['selection_parent'])/'PREDICTION_LOCK.json')
 for p in config['protocols']:
  for f in config['folds']:
   folder=out/'fusion_predictions'/p/f;folder.mkdir(parents=True,exist_ok=True)
   xp=Path(config['selection_parent'])/'predictions'/p/f/'XGB.npy';assert common.sha(xp)==old['files'][str(xp)];x=np.load(xp)
   (folder/'XGB.npy').write_bytes(xp.read_bytes());files[str(folder/'XGB.npy')]=common.sha(folder/'XGB.npy')
   for seed in config['seeds']:
    folder=common.case_dir(config,'prediction',p,f,MODEL,seed);worker._receipt(folder,'COMPLETE_PREDICTION');files[str(folder/'test_ema.npy')]=common.sha(folder/'test_ema.npy')
    b=np.load(folder/'test_ema.npy');fp=out/'fusion_predictions'/p/f/f'fusion_seed{seed}.npy';np.save(fp,anchor.fused(x,b,config));files[str(fp)]=common.sha(fp)
 common.save(out/'PREDICTION_LOCK.json',{'files':files,'unix':time.time(),'training_lock_sha256':common.sha(out/'TRAINING_LOCK.json'),'original_prediction_lock_sha256':common.sha(Path(config['selection_parent'])/'PREDICTION_LOCK.json'),'test_labels_read':False})
 for path,h in files.items():assert common.sha(path)==h
 common.save(out/'SCORING_STARTED.json',{'unix':time.time()});records=[]
 for p in config['protocols']:
  for f in config['folds']:
   root,receipt=common._formal_receipt(config,p,f);yp=root/'TEST_TARGETS.jsonl';assert common.sha(yp)==receipt['outputs']['TEST_TARGETS.jsonl'];y=np.array([json.loads(s)['label']-1 for s in yp.read_text().splitlines()]);counts=np.asarray(receipt['fit_class_counts']);weights=np.sqrt(counts.sum()/(7*counts))
   xp=out/'fusion_predictions'/p/f/'XGB.npy';assert common.sha(xp)==files[str(xp)];x=np.load(xp)
   for seed in config['seeds']:
    path=common.case_dir(config,'prediction',p,f,MODEL,seed)/'test_ema.npy';assert common.sha(path)==files[str(path)];b=np.load(path);fp=out/'fusion_predictions'/p/f/f'fusion_seed{seed}.npy';assert common.sha(fp)==files[str(fp)];fusion=np.load(fp)
    records.append({'protocol':p,'fold':f,'seed':seed,'neural':parent.metric(b,y,weights)[0],'tree':parent.metric(x,y,weights)[0],'fusion':parent.metric(fusion,y,weights)[0]})
 summary={}
 for p in config['protocols']:
  summary[p]={}
  for key in ['neural','tree','fusion']:
   folds=[float(np.mean([row[key]['macro_f1_percent'] for row in records if row['protocol']==p and row['fold']==f])) for f in config['folds']]
   summary[p][key]={'fold_means':folds,'mean':float(np.mean(folds)),'sample_sd':float(np.std(folds,ddof=1))}
 common.save(out/'RESULTS.json',{'status':'COMPLETE','records':records,'summary':summary,'role':'matched selection comparison, not a replacement selected by Test','automatic_manuscript_update':False})
def execute(config):
 out=Path(config['output']);handle=(out/'QUEUE.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (out/'QUEUE_STARTED.json').exists();common.save(out/'QUEUE_STARTED.json',{'pid':os.getpid(),'unix':time.time()});jobs=[]
 for p in config['protocols']:
  for f in config['folds']:
   for seed in config['seeds']:
    for stage in ['calibration','refit']:jobs.append({'id':f'{p}_{f}_{seed}_{stage}','protocol':p,'fold':f,'seed':seed,'stage':stage,'status':'pending'})
 running={};start=time.time()
 try:
  while any(j['status']!='completed' for j in jobs):
   assert time.time()-start<config['fair_queue_seconds']
   for ident,(proc,stream,j) in list(running.items()):
    if proc.poll() is None:
     assert time.time()-j['started_unix']<config['budget']['worker_seconds']+120;continue
    stream.close();j['status']='completed' if proc.returncode==0 else 'failed';j['ended_unix']=time.time();del running[ident];assert proc.returncode==0,j['id']
    folder=common.case_dir(config,j['stage'],j['protocol'],j['fold'],MODEL,j['seed']);worker._receipt(folder,'COMPLETE_CALIBRATION' if j['stage']=='calibration' else 'COMPLETE_REFIT')
   for j in jobs:
    if j['status']!='pending' or len(running)>=2:continue
    if j['stage']=='refit' and not any(q['stage']=='calibration' and q['protocol']==j['protocol'] and q['fold']==j['fold'] and q['seed']==j['seed'] and q['status']=='completed' for q in jobs):continue
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free<11*1024:continue
    slot=next(i for i in [0,1] if i not in [v[2]['slot'] for v in running.values()]);log=out/'logs'/(j['id']+'.log');log.parent.mkdir(exist_ok=True);stream=log.open('w')
    args=[sys.executable,'-s','-u',__file__,'--mode',j['stage'],'--protocol',j['protocol'],'--fold',j['fold'],'--seed',str(j['seed']),'--slot',str(slot)];proc=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    j.update(status='running',pid=proc.pid,slot=slot,log=str(log),started_unix=time.time());running[j['id']]=(proc,stream,j);time.sleep(2)
   common.save(out/'QUEUE_STATE.json',{'status':'RUNNING','jobs':jobs,'unix':time.time()});time.sleep(5)
  training_lock(config)
  for p in config['protocols']:
   for f in config['folds']:
    for seed in config['seeds']:worker.predict(config,p,f,MODEL,seed,0)
  score(config);common.save(out/'QUEUE_STATE.json',{'status':'COMPLETE','jobs':jobs,'unix':time.time()})
 except BaseException as e:
  common.save(out/'QUEUE_STATE.json',{'status':'STOPPED_ERROR','error':str(e),'traceback':traceback.format_exc(),'jobs':jobs,'unix':time.time()})
  for proc,stream,j in running.values():proc.wait();stream.close()
  raise
def guardian(config):
 out=Path(config['output']);h=(out/'DEPENDENCY.lock').open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);began=time.time()
 while True:
  state=common.read(Path(config['wait_for'])/'QUEUE_STATE.json');status=state['status']
  if status=='COMPLETE':break
  if status=='STOPPED_ERROR':common.save(out/'QUEUE_STATE.json',{'status':'HELD_DEPENDENCY_ERROR','dependency':config['wait_for'],'unix':time.time()});return
  common.save(out/'QUEUE_STATE.json',{'status':'WAITING_FOR_ABLATIONS','dependency_status':status,'unix':time.time()});assert time.time()-began<config['dependency_wait_seconds'];time.sleep(60)
 check(config)
 log=(out/'gpu_witness.log').open('w')
 proc=subprocess.Popen([sys.executable,'-s','-u',__file__,'--mode','gpu_witness'],stdout=log,stderr=subprocess.STDOUT);rc=proc.wait();log.close();assert rc==0,'GPU witness failed'
 assert common.read(out/'GPU_WITNESS.json')['code']==common.read(out/'SEAL.json')['code']
 execute(config)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['witness','gpu_witness','seal','launch','guardian','calibration','refit'],required=True);ap.add_argument('--protocol');ap.add_argument('--fold');ap.add_argument('--seed',type=int);ap.add_argument('--slot',type=int,default=0);a=ap.parse_args();config=setup();out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
 if a.mode=='witness':witness(config)
 elif a.mode=='gpu_witness':witness(config,'cuda')
 elif a.mode=='seal':seal(config)
 elif a.mode=='launch':
  check(config);assert not (out/'DISPATCH.json').exists();log=(out/'guardian.log').open('w');p=subprocess.Popen([sys.executable,'-s','-u',__file__,'--mode','guardian'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);common.save(out/'DISPATCH.json',{'pid':p.pid,'unix':time.time()});print('GUARDIAN_PID',p.pid)
 elif a.mode=='guardian':guardian(config)
 elif a.mode=='calibration':worker.calibrate(config,a.protocol,a.fold,MODEL,a.seed,a.slot)
 elif a.mode=='refit':worker.refit(config,a.protocol,a.fold,MODEL,a.seed,a.slot)
if __name__=='__main__':main()
