"""Versioned data bug correction, retaining input-equivalent completed fits."""
from pathlib import Path
import argparse,copy,gc,json,os,subprocess,sys,time,traceback
import numpy as np
import torch
from scripts.sensor_long60_resource_retry_20261114 import main as base

ROOT=base.ROOT;OLD=base.OUT;OUT=ROOT/'reports/sensor_long60_scale1_corrected_20261114';SOURCE='scripts/sensor_scale_normalization_fix_20261114/resume.py'
OLD_CODE=base.code
def code(cfg):
 return {**base.ORIGINAL_CODE(cfg),'scripts/sensor_long60_resource_retry_20261114/main.py':base.c.sha(ROOT/'scripts/sensor_long60_resource_retry_20261114/main.py'),SOURCE:base.c.sha(ROOT/SOURCE)}
def check(cfg):
 s=base.c.read(OUT/'SEAL.json');assert s['config']==cfg and s['code']==code(cfg)
 for p,h in s['sources'].items():assert base.c.sha(p)==h,p
 return s
def prepare(cfg):
 c=base.c;assert not (OUT/'SEAL.json').exists();OUT.mkdir(parents=True,exist_ok=True)
 old=c.read(OLD/'SEAL.json');oldcfg=dict(cfg,output=str(OLD.relative_to(ROOT)));assert old['config']==oldcfg
 for p,h in old['code'].items():assert c.sha(p)==h
 for p,h in old['sources'].items():assert c.sha(p)==h
 inputs=c.read(OLD/'INPUT_LOCK.json');m=inputs['cases']['TS/fold_3'];bad=Path(m['dev']);assert c.sha(bad)==m['hashes'][str(bad)];canonical=ROOT/'reports/sensor_converged_fivefold_20261027_v1/data/TS/fold_3';receipt=c.read(canonical/'RECEIPT.json');ref=canonical/'DEV_INPUTS.npz';assert c.sha(ref)==receipt['outputs']['DEV_INPUTS.npz']
 # Long-window state/context must be identical to the original60s state context.
 bad_targets=ROOT/'reports/sensor_ablation_20261113/data/long_60_60/TS/fold_3/DEV_TARGETS.jsonl'
 keys=lambda p:[(r['cow_id'],r['timestamp'],r['label']) for r in map(json.loads,Path(p).read_text().splitlines())]
 assert keys(bad_targets)==keys(canonical/'DEV_TARGETS.jsonl')
 with np.load(bad) as x,np.load(ref) as y:
  for k in ['state','state_valid','current']:np.testing.assert_array_equal(x[k],y[k])
  values={k:x[k].copy() for k in x.files};ctx=y['context'].copy();oldmax=float(abs(values['context']).max());values['context']=ctx;values['features'][:,:136]=ctx
 fixed=OUT/'corrected_inputs/TS_fold3_DEV_INPUTS.npz';fixed.parent.mkdir(parents=True,exist_ok=True);np.savez(fixed,**values);del values
 m['dev']=str(fixed);m['hashes'][str(fixed)]=c.sha(fixed)
 (OUT/'INPUT_LOCK.json').write_text(json.dumps(inputs,indent=2))
 sources={**old['sources'],str(OLD/'SEAL.json'):c.sha(OLD/'SEAL.json'),str(OLD/'INPUT_LOCK.json'):c.sha(OLD/'INPUT_LOCK.json'),str(OUT/'INPUT_LOCK.json'):c.sha(OUT/'INPUT_LOCK.json'),str(ref):c.sha(ref),str(canonical/'RECEIPT.json'):c.sha(canonical/'RECEIPT.json')}
 # Verify all10 long-window calibration contexts against original v2 policy.
 equality=[]
 for p in cfg['protocols']:
  for f in cfg['folds']:
   z=inputs['cases'][p+'/'+f];norm=ROOT/'reports/sensor_ablation_20261113/data/long_60_60'/p/f/'NORMALIZERS.json';doc=c.read(norm);s=np.array(doc['context']['std']);scale=np.where(s<=1e-6,1.,s);sources[str(norm)]=c.sha(norm)
   for name,key in [('FIT.npz','cal_fit'),('DEV_INPUTS.npz','dev')]:
    original=Path(c.read(OLD/'INPUT_LOCK.json')['cases'][p+'/'+f][key]);b=ROOT/'reports/sensor_converged_fivefold_20261027_v1/data'/p/f/name;br=c.read(b.parent/'RECEIPT.json');assert c.sha(b)==br['outputs'][name]
    with np.load(original) as ox,np.load(b) as bx:
     corrected=(ox['context'].astype(float)*(s/scale)).astype(np.float32);np.testing.assert_array_equal(corrected,bx['context']);same=bool(np.array_equal(ox['context'],corrected));equality.append({'protocol':p,'fold':f,'file':name,'unchanged':same})
     assert same or (p=='TS' and f=='fold_3' and name=='DEV_INPUTS.npz')
    sources[str(b)]=c.sha(b)
 c.save(OUT/'INPUT_EQUIVALENCE.json',{'records':equality,'affected_case':'TS/fold_3','old_DEV_context_absmax':oldmax,'corrected_DEV_context_absmax':float(abs(ctx).max()),'policy':'TRAIN near-constant std<=1e-6 uses scale1, original v2 policy','no_Test_scores_used':True})
 reused=[]
 for f in ['fold_1','fold_2']:
  for seed in cfg['seeds']:
   for stage,status in [('calibration','COMPLETE_CALIBRATION'),('refit','COMPLETE_REFIT')]:
    src=OLD/stage/'TS'/f/'long_60_60'/f'seed{seed}';done=c.read(src/'COMPLETE.json');assert done['status']==status
    for p,h in done['files'].items():assert c.sha(p)==h
    dst=OUT/stage/'TS'/f/'long_60_60'/f'seed{seed}';dst.mkdir(parents=True,exist_ok=True)
    for p in src.iterdir():
     if p.is_file() and not (dst/p.name).exists():os.link(p,dst/p.name)
     if p.is_file():assert c.sha(dst/p.name)==c.sha(p);sources[str(dst/p.name)]=c.sha(dst/p.name)
    sources[str(src/'COMPLETE.json')]=c.sha(src/'COMPLETE.json');reused.append(str(dst/'COMPLETE.json'))
 for p in cfg['protocols']:
  for f in cfg['folds']:
   src=OLD/'trees'/p/f;dst=OUT/'trees'/p/f;dst.mkdir(parents=True,exist_ok=True)
   for n in ['model.ubj','COMPLETE.json']:
    if not (dst/n).exists():os.link(src/n,dst/n)
    assert c.sha(dst/n)==c.sha(src/n);sources[str(dst/n)]=c.sha(dst/n)
 c.save(OUT/'SEAL.json',{'config':cfg,'code':code(cfg),'sources':sources,'data_correction':'restore original constant-feature scaling, entire long-window policy verified; TSfold3 DEV only changes','reused_completed_fit_files':reused,'resource_changes':{'gpu_workers':1,'memory_fraction':.85},'Test_based_selection':False,'unix':time.time()})
def execute(cfg):
 c=base.c;h=(OUT/'QUEUE.lock').open('a');base.fcntl.flock(h,base.fcntl.LOCK_EX|base.fcntl.LOCK_NB);assert not (OUT/'QUEUE_STARTED.json').exists();check(cfg);seal=c.read(OUT/'SEAL.json');allowed=set(seal['reused_completed_fit_files']);jobs=[]
 for p in cfg['protocols']:
  for f in cfg['folds']:
   for seed in cfg['seeds']:
    for mode in ['calibration','refit']:
     cp=c.fair.common.case_dir(cfg,mode,p,f,'long_60_60',seed)/'COMPLETE.json';reused=str(cp.resolve()) in allowed
     if reused:c.fair.worker._receipt(cp.parent,'COMPLETE_CALIBRATION' if mode=='calibration' else 'COMPLETE_REFIT')
     jobs.append({'protocol':p,'fold':f,'seed':seed,'mode':mode,'status':'completed' if reused else 'pending','reused_input_equivalent_fit':reused})
 c.save(OUT/'QUEUE_STARTED.json',{'pid':os.getpid(),'unix':time.time()})
 try:
  for j in jobs:
   if j['status']=='completed':continue
   log=OUT/'logs'/f'{j["protocol"]}_{j["fold"]}_{j["seed"]}_{j["mode"]}.log';log.parent.mkdir(exist_ok=True)
   with log.open('w') as stream:
    proc=subprocess.Popen([sys.executable,'-s','-u',__file__,'--mode',j['mode'],'--protocol',j['protocol'],'--fold',j['fold'],'--seed',str(j['seed'])],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);j.update(status='running',pid=proc.pid,log=str(log),started_unix=time.time());c.save(OUT/'QUEUE_STATE.json',{'status':'RUNNING','jobs':jobs,'unix':time.time()})
    try:rc=proc.wait(timeout=cfg['budget']['worker_seconds']+120)
    except subprocess.TimeoutExpired:
     proc.terminate()
     try:proc.wait(timeout=30)
     except subprocess.TimeoutExpired:proc.kill();proc.wait()
     raise
   j.update(status='completed' if rc==0 else 'failed',ended_unix=time.time());assert rc==0,str(log);c.fair.worker._receipt(c.fair.common.case_dir(cfg,j['mode'],j['protocol'],j['fold'],'long_60_60',j['seed']),'COMPLETE_CALIBRATION' if j['mode']=='calibration' else 'COMPLETE_REFIT')
  c.training_lock(cfg)
  for p in cfg['protocols']:
   for f in cfg['folds']:
    for seed in cfg['seeds']:c.fair.worker.predict(cfg,p,f,'long_60_60',seed,0);gc.collect();torch.cuda.empty_cache()
  c.score(cfg);c.save(OUT/'QUEUE_STATE.json',{'status':'COMPLETE','jobs':jobs,'unix':time.time()})
 except BaseException as e:c.save(OUT/'QUEUE_STATE.json',{'status':'STOPPED_ERROR','jobs':jobs,'error':str(e),'traceback':traceback.format_exc(),'unix':time.time()});raise
base.OUT=OUT;base.__file__=__file__;base.code=code;base.check=check;base.prepare=prepare;base.execute=execute
if __name__=='__main__':base.main()
