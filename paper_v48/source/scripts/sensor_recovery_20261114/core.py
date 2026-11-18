"""Consistent DS-selection recovery arms, reusing frozen model/data interventions."""
from pathlib import Path
import argparse,copy,gc,json,os,sys,time
import numpy as np
import torch
import xgboost as xgb
from scripts.sensor_fair_selection_20261113 import main as fair
from scripts.sensor_ablation_20261113 import data as dataops
from scripts.sensor_ablation_20261113.models import AblationModel

ROOT=fair.ROOT;OUT=ROOT/'reports/sensor_recovery_20261114'
ARM=os.environ.get('RECOVERY_ARM','no_eri')
ORIGINAL_SYNTHETIC=fair.parent.old.synthetic_batch
ORIGINAL_TRAINING_LOCK=fair.training_lock
def sha(p):return fair.common.sha(p)
def read(p):return fair.common.read(p)
def save(p,x):return fair.common.save(p,x)
def code(cfg):
 paths=set(read(ROOT/'reports/sensor_ablation_20261113/MATRIX_LOCK.json')['code'])|set(read(ROOT/'reports/sensor_fair_selection_20261113/SEAL.json')['code'])
 paths.update(str(p.relative_to(ROOT)) for p in (ROOT/'scripts/sensor_recovery_20261114').glob('*.py'))
 paths.update(['configs/sensor_recovery_20261114.json',str(fair.CONFIG)])
 paths.update(str(p.relative_to(ROOT)) for p in (ROOT/'configs').glob('sensor_recovery_20261114_*.json'))
 return {str(Path(p).resolve().relative_to(ROOT)):sha(p) for p in sorted(paths)}
def check(cfg):
 s=read(Path(cfg['output'])/'SEAL.json');assert s['config']==cfg and s['code']==code(cfg)
 for p,h in s['sources'].items():assert sha(p)==h,p
 return s
def case(cfg,p,f):return read(Path(cfg['output'])/'INPUT_LOCK.json')['cases'][p+'/'+f]
def load(cfg,p,f,stage,device='cuda'):
 m=case(cfg,p,f);fp=Path(m['cal_fit' if stage=='calibration' else 'fit']);assert sha(fp)==m['hashes'][str(fp)]
 with np.load(fp) as z:y=z['labels'].copy();values={k:z[k].copy() for k in fair.common.FIELDS}
 counts=np.bincount(y,minlength=7);assert (counts>0).all();n=len(y);truth=None
 if stage in ['calibration','test']:
  ep=Path(m['dev' if stage=='calibration' else 'test']);assert sha(ep)==m['hashes'][str(ep)]
  with np.load(ep) as z:
   assert 'labels' not in z.files;ev={k:z[k].copy() for k in fair.common.FIELDS}
  if stage=='calibration':
   tp=Path(m['dev_targets']);assert sha(tp)==m['hashes'][str(tp)];truth=np.array([json.loads(s)['label']-1 for s in tp.read_text().splitlines()]);values={k:np.concatenate([v,ev[k]]) for k,v in values.items()};labels=np.r_[y,np.full(len(truth),-100)];train=np.arange(n);evaluation=np.arange(n,n+len(truth))
  else:
   values=ev;labels=np.full(len(ev['dynamic']),-100);train=np.array([],dtype=np.int64);evaluation=np.arange(len(labels))
 else:labels=y;train=np.arange(n);evaluation=np.array([],dtype=np.int64)
 tensors={k:torch.as_tensor(v,device=device) for k,v in values.items()};tensors['labels']=torch.as_tensor(labels,device=device,dtype=torch.long);dataops.transform_tensors(tensors,cfg['active_arm'])
 result={'tensors':tensors,'prior':counts/counts.sum(),'weights':torch.tensor(np.sqrt(counts.sum()/(7*counts)),device=device,dtype=torch.float32),'train':train,'evaluation':evaluation,'fit_truth':y}
 if truth is not None:result['dev_truth']=truth
 return result
def model(cfg,arm,prior,device):
 assert arm==cfg['active_arm'];m=AblationModel(read(cfg['architecture_config']),prior,arm).to(device)
 for x in m.modules():
  if isinstance(x,torch.nn.modules.batchnorm._BatchNorm):x.momentum=cfg['BN_momentum']
 return m
def synthetic(device,n):
 b=ORIGINAL_SYNTHETIC(device,n)
 if ARM=='long_60_60':b['dynamic']=torch.randn(n,600,8,device=device);b['dynamic'][:,:,6]=1;b['dynamic'][:,:,7]=.1
 if ARM=='short_10_10':b['state']=b['state'][:,-10:].clone();b['state_valid']=b['state_valid'][:,-10:].clone();b['current']=b['state'][:,-1].clone()
 return dataops.transform_tensors(b,ARM)
def setup(arm):
 global ARM
 ARM=arm;os.environ['RECOVERY_ARM']=arm;fair.MODEL=arm;fair.CONFIG=Path('configs/sensor_recovery_20261114_'+arm+'.json');fair.__file__=__file__;fair.model=model;fair.check=check;fair.code=code
 cfg=fair.setup();fair.parent.old.synthetic_batch=synthetic
 fair.common.load_selection=lambda c,p,f,device='cuda':load(c,p,f,'calibration',device)
 fair.common.load_refit=lambda c,p,f,device='cuda':load(c,p,f,'refit',device)
 def test(c,p,f,device='cuda'):
  fair.common.check_training_lock(c);return load(c,p,f,'test',device)
 fair.common.load_test=test;fair.training_lock=training_lock;fair.score=score
 return cfg
def lock_inputs(cfg):
 out=Path(cfg['output']);assert not (out/'INPUT_LOCK.json').exists();cases={};sources={}
 for p in cfg['protocols']:
  for f in cfg['folds']:
   folder=dataops.data_dir(ARM,p,f);receipt=read(folder/'RECEIPT.json');ref=read(folder/'REFIT_SOURCE.json')
   for name in ['RECEIPT.json','REFIT_SOURCE.json']:sources[str(folder/name)]=sha(folder/name)
   for name,h in receipt['outputs'].items():assert sha(folder/name)==h
   m={'cal_fit':str(folder/'FIT.npz'),'dev':str(folder/'DEV_INPUTS.npz'),'dev_targets':str(folder/'DEV_TARGETS.jsonl'),'fit':str(Path(ref['FIT_path']).resolve()),'test':str(Path(ref['TEST_INPUTS_path']).resolve()),'test_targets':str(Path(ref['TEST_TARGETS_path']).resolve())}
   m['hashes']={str(folder/name):receipt['outputs'][name] for name in ['FIT.npz','DEV_INPUTS.npz','DEV_TARGETS.jsonl']};m['hashes'].update({m['fit']:ref['FIT_sha256'],m['test']:ref['TEST_INPUTS_sha256'],m['test_targets']:ref['TEST_TARGETS_sha256']})
   for k in ['cal_fit','dev','fit','test']:assert sha(m[k])==m['hashes'][m[k]]
   with np.load(m['fit']) as z:m['class_counts']=np.bincount(z['labels'],minlength=7).tolist()
   cases[p+'/'+f]=m
 save(out/'INPUT_LOCK.json',{'cases':cases,'metadata':sources,'unix':time.time()})
def seal(cfg):
 out=Path(cfg['output']);assert not (out/'SEAL.json').exists();proof=read(out/'CPU_WITNESS.json');assert proof['code']==code(cfg)
 lock=read(out/'INPUT_LOCK.json');sources=dict(lock['metadata']);sources[str(out/'INPUT_LOCK.json')]=sha(out/'INPUT_LOCK.json')
 for name in ['PREDICTION_LOCK.json','TRAINING_LOCK.json']:
  p=ROOT/'reports/sensor_converged_fivefold_20261027_v1'/name;sources[str(p)]=sha(p)
 save(out/'SEAL.json',{'config':cfg,'code':code(cfg),'sources':sources,'unix':time.time()})
def trees(cfg):
 if ARM=='no_eri':return
 check(cfg);params=dict(objective='multi:softprob',num_class=7,tree_method='hist',device='cpu',nthread=4,eta=.05,max_depth=6,gamma=1.,min_child_weight=10.,reg_lambda=10.,subsample=1.,colsample_bytree=1.,seed=17,disable_default_eval_metric=1)
 for p in cfg['protocols']:
  for f in cfg['folds']:
   out=Path(cfg['output'])/'trees'/p/f;out.mkdir(parents=True,exist_ok=True);assert not (out/'STARTED.json').exists();save(out/'STARTED.json',{'unix':time.time()});m=case(cfg,p,f);assert sha(m['fit'])==m['hashes'][m['fit']]
   with np.load(m['fit']) as z:x=dataops.transform_features(z['features'].copy(),ARM);y=z['labels'].copy()
   cnt=np.bincount(y,minlength=7);w=np.sqrt(cnt.sum()/(7*cnt));dm=xgb.DMatrix(x,label=y,weight=w[y],nthread=4);b=xgb.train(params,dm,num_boost_round=500);b.save_model(out/'model.ubj')
   save(out/'COMPLETE.json',{'status':'FIXED500_COMPLETE','rounds':500,'model_sha256':sha(out/'model.ubj'),'fit_sha256':m['hashes'][m['fit']],'convergence_claimed':False})
def training_lock(cfg):
 ORIGINAL_TRAINING_LOCK(cfg);out=Path(cfg['output']);lock=read(out/'TRAINING_LOCK.json')
 if ARM!='no_eri':
  for p in cfg['protocols']:
   for f in cfg['folds']:
    q=out/'trees'/p/f/'COMPLETE.json';z=read(q);assert z['status']=='FIXED500_COMPLETE' and sha(q.parent/'model.ubj')==z['model_sha256'];lock['files'][str(q)]=sha(q)
 save(out/'TRAINING_LOCK.json',lock)
def score(cfg):
 check(cfg);fair.common.check_training_lock(cfg);out=Path(cfg['output']);assert not (out/'SCORING_STARTED.json').exists();files={};old=read(ROOT/'reports/sensor_converged_fivefold_20261027_v1/PREDICTION_LOCK.json')
 for p in cfg['protocols']:
  for f in cfg['folds']:
   folder=out/'fusion_predictions'/p/f;folder.mkdir(parents=True,exist_ok=True);m=case(cfg,p,f)
   if ARM=='no_eri':
    path=Path('reports/sensor_converged_fivefold_20261027_v1/predictions')/p/f/'XGB.npy';assert sha(path)==old['files'][str(path)];x=np.load(path)
   else:
    tree=out/'trees'/p/f;b=xgb.Booster();assert sha(tree/'model.ubj')==read(tree/'COMPLETE.json')['model_sha256'];b.load_model(tree/'model.ubj');assert sha(m['test'])==m['hashes'][m['test']]
    with np.load(m['test']) as z:assert 'labels' not in z.files;x=b.predict(xgb.DMatrix(dataops.transform_features(z['features'].copy(),ARM),nthread=4))
   np.save(folder/'XGB.npy',x);files[str(folder/'XGB.npy')]=sha(folder/'XGB.npy')
   for seed in cfg['seeds']:
    path=fair.common.case_dir(cfg,'prediction',p,f,ARM,seed)/'test_ema.npy';fair.worker._receipt(path.parent,'COMPLETE_PREDICTION');files[str(path)]=sha(path);fp=folder/f'fusion_seed{seed}.npy';np.save(fp,fair.anchor.fused(x,np.load(path),cfg));files[str(fp)]=sha(fp)
 save(out/'PREDICTION_LOCK.json',{'files':files,'training_lock_sha256':sha(out/'TRAINING_LOCK.json'),'unix':time.time(),'test_labels_parsed':False});save(out/'SCORING_STARTED.json',{'unix':time.time()});records=[]
 for p in cfg['protocols']:
  for f in cfg['folds']:
   m=case(cfg,p,f);assert sha(m['test_targets'])==m['hashes'][m['test_targets']];y=np.array([json.loads(s)['label']-1 for s in Path(m['test_targets']).read_text().splitlines()]);cnt=np.array(m['class_counts']);w=np.sqrt(cnt.sum()/(7*cnt));folder=out/'fusion_predictions'/p/f
   for seed in cfg['seeds']:
    paths={'tree':folder/'XGB.npy','fusion':folder/f'fusion_seed{seed}.npy','neural':fair.common.case_dir(cfg,'prediction',p,f,ARM,seed)/'test_ema.npy'}
    values={}
    for k,path in paths.items():assert sha(path)==files[str(path)];values[k]=fair.parent.metric(np.load(path),y,w)[0]
    records.append({'protocol':p,'fold':f,'seed':seed,**values})
 summary={p:{k:{'mean':float(np.mean([z[k]['macro_f1_percent'] for z in records if z['protocol']==p])),'fold_sd':float(np.std([np.mean([z[k]['macro_f1_percent'] for z in records if z['protocol']==p and z['fold']==f]) for f in cfg['folds']],ddof=1))} for k in ['tree','neural','fusion']} for p in cfg['protocols']}
 save(out/'RESULTS.json',{'status':'COMPLETE','records':records,'summary':summary,'selection_protocol':'DS minimum DEV CE/same_epochs','old_failed_protocol_results_mixed':False})
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',default=os.environ.get('RECOVERY_ARM','no_eri'));ap.add_argument('--mode',choices=['prepare','witness','gpu_witness','seal','trees','execute','calibration','refit'],required=True);ap.add_argument('--protocol');ap.add_argument('--fold');ap.add_argument('--seed',type=int);ap.add_argument('--slot',type=int,default=0);a=ap.parse_args();cfg=setup(a.arm);Path(cfg['output']).mkdir(parents=True,exist_ok=True)
 if a.mode=='prepare':lock_inputs(cfg)
 elif a.mode=='witness':fair.witness(cfg)
 elif a.mode=='gpu_witness':fair.witness(cfg,'cuda')
 elif a.mode=='seal':seal(cfg)
 elif a.mode=='trees':trees(cfg)
 elif a.mode=='execute':fair.execute(cfg)
 elif a.mode=='calibration':fair.worker.calibrate(cfg,a.protocol,a.fold,ARM,a.seed,a.slot)
 elif a.mode=='refit':fair.worker.refit(cfg,a.protocol,a.fold,ARM,a.seed,a.slot)
if __name__=='__main__':main()
