"""Adapters around the frozen calibration/refit engine, without changing old code."""
from pathlib import Path
import argparse,copy,json,os,shutil,time,gc,hashlib,subprocess
import numpy as np
import torch
import xgboost as xgb
from scripts import run_converged_fivefold_20261027 as engine
from scripts import converged_fivefold_common_20261027 as c
from scripts.sensor_ablation_20261113 import data as d
from scripts.sensor_ablation_20261113.models import AblationModel,ARMS
from scripts.sensor_uwb_event_motion_20261020 import ParameterEMA

BASE_CONFIG='configs/sensor_converged_fivefold_20261027_v1.json'
ACTIVE_ARM=None
ORIG_LOAD_SELECTION=c.load_selection;ORIG_LOAD_REFIT=c.load_refit
ORIG_COMPLETED=engine.completed
def cfg_for(arm):
 cfg=c.read(BASE_CONFIG);cfg['output']=str((d.OUT/'runs'/arm).relative_to(d.ROOT));cfg['ablation_arm']=arm
 cfg['shared_GPU']={'Torch_memory_fraction':.45,'minimum_free_GiB_before_allocation':11}
 cfg['budget']={'queue_seconds':96*3600,'worker_seconds':12*3600}
 return cfg
def source_hashes(cfg):
 old=c.read(d.BASE/'TRAINING_SEAL.json')['code']
 for p,h in old.items():assert c.sha(p)==h,'Original source changed: '+p
 new={str(p.relative_to(d.ROOT)):c.sha(p) for p in (d.ROOT/'scripts/sensor_ablation_20261113').glob('*.py')}
 return {**old,**new,BASE_CONFIG:c.sha(BASE_CONFIG),'configs/sensor_ablation_20261113_env.json':c.sha('configs/sensor_ablation_20261113_env.json')}
def check_seal(cfg):
 seal=c.read(d.OUT/'MATRIX_LOCK.json');assert seal['code']==source_hashes(cfg)
 assert cfg['ablation_arm'] in seal['arms']
 assert cfg==seal['arm_configs'][cfg['ablation_arm']]
 for p,h in seal['source_inputs'].items():assert c.sha(p)==h,p
 lock=c.read(Path(cfg['output'])/'DATA_LOCK.json')
 assert lock['matrix_sha256']==c.sha(d.OUT/'MATRIX_LOCK.json')
 for p,h in lock['data'].items():assert c.sha(p)==h,p
 return seal
def checked_completed(folder):
 folder=Path(folder)
 if 'no_eri' in folder.parts and folder.parts[-3] in ['calibration_tree','final_tree']:
  done=c.read(folder/'COMPLETE.json');assert done.get('reconstructed_original_tree_sha256_exact') or done.get('reuse_original_tree')
  stage,p,f=folder.parts[-3:];old=c.read(d.BASE/stage/p/f/'COMPLETE.json')
  for name in ['FIT.npy']+(['DEV.npy'] if stage=='calibration_tree' else []):assert c.sha(folder/name)==old['files'][name]
 return ORIG_COMPLETED(folder)
def new_state(cfg,data,seed,lr=None):
 c.native.core.seed_all(seed);arch=c.read(cfg['architecture_config']);model=AblationModel(arch,data['prior'],cfg['ablation_arm']).to(data['tensors']['dynamic'].device)
 opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'] if lr is None else lr,weight_decay=cfg['weight_decay'],eps=1e-8)
 ema=ParameterEMA(model,cfg['ema_decay'],cfg['ema_warmup_steps'])
 for net in [model,ema.model]:
  for module in net.modules():
   if isinstance(module,torch.nn.BatchNorm2d):module.momentum=cfg['BN_momentum']
 return {'model':model,'opt':opt,'ema':ema,'step':0,'arm':'Constant_3e4','scheduler':torch.optim.lr_scheduler.ReduceLROnPlateau(opt,**cfg['scheduler'])}
def selection(cfg,p,f,device='cuda'):
 data=ORIG_LOAD_SELECTION(cfg,p,f,device);d.transform_tensors(data['tensors'],cfg['ablation_arm']);return data
def refit(cfg,p,f,device='cuda'):
 data=ORIG_LOAD_REFIT(cfg,p,f,device);d.transform_tensors(data['tensors'],cfg['ablation_arm']);return data
def tree_arrays(cfg,p,f,is_refit):
 folder=d.data_dir(cfg['ablation_arm'],p,f);src=c.read(folder/'REFIT_SOURCE.json') if is_refit else None
 path=Path(src['FIT_path']) if is_refit else folder/'FIT.npz'
 if src:assert c.sha(path)==src['FIT_sha256']
 else:assert c.sha(path)==c.read(folder/'RECEIPT.json')['outputs']['FIT.npz']
 with np.load(path) as z:features=d.transform_features(z['features'].copy(),cfg['ablation_arm']);y=z['labels'].copy()
 counts=np.bincount(y,minlength=7);weights=np.sqrt(counts.sum()/(7*counts));fit=xgb.DMatrix(features,label=y,weight=weights[y],nthread=4)
 if is_refit:return fit,y,weights,None,None
 receipt=c.read(folder/'RECEIPT.json')
 for name in ['DEV_INPUTS.npz','DEV_TARGETS.jsonl']:assert c.sha(folder/name)==receipt['outputs'][name]
 with np.load(folder/'DEV_INPUTS.npz') as z:dev=xgb.DMatrix(d.transform_features(z['features'].copy(),cfg['ablation_arm']),nthread=4)
 truth=np.array([json.loads(s)['label']-1 for s in (folder/'DEV_TARGETS.jsonl').read_text().splitlines()])
 return fit,y,weights,dev,truth
def install(arm):
 c.data_dir=lambda cfg,p,f:d.data_dir(cfg['ablation_arm'],p,f)
 c.check_seal=check_seal;c.source_files=source_hashes;c.new_state=new_state;c.load_selection=selection;c.load_refit=refit;engine.tree_arrays=tree_arrays;engine.completed=checked_completed
 def gpu(cfg):
  free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=11*1024,'Resource gate: leave other jobs untouched'
  torch.cuda.set_per_process_memory_fraction(.45);c.native.core.configure()
  out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True);return (out/f'GPU_WORKER_{os.getpid()}.lock').open('a')
 engine.gpu=gpu
 return cfg_for(arm)
def reuse_tree(cfg,p,f):
 assert cfg['ablation_arm']=='no_eri'
 for stage in ['calibration_tree','final_tree']:
  src=d.BASE/stage/p/f;done=c.read(src/'COMPLETE.json');out=c.case_dir(cfg,stage,p,f);out.mkdir(parents=True,exist_ok=True)
  if stage=='final_tree' and not (src/'FIT.npy').exists():
   engine.tree_worker(cfg,p,f,True)
   matches=all(c.sha(out/name)==done['files'][name] for name in ['model.ubj','FIT.npy'])
   if not matches:
    rec=c.read(out/'COMPLETE.json');rec['status']='FAILED_ORIGINAL_HASH_COMPARISON';c.save(out/'COMPLETE.json',rec);raise RuntimeError('Reconstructed tree differs from original')
   rec=c.read(out/'COMPLETE.json');rec['reconstructed_original_tree_sha256_exact']=True;c.save(out/'COMPLETE.json',rec)
   continue
  names=['FIT.npy']+(['DEV.npy'] if stage=='calibration_tree' else [])
  for name in names:
   assert c.sha(src/name)==done['files'][name]
   if not (out/name).exists():os.link(src/name,out/name)
   assert c.sha(out/name)==done['files'][name]
  c.save(out/'COMPLETE.json',{'status':'PASS','converged':True,'rounds':done['rounds'],'reuse_original_tree':True,'files':{n:c.sha(out/n) for n in names}})
def predict(cfg):
 check_seal(cfg);out=Path(cfg['output']);assert not (out/'TRAINING_LOCK.json').exists();models={}
 for p in cfg['protocols']:
  for f in cfg['folds']:
   for stage in ['calibration_tree','final_tree']:
    folder=c.case_dir(cfg,stage,p,f);engine.completed(folder);models[str(folder/'COMPLETE.json')]=c.sha(folder/'COMPLETE.json')
   for seed in cfg['seeds']:
    for stage in ['calibration_neural','final_neural']:
     folder=c.case_dir(cfg,stage,p,f,seed);engine.completed(folder);models[str(folder/'COMPLETE.json')]=c.sha(folder/'COMPLETE.json')
 c.save(out/'TRAINING_LOCK.json',{'models':models,'unix':time.time(),'test_truth_read':False});lock=engine.gpu(cfg);files={}
 for p in cfg['protocols']:
  for f in cfg['folds']:
   source=c.read(c.data_dir(cfg,p,f)/'REFIT_SOURCE.json');path=Path(source['TEST_INPUTS_path']);assert c.sha(path)==source['TEST_INPUTS_sha256']
   with np.load(path) as z:
    assert 'labels' not in z.files;values={k:torch.as_tensor(z[k],device='cuda') for k in c.FIELDS};features=d.transform_features(z['features'].copy(),cfg['ablation_arm'])
   d.transform_tensors(values,cfg['ablation_arm'])
   # Prior and class weights are from FIT targets only; no Test labels here.
   fitpath=Path(source['FIT_targets_path']);assert c.sha(fitpath)==source['FIT_targets_sha256'];fy=np.array([json.loads(s)['label']-1 for s in fitpath.read_text().splitlines()]);counts=np.bincount(fy,minlength=7)
   data={'tensors':values,'prior':counts/counts.sum(),'weights':torch.tensor(np.sqrt(counts.sum()/(7*counts)),dtype=torch.float32,device='cuda'),'evaluation':np.arange(len(features))}
   if cfg['ablation_arm']=='no_eri':
    path=d.BASE/'predictions'/p/f/'XGB.npy';old=c.read(d.BASE/'PREDICTION_LOCK.json');assert c.sha(path)==old['files'][str(path.relative_to(d.ROOT))];px=np.load(path)
   else:
    tree=xgb.Booster();tree.load_model(c.case_dir(cfg,'final_tree',p,f)/'model.ubj');px=tree.predict(xgb.DMatrix(features,nthread=4))
   folder=out/'predictions'/p/f;folder.mkdir(parents=True);np.save(folder/'XGB.npy',px)
   for seed in cfg['seeds']:
    state=new_state(cfg,data,seed);state['model'].load_state_dict(torch.load(c.case_dir(cfg,'final_neural',p,f,seed)/'evaluation.pt',map_location='cpu',weights_only=False)['model'])
    pb=c.probabilities(state['model'].eval(),data,data['evaluation']);np.save(folder/f'B_seed{seed}.npy',pb);np.save(folder/f'fusion_seed{seed}.npy',c.fused(px,pb,cfg));del state;gc.collect();torch.cuda.empty_cache()
   files.update({str(path):c.sha(path) for path in folder.glob('*.npy')});del values,data;gc.collect();torch.cuda.empty_cache()
 c.save(out/'PREDICTION_LOCK.json',{'files':files,'training_lock_sha256':c.sha(out/'TRAINING_LOCK.json'),'unix':time.time(),'test_truth_read':False});lock.close()
def witness():
 assert Path.cwd()==d.ROOT;cfg=cfg_for('full');torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
 import pandas as pd
 frames={1:{m:pd.DataFrame({'timestamp':[70.,101.],**{name:[1.,9.] for name in d.FEATURES[m]}}) for m in d.MODALITIES}}
 norms={m:{'mean':[0.]*(6 if m=='hd' else len(d.FEATURES[m])),'std':[1.]*(6 if m=='hd' else len(d.FEATURES[m]))} for m in d.MODALITIES}
 rows=[{'cow_id':1,'timestamp':100,'start':0,'end':200,'split':'train','segment':'a'}]
 short=d.state_windows(rows,frames,norms,10);long=d.state_windows(rows,frames,norms,60)
 for m in d.MODALITIES:assert not short[m]['valid'].any() and long[m]['valid'][0,-1].all()
 for arm in ['dynamic_only','state_only','no_uwb','no_ankle']:
  x=np.arange(2*866,dtype=np.float32).reshape(2,866);before=x.copy();cols=d.drop_columns(arm)[2];d.transform_features(x,arm);keep=np.setdiff1d(np.arange(866),cols)
  assert (x[:,cols]==0).all();np.testing.assert_array_equal(x[:,keep],before[:,keep])
 records=[]
 for arm in ARMS:
  cfg=cfg_for(arm);n=4;dl=600 if arm=='long_60_60' else 100;sl=10 if arm=='short_10_10' else 60
  c.native.core.seed_all(17);dyn=torch.randn(n,dl,8,device='cuda');dyn[:,:,6]=1;dyn[:,:,7]=.1;state=torch.randn(n,sl,26,device='cuda')
  data={'prior':np.ones(7)/7,'weights':torch.ones(7,device='cuda'),'train':np.arange(n),'evaluation':np.arange(n),'tensors':{'dynamic':dyn,'state':state,'state_valid':torch.ones(n,sl,dtype=torch.bool,device='cuda'),'context':torch.randn(n,136,device='cuda'),'current':state[:,-1].clone(),'labels':torch.arange(n,device='cuda')%7}}
  d.transform_tensors(data['tensors'],arm);s=new_state(cfg,data,17)
  if arm=='full':
   ref=c.native.new(cfg,c.read(cfg['parent_training_config']),data,17,'cuda');c.native.base.equal_tree(ref['model'].state_dict(),s['model'].state_dict());del ref
  before=time.monotonic();loss,mass=c.update(s,data,np.arange(n));torch.cuda.synchronize();elapsed=time.monotonic()-before
  pred=c.probabilities(s['ema'].model,data,np.arange(n));assert pred.shape==(n,7) and np.isfinite(pred).all()
  records.append({'arm':arm,'parameters':sum(p.numel() for p in s['model'].parameters()),'update_seconds_batch4':elapsed,'loss_finite':bool(np.isfinite(loss))});del s,data;gc.collect();torch.cuda.empty_cache()
 c.save(d.OUT/'GPU_WITNESS.json',{'status':'PASS','synthetic_only':True,'real_training_started':False,'source_older_than_short_window_rejected':True,'future_sources_rejected':True,'feature_masks_preserve_other_columns':True,'records':records,'torch':torch.__version__,'numpy':np.__version__,'xgboost':xgb.__version__});print(json.dumps({'status':'PASS','records':records}),flush=True)
def timing():
 records=[]
 for arm in ARMS:
  cfg=cfg_for(arm);n=256;dl=600 if arm=='long_60_60' else 100;sl=10 if arm=='short_10_10' else 60
  c.native.core.seed_all(17);dyn=torch.randn(n,dl,8,device='cuda');dyn[:,:,6]=1;dyn[:,:,7]=.1;st=torch.randn(n,sl,26,device='cuda')
  data={'prior':np.ones(7)/7,'weights':torch.ones(7,device='cuda'),'train':np.arange(n),'evaluation':np.arange(n),'tensors':{'dynamic':dyn,'state':st,'state_valid':torch.ones(n,sl,dtype=torch.bool,device='cuda'),'context':torch.randn(n,136,device='cuda'),'current':st[:,-1].clone(),'labels':torch.arange(n,device='cuda')%7}}
  d.transform_tensors(data['tensors'],arm);s=new_state(cfg,data,17);c.update(s,data,data['train']);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
  for k in range(3):
   start=time.monotonic();c.update(s,data,data['train']);torch.cuda.synchronize();times.append(time.monotonic()-start)
  start=time.monotonic();c.probabilities(s['ema'].model,data,data['evaluation']);torch.cuda.synchronize();infer=time.monotonic()-start
  rec={'arm':arm,'batch':n,'update_median_s':float(np.median(times)),'inference_s':infer,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'synthetic_compute_timing_only':True};records.append(rec);print(json.dumps(rec),flush=True)
  del s,data,dyn,st;gc.collect();torch.cuda.empty_cache()
 # Real TRAIN witness for the first structural arm, no evaluation scoring.
 cfg=install('no_eri');data=selection(cfg,'TS','fold_1');s=new_state(cfg,data,17);times=[]
 for k in range(4):
  ids=data['train'][k*256:(k+1)*256];start=time.monotonic();loss,mass=c.update(s,data,ids);torch.cuda.synchronize();times.append(time.monotonic()-start)
 c.save(d.OUT/'TIMING_PREFLIGHT.json',{'status':'PASS','synthetic':records,'real_train':{'arm':'no_eri','targets_used':1024,'update_median_s':float(np.median(times[1:])),'fit_rows':len(data['train']),'dev_scores_computed':False,'test_data_read':False},'unix':time.time()})
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=ARMS);ap.add_argument('--mode',choices=['witness','timing','reuse_tree','calibration_tree','final_tree','calibration_neural','final_neural','predict','score'],required=True);ap.add_argument('--protocol');ap.add_argument('--fold');ap.add_argument('--seed',type=int);a=ap.parse_args();assert Path.cwd()==d.ROOT
 assert (torch.__version__,np.__version__,xgb.__version__)==('2.4.1+cu121','1.24.4','2.1.4');torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
 if a.mode=='witness':witness();return
 if a.mode=='timing':timing();return
 cfg=install(a.arm)
 if a.mode=='reuse_tree':reuse_tree(cfg,a.protocol,a.fold)
 elif a.mode in ['calibration_tree','final_tree']:engine.tree_worker(cfg,a.protocol,a.fold,a.mode=='final_tree')
 elif a.mode in ['calibration_neural','final_neural']:engine.neural_worker(cfg,a.protocol,a.fold,a.seed,a.mode=='final_neural')
 elif a.mode=='predict':predict(cfg)
 elif a.mode=='score':engine.score_once(cfg)
if __name__=='__main__':main()
