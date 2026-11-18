"""Cloud-only, predeclared fusion-rule comparison using frozen matched experts."""
from pathlib import Path
import argparse,hashlib,json,time,sys
import numpy as np

ROOT=Path('/q3/mmcows-sensor-behavior-20261007')
SOURCE=ROOT/'reports/sensor_converged_fivefold_20261027_v1'
OUT=ROOT/'reports/sensor_ablation_20261113/stage1_fixed_fusion'
SEEDS=(17,23,29)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)
def metric(prob,y):
 assert prob.shape==(len(y),7) and np.isfinite(prob).all()
 assert np.allclose(prob.sum(1),1,atol=2e-5) and (prob>=-1e-7).all()
 pred=prob.argmax(1);cm=np.bincount(y*7+pred,minlength=49).reshape(7,7)
 den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(7,dtype=float),where=den>0)*100
 return {'macro_f1':float(f.mean()),'class_f1':f.tolist(),'confusion_matrix':cm.tolist()}
def margin(x,b):
 mx=np.diff(np.sort(x,axis=1)[:,-2:],axis=1)[:,0]*.7
 mb=np.diff(np.sort(b,axis=1)[:,-2:],axis=1)[:,0]*.3
 w=np.divide(mx,mx+mb,out=np.full(len(mx),.7),where=mx+mb>1e-15)
 return w[:,None]*x+(1-w[:,None])*b
def witness():
 y=np.arange(7);p=np.eye(7);assert metric(p,y)['macro_f1']==100
 assert metric(np.roll(p,1,axis=1),y)['macro_f1']==0
 np.testing.assert_allclose(margin(np.ones((7,7))/7,np.ones((7,7))/7),np.ones((7,7))/7)
 print(json.dumps({'status':'PASS','numpy':np.__version__,'real_data_read':False,'test_scores_read':False}))
def run():
 assert Path.cwd()==ROOT
 assert not (OUT/'STARTED.json').exists(),'Preserve existing attempt'
 cfg={'protocols':['TS','OS'],'folds':[f'fold_{i}' for i in range(1,6)],'seeds':list(SEEDS),'grid':[i/20 for i in range(21)],'selection':'maximum mean three-seed DEV macro-F1 per fold; ties nearest0.5 then smaller alpha','rules':['XGBoost','ConvTran_ERI','equal_0.5','fixed_0.7','validation_fixed','margin_0.7'],'no_test_based_selection':True,'no_new_training':True}
 save(OUT/'STARTED.json',{'unix':time.time(),'configuration':cfg,'code_sha256':sha(__file__)})
 weights=[];input_hashes={};predlock=read(SOURCE/'PREDICTION_LOCK.json')
 assert sha(SOURCE/'TRAINING_LOCK.json')==predlock['training_lock_sha256']
 for protocol in cfg['protocols']:
  for fold in cfg['folds']:
   d=SOURCE/'data'/protocol/fold;receipt=read(d/'RECEIPT.json');yp=d/'DEV_TARGETS.jsonl'
   assert sha(yp)==receipt['outputs']['DEV_TARGETS.jsonl'];y=np.array([json.loads(s)['label']-1 for s in yp.read_text().splitlines()]);input_hashes[str(yp)]=sha(yp)
   xp=SOURCE/'calibration_tree'/protocol/fold/'DEV.npy';assert sha(xp)==read(xp.parent/'COMPLETE.json')['files']['DEV.npy'];x=np.load(xp);input_hashes[str(xp)]=sha(xp)
   bs=[]
   for seed in SEEDS:
    p=SOURCE/'calibration_neural'/protocol/fold/f'seed{seed}'/'DEV.npy';assert sha(p)==read(p.parent/'COMPLETE.json')['files']['DEV.npy'];bs.append(np.load(p));input_hashes[str(p)]=sha(p)
   candidates=[{'alpha':a,'mean_dev_macro_f1':float(np.mean([metric(a*x+(1-a)*b,y)['macro_f1'] for b in bs]))} for a in cfg['grid']]
   best=max(candidates,key=lambda r:(r['mean_dev_macro_f1'],-abs(r['alpha']-.5),-r['alpha']))
   weights.append({'protocol':protocol,'fold':fold,'alpha':best['alpha'],'candidates':candidates})
 save(OUT/'WEIGHT_LOCK.json',{'unix':time.time(),'configuration':cfg,'weights':weights,'input_hashes':input_hashes,'test_truth_opened':False})
 save(OUT/'SCORING_STARTED.json',{'unix':time.time(),'weight_lock_sha256':sha(OUT/'WEIGHT_LOCK.json')})
 rows=[]
 for item in weights:
  p,f=item['protocol'],item['fold'];ref=read(SOURCE/'data'/p/f/'REFIT_SOURCE.json');yp=ROOT/ref['TEST_TARGETS_path'];assert sha(yp)==ref['TEST_TARGETS_sha256'];y=np.array([json.loads(s)['label']-1 for s in yp.read_text().splitlines()]);input_hashes[str(yp)]=sha(yp)
  folder=SOURCE/'predictions'/p/f
  def pred(name):
   path=folder/name;rel=str(path.relative_to(ROOT));assert sha(path)==predlock['files'][rel];input_hashes[str(path)]=sha(path);return np.load(path)
  x=pred('XGB.npy')
  for seed in SEEDS:
   b=pred(f'B_seed{seed}.npy');orig=pred(f'fusion_seed{seed}.npy');adaptive=margin(x,b)
   np.testing.assert_allclose(adaptive,orig,atol=2e-7,rtol=2e-6)
   probs={'XGBoost':x,'ConvTran_ERI':b,'equal_0.5':.5*x+.5*b,'fixed_0.7':.7*x+.3*b,'validation_fixed':item['alpha']*x+(1-item['alpha'])*b,'margin_0.7':orig}
   for rule,prob in probs.items():rows.append({'protocol':p,'fold':f,'seed':seed,'rule':rule,**metric(prob,y)})
 summary={}
 for p in cfg['protocols']:
  summary[p]={}
  for rule in cfg['rules']:
   folds=[float(np.mean([r['macro_f1'] for r in rows if r['protocol']==p and r['fold']==f and r['rule']==rule])) for f in cfg['folds']]
   summary[p][rule]={'fold_means':folds,'mean':float(np.mean(folds)),'fold_sample_sd':float(np.std(folds,ddof=1))}
 save(OUT/'RESULTS.json',{'status':'COMPLETE','configuration':cfg,'summary':summary,'records':rows,'weights':weights,'input_hashes':input_hashes,'scope':'Existing-data internal comparison, no independent new test set; original experts and manuscript unchanged'})
 print(json.dumps({'status':'COMPLETE','summary':summary},ensure_ascii=False))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--witness',action='store_true');a=ap.parse_args();witness() if a.witness else run()
