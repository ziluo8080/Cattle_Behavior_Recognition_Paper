"""Repeat the already-declared fusion rules on the new, matched full reference."""
from pathlib import Path
import json,time
import numpy as np
from scripts.sensor_ablation_20261113.fixed_fusion import metric,margin
from scripts.sensor_recovery_20261114 import core as c
def run():
 root=c.ROOT/'reports/sensor_fair_selection_20261113';out=c.OUT/'fixed_fusion_new_reference';assert not (out/'WEIGHT_LOCK.json').exists();plan=c.read(c.ROOT/'configs/sensor_recovery_20261114.json');grid=plan['repeat_fusion_rules_on_new_full']['alpha_grid'];locks={};weights=[]
 ref=c.read(root/'SEAL.json');tl=c.read(root/'TRAINING_LOCK.json');pl=c.read(root/'PREDICTION_LOCK.json');assert pl['training_lock_sha256']==c.sha(root/'TRAINING_LOCK.json') and tl['seal_sha256']==c.sha(root/'SEAL.json')
 assert c.read(c.OUT/'RECOVERY_LOCK.json')['reference_seal_sha256']==c.sha(root/'SEAL.json')
 def locked(path):return pl['files'].get(str(path),pl['files'].get(str(path.relative_to(c.ROOT))))
 for p in ['TS','OS']:
  for fi in range(1,6):
   f=f'fold_{fi}';base=c.ROOT/'reports/sensor_converged_fivefold_20261027_v1';yp=base/'data'/p/f/'DEV_TARGETS.jsonl';receipt=c.read(base/'data'/p/f/'RECEIPT.json');assert c.sha(yp)==receipt['outputs']['DEV_TARGETS.jsonl'];y=np.array([json.loads(s)['label']-1 for s in yp.read_text().splitlines()]);locks[str(yp)]=c.sha(yp)
   xp=base/'calibration_tree'/p/f/'DEV.npy';assert c.sha(xp)==c.read(xp.parent/'COMPLETE.json')['files']['DEV.npy'];x=np.load(xp);locks[str(xp)]=c.sha(xp);bs=[]
   for seed in [17,23,29]:
    path=root/'calibration'/p/f/'ConvTran_ERI'/f'seed{seed}'/'selected_VAL_EMA.npy';cp=path.parent/'COMPLETE.json';assert c.sha(cp)==tl['files'].get(str(cp),tl['files'].get(str(cp.relative_to(c.ROOT))));done=c.read(cp);assert done['status']=='COMPLETE_CALIBRATION' and c.sha(path)==done['files'][str(path)];bs.append(np.load(path));locks[str(path)]=c.sha(path)
   candidates=[{'alpha':a,'dev_mean_f1':float(np.mean([metric(a*x+(1-a)*b,y)['macro_f1'] for b in bs]))} for a in grid];best=max(candidates,key=lambda z:(z['dev_mean_f1'],-abs(z['alpha']-.5),-z['alpha']));weights.append({'protocol':p,'fold':f,'alpha':best['alpha'],'candidates':candidates})
 c.save(out/'WEIGHT_LOCK.json',{'weights':weights,'inputs':locks,'unix':time.time(),'test_labels_parsed':False,'plan_sha256':c.sha(c.ROOT/'configs/sensor_recovery_20261114.json')})
 rows=[]
 for z in weights:
  p,f=z['protocol'],z['fold'];source=c.read(c.ROOT/'reports/sensor_converged_fivefold_20261027_v1/data'/p/f/'REFIT_SOURCE.json');yp=c.ROOT/source['TEST_TARGETS_path'];assert c.sha(yp)==source['TEST_TARGETS_sha256'];y=np.array([json.loads(s)['label']-1 for s in yp.read_text().splitlines()]);xp=root/'fusion_predictions'/p/f/'XGB.npy';assert c.sha(xp)==locked(xp);x=np.load(xp)
  for seed in [17,23,29]:
   bp=root/'prediction'/p/f/'ConvTran_ERI'/f'seed{seed}'/'test_ema.npy';assert c.sha(bp)==locked(bp);b=np.load(bp)
   for name,prob in {'equal_0.5':.5*x+.5*b,'fixed_0.7':.7*x+.3*b,'validation_fixed':z['alpha']*x+(1-z['alpha'])*b,'margin_0.7':margin(x,b)}.items():rows.append({'protocol':p,'fold':f,'seed':seed,'rule':name,**metric(prob,y)})
 c.save(out/'RESULTS.json',{'status':'COMPLETE','records':rows,'weights':weights,'reference':'new matched DS selection full model','no_new_fits':True})
if __name__=='__main__':run()
