"""Rebuild all input pathways for equal physical time horizons, with old targets."""
from pathlib import Path
import argparse,json,time,gc,hashlib
import numpy as np
import pandas as pd
import torch
from scripts.sensor_ablation_20261113 import data as d
from scripts.run_sensor_history_controls_20261011 import raw_frames
from q3_sensor.endpoint_models import tokens
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**22),b''):h.update(b)
 return h.hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['short_10_10','long_60_60'],required=True);ap.add_argument('--protocol',required=True);ap.add_argument('--fold',required=True);a=ap.parse_args();assert Path.cwd()==d.ROOT;torch.set_num_threads(4)
 out=d.data_dir(a.arm,a.protocol,a.fold)
 assert not (out/'RECEIPT.json').exists(),'Never overwrite completed preparation'
 out.mkdir(parents=True,exist_ok=True);save(out/'PREPARATION_STARTED.json',{'unix':time.time(),'arm':a.arm,'test_metrics_computed':False})
 frames=raw_frames({'formal_source':'data/sensor_formal_0725_20261007'});sources={}
 inventory=d.read('reports/sensor_native_immu_inventory_20261021_v1/FULL_DAY_TRAIN_INVENTORY.json');source_records={int(r['cow_id']):r for r in inventory['cows']}
 for cow in range(1,11):
  p=Path('reports/sensor_native_immu_inventory_20261021_v1/raw_os_train_union')/f'T{cow:02d}_0725.csv';assert sha(p)==source_records[cow]['cached_csv_sha256'];sources[cow]=d.canonical_source(pd.read_csv(p,usecols=['timestamp']+list(d.CHANNELS)))
 outputs={};length=10 if a.arm=='short_10_10' else 60
 for phase in ['selection','formal']:
  parent=Path('reports/sensor_architecture_comparison_20261011/data')/phase/a.protocol/a.fold
  parent_receipt=d.read(parent/'DATA_RECEIPT.json')
  for name in ['targets.jsonl','normalizer.json']:assert sha(parent/name)==parent_receipt['outputs'][name]
  rows=[json.loads(s) for s in (parent/'targets.jsonl').read_text().splitlines()];state_norm=d.read(parent/'normalizer.json');fit=np.flatnonzero([r['split']=='train' for r in rows]);ev=np.flatnonzero([r['split']!='train' for r in rows]);y=np.array([rows[i]['label']-1 for i in fit],np.int64)
  if phase=='selection':
   nd=Path('reports/sensor_native_immu_preflight_20261021_v2/data')/a.protocol/a.fold;assert sha(nd/'normalizer.json')==d.read(nd/'DATA_RECEIPT.json')['outputs']['normalizer.json'];native_norm=d.read(nd/'normalizer.json')['physical']
  else:
   nd=Path('reports/sensor_shared_fusion_fivefold_20261024_v1/data')/a.protocol/a.fold;assert sha(nd/'NORMALIZERS.json')==d.read(nd/'DATA_RECEIPT.json')['outputs']['NORMALIZERS.json'];native_norm=d.read(nd/'NORMALIZERS.json')['native']
  arrays=d.state_windows(rows,frames,state_norm,length);parts=[];masks=[]
  for m in d.MODALITIES:
   fields={k:torch.from_numpy(v.copy()) for k,v in arrays[m].items()};fields['valid']=fields['valid'].bool();fields['x']=torch.where(fields['valid'],fields['x'],0.);parts.append(tokens(fields).numpy());masks.append(fields['valid'].any(-1).numpy())
  state=np.concatenate(parts,-1).astype(np.float32);current=state[:,-1].copy();unscaled=np.concatenate([d.descriptors(arrays,state_norm),current],-1)
  mean=unscaled[fit].astype(float).mean(0);raw=unscaled[fit].astype(float).std(0);std=np.maximum(raw,1e-6) if phase=='selection' else np.where(raw<=1e-6,1.,raw)
  context=((unscaled-mean)/std).astype(np.float32);dynamic=d.native_windows(rows,sources,native_norm,length)
  data={'dynamic':dynamic,'state':state,'state_valid':np.stack(masks,-1).any(-1),'context':context,'current':current};data['features']=d.summaries(data)
  dest=out if phase=='selection' else out/'refit';dest.mkdir(exist_ok=True)
  coverage={}
  for label,ids in [('FIT',fit),('EVALUATION',ev)]:
   coverage[label]={'targets':len(ids),'dynamic_valid_fraction':float((dynamic[ids,:,6]>.5).mean()),'state_valid_fraction':float(data['state_valid'][ids].mean()),'modality_valid_fraction':{m:float(arrays[m]['valid'][ids].astype(bool).mean()) for m in d.MODALITIES}}
  save(dest/'COVERAGE.json',coverage)
  np.savez(dest/'FIT.npz',**{k:v[fit] for k,v in data.items()},labels=y)
  evalname='DEV_INPUTS.npz' if phase=='selection' else 'TEST_INPUTS.npz';targetname='DEV_TARGETS.jsonl' if phase=='selection' else 'TEST_TARGETS.jsonl'
  np.savez(dest/evalname,**{k:v[ev] for k,v in data.items()})
  for name,ids in [('FIT_TARGETS.jsonl',fit),(targetname,ev)]:
   (dest/name).write_text(''.join(json.dumps(rows[i])+'\n' for i in ids))
  save(dest/'NORMALIZERS.json',{'state':state_norm,'native':native_norm,'context':{'mean':mean.tolist(),'std':std.tolist()},'context_fit_rows':len(fit),'window_seconds':length,'scope':'fit roles only; no evaluation target descriptors fitted'})
  if phase=='selection':
   outputs={name:sha(dest/name) for name in ['FIT.npz','DEV_INPUTS.npz','FIT_TARGETS.jsonl','DEV_TARGETS.jsonl','NORMALIZERS.json']}
  else:
   source={k:str((dest/name).relative_to(d.ROOT)) for k,name in [('FIT_path','FIT.npz'),('TEST_INPUTS_path','TEST_INPUTS.npz'),('TEST_TARGETS_path','TEST_TARGETS.jsonl'),('FIT_targets_path','FIT_TARGETS.jsonl')]}
   source.update({k.replace('_path','_sha256'):sha(d.ROOT/v) for k,v in list(source.items())});save(out/'REFIT_SOURCE.json',source);outputs['REFIT_SOURCE.json']=sha(out/'REFIT_SOURCE.json')
  del arrays,parts,masks,unscaled,context,dynamic,state,current,data;gc.collect()
 save(out/'RECEIPT.json',{'status':'PASS','outputs':outputs,'arm':a.arm,'role_bounded_source_rebuild':True,'all_pathways_recomputed':True,'window_seconds':length,'test_metrics_computed':False,'preparation_code':{name:sha(name) for name in ['scripts/sensor_ablation_20261113/data.py','scripts/sensor_ablation_20261113/prepare_scales.py']}})
 print(json.dumps({'status':'PASS','arm':a.arm,'protocol':a.protocol,'fold':a.fold}),flush=True)
if __name__=='__main__':main()
