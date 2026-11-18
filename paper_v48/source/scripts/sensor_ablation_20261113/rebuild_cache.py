"""Cloud-only reconstruction; publish missing files only if old sealed hashes match."""
from pathlib import Path
import argparse,json,hashlib,time,os,gc
import numpy as np
import torch
from q3_sensor.protocol import build_windows,MODALITIES
from q3_sensor.endpoint_models import tokens
from scripts.run_sensor_history_controls_20261011 import raw_frames
from scripts.sensor_native_immu_data_20261021 import canonical_source,CHANNELS,windows as selection_native_windows
from scripts.prepare_shared_fusion_fivefold_20261024 import native_windows
from scripts.sensor_classic_features_20261012 import descriptors
from experiments.sensor_comparator_pilot_20261022_v2.cpu import summary_features
import pandas as pd
ROOT=Path('/q3/mmcows-sensor-behavior-20261007');OUT=ROOT/'reports/sensor_ablation_20261113/reconstruction'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**22),b''):h.update(b)
 return h.hexdigest()
def publish(tmp,target,expected):
 actual=sha(tmp);assert actual==expected,dict(target=str(target),expected=expected,actual=actual)
 if target.exists():assert sha(target)==expected
 else:os.link(tmp,target)
 return {'path':str(target.resolve().relative_to(ROOT)),'sha256':actual,'bytes':target.stat().st_size}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--protocol',choices=['TS','OS']);ap.add_argument('--fold');a=ap.parse_args();assert Path.cwd()==ROOT;torch.set_num_threads(4)
 frames=raw_frames({'formal_source':'data/sensor_formal_0725_20261007'});sources={}
 for cow in range(1,11):
  p=Path('reports/sensor_native_immu_inventory_20261021_v1/raw_os_train_union')/f'T{cow:02d}_0725.csv';sources[cow]=canonical_source(pd.read_csv(p,usecols=['timestamp']+list(CHANNELS)))
 cases=[(p,f'fold_{f}') for p in ['TS','OS'] for f in range(1,6)] if a.protocol is None else [(a.protocol,a.fold)]
 for protocol,fold in cases:
  start=time.monotonic();records=[];case=OUT/protocol/fold;case.mkdir(parents=True,exist_ok=True)
  if (case/'COMPLETE.json').exists():continue
  work=case/str(time.time_ns());work.mkdir()
  for phase in ['selection','formal']:
   parent=Path('reports/sensor_architecture_comparison_20261011/data')/phase/protocol/fold
   rows=[json.loads(s) for s in (parent/'targets.jsonl').read_text().splitlines()];normal=read(parent/'normalizer.json');w=None
   expected=read(parent/'DATA_RECEIPT.json')['outputs']['windows.npz']
   if (parent/'windows.npz').exists():
    assert sha(parent/'windows.npz')==expected
    with np.load(parent/'windows.npz') as zz:arrays={m:{k:zz[f'{m}_{k}'] for k in ['x','valid','fresh','age']} for m in MODALITIES}
   else:
    w=build_windows(rows,frames,normal);arrays={m:{k:w[m][k] for k in ['x','valid','fresh','age']} for m in MODALITIES}
    tmp=work/(phase+'_windows.npz');np.savez(tmp,**{f'{m}_{k}':arrays[m][k] for m in MODALITIES for k in ['x','valid','fresh','age']})
    records.append(publish(tmp,parent/'windows.npz',expected))
   state=[];valid=[]
   for m in MODALITIES:
    data={k:torch.from_numpy(arrays[m][k].copy()) for k in arrays[m]};data['valid']=data['valid'].bool();data['x']=torch.where(data['valid'],data['x'],0.)
    state.append(tokens(data).numpy());valid.append(data['valid'].any(-1).numpy())
   state=np.concatenate(state,-1).astype(np.float32);current=state[:,-1].copy();desc,_=descriptors(arrays,normal);unscaled=np.concatenate([desc,current],-1)
   fit=np.flatnonzero([r['split']=='train' for r in rows]);evaluation=np.flatnonzero([r['split']!='train' for r in rows]);y=np.array([rows[i]['label']-1 for i in fit],np.int64)
   target=Path('reports/sensor_converged_fivefold_20261027_v1/data' if phase=='selection' else 'reports/sensor_shared_fusion_fivefold_20261024_v1/data')/protocol/fold
   if phase=='selection':
    norm=read(Path('reports/sensor_native_immu_preflight_20261021_v2/data')/protocol/fold/'normalizer.json');native_norm=norm['physical'];context_norm=norm['context'];receipt=read(target/'RECEIPT.json')
   else:
    norm=read(target/'NORMALIZERS.json');native_norm=norm['native'];context_norm=norm['context'];receipt=read(target/'DATA_RECEIPT.json')
   context=((unscaled-np.array(context_norm['mean']))/np.array(context_norm['std'])).astype(np.float32)
   dynamic,ticks=(selection_native_windows if phase=='selection' else native_windows)(rows,sources,native_norm)
   data={'dynamic':dynamic,'context':context,'state':state,'state_valid':np.stack(valid,-1).any(-1),'current':current} if phase=='selection' else {'state':state,'state_valid':np.stack(valid,-1).any(-1),'current':current,'context':context,'dynamic':dynamic}
   data['features']=summary_features({'context':context,'current':current,'native':dynamic,'state':state})
   counts=np.bincount(y,minlength=7);prior=counts/counts.sum();weights=np.sqrt(counts.sum()/(7*counts))
   tmp=work/(phase+'_FIT.npz');extra={'labels':y} if phase=='selection' else {'labels':y,'prior':prior,'class_weights':weights}
   np.savez(tmp,**{k:v[fit] for k,v in data.items()},**extra);records.append(publish(tmp,target/'FIT.npz',receipt['outputs']['FIT.npz']))
   name='DEV_INPUTS.npz' if phase=='selection' else 'TEST_INPUTS.npz';tmp=work/(phase+'_'+name)
   np.savez(tmp,**{k:v[evaluation] for k,v in data.items()},**({} if phase=='selection' else {'prior':prior}));records.append(publish(tmp,target/name,receipt['outputs'][name]))
   if phase=='formal':
    tmp=work/'NATIVE_SOURCE_TICKS.npy';np.save(tmp,ticks);records.append(publish(tmp,target/tmp.name,receipt['outputs'][tmp.name]))
   del w,arrays,state,current,desc,unscaled,context,dynamic,ticks,data;gc.collect()
  result={'status':'PASS','protocol':protocol,'fold':fold,'elapsed':time.monotonic()-start,'files':records};(case/'COMPLETE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
