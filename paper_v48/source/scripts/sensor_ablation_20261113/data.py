"""Role-bounded horizon controls and whole-path modality interventions."""
from pathlib import Path
import json,copy
import numpy as np
import pandas as pd
import torch
from q3_sensor.protocol import MODALITIES,FEATURES,observations
from q3_sensor.endpoint_models import tokens
from scripts.sensor_native_immu_data_20261021 import canonical_source,CHANNELS

ROOT=Path('/q3/mmcows-sensor-behavior-20261007')
BASE=ROOT/'reports/sensor_converged_fivefold_20261027_v1'
OUT=ROOT/'reports/sensor_ablation_20261113'
STATE_SLICES={'uwb':list(range(0,8)),'hd':list(range(8,22)),'akl':list(range(22,26))}
CONTEXT_SLICES={'uwb':list(range(0,31))+list(range(102,118)),'hd':list(range(31,89))+list(range(118,132)),'akl':list(range(89,102))+list(range(132,136))}
def read(p):return json.loads(Path(p).read_text())
def data_dir(arm,p,f):return (OUT/'data'/arm/p/f) if arm in ('short_10_10','long_60_60') else BASE/'data'/p/f
def drop_columns(arm):
 if arm=='dynamic_only':return list(range(136)),list(range(26)),list(range(0,162))+list(range(322,842))
 if arm=='state_only':return [],[],list(range(162,322))+list(range(842,866))
 m={'no_uwb':'uwb','no_ankle':'akl'}.get(arm)
 if m is None:return [],[],[]
 state=STATE_SLICES[m];ctx=CONTEXT_SLICES[m]
 cols=ctx+[136+k for k in state]+[322+j*26+k for j in range(20) for k in state]
 return ctx,state,cols
def transform_tensors(values,arm):
 ctx,st,_=drop_columns(arm)
 if ctx:values['context'][...,ctx]=0
 if st:
  values['state'][...,st]=0;values['current'][...,st]=0
  valid_idx=[3,4,5,14,15,16,17,18,19,23]
  values['state_valid']=(values['state'][...,valid_idx]>.5).any(-1)
 if arm=='state_only':values['dynamic'].zero_()
 return values
def transform_features(x,arm):
 cols=drop_columns(arm)[2]
 if cols:x[:,cols]=0
 return x

def state_windows(rows,frames,norm,length):
 result={}
 groups={}
 for i,r in enumerate(rows):groups.setdefault((r['cow_id'],r['start'],r['end'],r['split']),[]).append(i)
 for m in MODALITIES:
  dims=6 if m=='hd' else len(FEATURES[m]);x=np.zeros((len(rows),length,dims),np.float32);v=np.zeros_like(x,np.uint8);fresh=np.zeros((len(rows),length,1),np.uint8);age=np.full((len(rows),length,1),np.nan,np.float32)
  mean,std=np.array(norm[m]['mean']),np.array(norm[m]['std'])
  for (cow,start,end,split),ids in groups.items():
   df=frames[cow][m];df=df[(df.timestamp>=start)&(df.timestamp<end)];ep,ot,val=observations(df,m)
   t=np.array([rows[i]['timestamp'] for i in ids]);grid=t[:,None]+np.arange(1-length,1)[None,:];sources=np.full((len(ids),length,dims),np.nan)
   for j in range(dims):
    good=np.isfinite(val[:,j]);e,s,z=ep[good],ot[good],val[good,j]
    if not len(e):continue
    pos=np.searchsorted(e,grid,side='right')-1;safe=np.maximum(pos,0);ok=(pos>=0)&(s[safe]>t[:,None]-length)
    assert (s[safe][ok]<=grid[ok]).all() and (s[safe][ok]>=start).all()
    x[ids,:,j]=np.where(ok,((z[safe]-mean[j])/std[j]).astype(np.float32),0);v[ids,:,j]=ok;sources[:,:,j]=np.where(ok,s[safe],np.nan)
    fresh[ids,:,0]|=(ok&(e[safe]>grid-1)&(e[safe]<=grid)).astype(np.uint8)
   oldest=np.where(np.isfinite(sources),sources,np.inf).min(-1);age[ids,:,0]=np.where(np.isfinite(oldest),grid-oldest,np.nan)
  result[m]={'x':x,'valid':v,'fresh':fresh,'age':age}
 return result
def native_windows(rows,sources,norm,length):
 n=length*10;out=np.zeros((len(rows),n,8),np.float32);out[:,:,7]=.2;groups={}
 for i,r in enumerate(rows):groups.setdefault((r['cow_id'],r['start'],r['end'],r['split']),[]).append(i)
 mean,std=np.array(norm['mean']),np.array(norm['std'])
 for (cow,start,end,split),ids in groups.items():
  ticks,values=sources[cow];keep=(ticks>=start*10)&(ticks<end*10)&np.isfinite(values).all(1);ticks,values=ticks[keep],values[keep]
  if not len(ticks):continue
  # Bound temporary 600-step physical arrays on the host.
  for k in range(0,len(ids),1000):
   ii=ids[k:k+1000];t=np.array([rows[i]['timestamp'] for i in ii],np.int64)*10;q=t[:,None]+np.arange(1-n,1)[None,:];pos=np.searchsorted(ticks,q,side='right')-1;safe=np.maximum(pos,0);picked=ticks[safe];age=q-picked;ok=(pos>=0)&(age>=0)&(age<=2)&(picked>t[:,None]-n)
   out[ii,:,:6]=np.where(ok[:,:,None],(values[safe]-mean)/std,0);out[ii,:,6]=ok;out[ii,:,7]=np.where(ok,age/10,.2)
   assert (picked[ok]>=start*10).all() and (picked[ok]<=q[ok]).all()
 return out
def descriptors(arrays,norm):
 pieces=[]
 for m in MODALITIES:
  d=arrays[m];x=d['x'];valid=d['valid'].astype(bool);count=valid.sum(1);safe=np.maximum(count,1);mean=(x*valid).sum(1)/safe;std=np.sqrt(((x-mean[:,None])**2*valid).sum(1)/safe);lo=np.where(valid,x,np.inf).min(1);hi=np.where(valid,x,-np.inf).max(1);lo[count==0]=hi[count==0]=0
  pieces.extend([x[:,-1],valid[:,-1],mean,std,lo,hi,hi-lo,count,count==0])
  pieces.append(np.column_stack([d['fresh'][:,:,0].sum(1),valid.any(2).mean(1),d['fresh'][:,-1,0],np.nan_to_num(d['age'][:,-1,0],nan=60.,posinf=60.,neginf=0.).clip(0,60)]))
 d=arrays['uwb'];motion=np.zeros((len(d['x']),8),np.float32);grid=np.arange(1-d['x'].shape[1],1,dtype=float);scale=np.array(norm['uwb']['std']);center=np.array(norm['uwb']['mean'])
 for i in range(len(motion)):
  usable=d['valid'][i].all(1)&np.isfinite(d['age'][i,:,0]);rel=grid[usable]-d['age'][i,usable,0].astype(float);xyz=d['x'][i,usable].astype(float)*scale+center;ts=[];points=[]
  for s in np.unique(rel):
   g=xyz[np.isclose(rel,s,rtol=0,atol=1e-7)]
   if len(g) and np.max(np.abs(g-g[0]))<=1e-6:ts.append(s);points.append(g[0])
  motion[i,0]=len(points);motion[i,7]=len(points)<2
  if len(points)>1:
   dt=np.diff(ts);assert (dt>0).all();pos=np.array(points);dist=np.linalg.norm(np.diff(pos[:,:2],axis=0),axis=1);speed=dist/dt
   motion[i,1:7]=[ts[-1]-ts[0],np.linalg.norm(pos[-1,:2]-pos[0,:2]),dist.sum(),speed.mean(),speed.max(),np.std(pos[:,2])]
 pieces.append(motion);out=np.concatenate(pieces,1).astype(np.float32);assert out.shape[1]==110;return out
def summaries(data):
 pieces=[data['context'],data['current']]
 for x,tail in [(data['dynamic'],10),(data['state'],5)]:
  for w in [x,x[:,-tail:]]:
   pieces.extend(np.asarray(a,np.float32) for a in [w.mean(1),w.std(1),w.min(1),w.max(1),np.quantile(w,.25,axis=1),np.median(w,axis=1),np.quantile(w,.75,axis=1),w[:,-1],w[:,-1]-w[:,0],np.abs(np.diff(w,axis=1)).mean(1)])
 signal=data['dynamic'][:,:,:6];n=signal.shape[1];spectrum=np.abs(np.fft.rfft(signal-signal.mean(1,keepdims=True),axis=1))**2/n;freq=np.fft.rfftfreq(n,.1)
 for lo,hi in [(0,.5),(.5,1),(1,2),(2,5.01)]:pieces.append(spectrum[:,(freq>=lo)&(freq<hi)].mean(1).astype(np.float32))
 out=np.concatenate(pieces,1).astype(np.float32);assert out.shape[1]==866 and np.isfinite(out).all();return out
