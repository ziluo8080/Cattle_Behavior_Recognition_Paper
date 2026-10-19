"""Official TRAIN+VAL/Fit to TEST data and complete-system ablation wrappers."""
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
import torch
from q3_sensor.endpoint_models import EndpointFusion
from scripts.sensor_reliability_data import DamagePool
from scripts.sensor_motion_architecture_common import grid_motion, setup, frozen_features
from scripts.sensor_motion_architecture_model import MaskedBase, ArchitectureRefiner
from scripts.sensor_native_motion_features import native_motion, fit_motion_scaler, scale_motion
from scripts.sensor_local_tcn_residual import read, write, sha, f1, metrics

CONFIG=Path('configs/sensor_motion_architecture_formal_20261009.json')
ROOT=Path('/q3/mmcows-sensor-behavior-20261007')


def original_path(cfg,case,seed):
    proto,fold=case.split('/')
    return Path(cfg['original_formal'][proto])/fold/f'NN_seed{seed}.pt'


def load_base(cfg,case,seed):
    checkpoint=torch.load(original_path(cfg,case,seed),map_location='cpu',weights_only=False)
    base=EndpointFusion('gru',checkpoint['prior']);base.load_state_dict(checkpoint['state_dict'],strict=True)
    return base.cuda()


@lru_cache(maxsize=10)
def cached_data(cache_root,case):
    cache=Path(cache_root)/case;rows=pd.read_json(cache/'targets.jsonl',lines=True,convert_dates=False)
    assert set(rows.split)=={'train','test'} and not rows.duplicated(['cow_id','timestamp']).any()
    fit=rows[rows.split.eq('train')].reset_index(drop=True);test=rows[rows.split.eq('test')].reset_index(drop=True)
    stats=read(cache/'fit_statistics.json');norm=read(cache/'normalizer.json')
    counts=np.bincount(fit.label.to_numpy(),minlength=8)[1:8]
    np.testing.assert_array_equal(counts,stats['class_counts'])
    np.testing.assert_allclose(stats['prior'],counts/counts.sum(),atol=1e-15,rtol=0)
    np.testing.assert_allclose(stats['CE_weights'],np.sqrt(counts.sum()/(7*counts)),atol=1e-15,rtol=0)
    pools=[DamagePool(cache/'train',['clean']),DamagePool(cache/'test',['clean'])];raw=[];rawgrid=[];pairs=[]
    for table,pool in zip([fit,test],pools):
        uwb=pool.arrays['uwb']['clean'];args=(uwb['x'],uwb['valid'],uwb['age'],table.timestamp.to_numpy(),table.start.to_numpy(),norm['uwb']['std'])
        x,p,_=native_motion(*args);g,pg,_=grid_motion(*args);np.testing.assert_array_equal(p,pg)
        raw.append(x);rawgrid.append(g);pairs.append(p)
    ns=fit_motion_scaler(raw[0],pairs[0]);gs=fit_motion_scaler(rawgrid[0],pairs[0])
    native=[scale_motion(x,p,ns) for x,p in zip(raw,pairs)];grid=[scale_motion(x,p,gs) for x,p in zip(rawgrid,pairs)]
    # val is an internal compatibility alias; all serialized scope is explicitly TEST.
    return dict(train=fit,val=test,stats=stats,trpool=pools[0],vapool=pools[1],native=native,grid=grid,pairs=pairs,raw=raw,
                native_scaler=ns,grid_scaler=gs,evaluation_role='TEST',fit='author_TRAIN_plus_VAL')


def get_data(cfg,case): return cached_data(cfg['cache'],case)


def drop_batch(batch,drop):
    if drop is None: return batch
    result=dict(batch);values=batch[drop]
    result[drop]={k:torch.full_like(v,60) if k=='age' else torch.zeros_like(v) for k,v in values.items()}
    return result


def variant(kind):
    k=kind.removesuffix('Motion')
    return {'NoUWB':'uwb','NoHD':'hd','NoAkl':'akl'}.get(k)


class FormalBase(MaskedBase):
    def __init__(self,prior,state,kind):
        mask=kind if kind in ['RecurrentOnly','CurrentOnly','FullUniform','FullBalanced'] else 'Full'
        super().__init__(prior,state,mask);self.drop=variant(kind)

    def forward(self,batch): return super().forward(drop_batch(batch,self.drop))


class FormalRefiner(ArchitectureRefiner):
    def __init__(self,base,kind,seed,mask='Full'):
        super().__init__(base,kind,seed,mask);self.drop=variant(kind)

    def forward(self,batch,motion):
        if self.drop=='uwb':
            motion=torch.zeros_like(motion) # all nine UWB motion/quality channels disabled; no information from deleted source
        return super().forward(drop_batch(batch,self.drop),motion)
