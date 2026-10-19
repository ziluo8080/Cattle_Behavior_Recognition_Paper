"""Cloud utilities shared by preflight and fixed architecture ablations."""
from pathlib import Path
import numpy as np
import torch
from scripts.sensor_local_tcn_residual import read, write, sha, f1, metrics, dataset
from scripts.sensor_native_motion_features import native_motion, fit_motion_scaler, scale_motion
from scripts.sensor_motion_architecture_model import base_outputs, coarse
from q3_sensor.endpoint_models import EndpointFusion

CONFIG = Path('configs/sensor_motion_pair_architecture_validation_20261009.json')
ROOT = Path('/q3/mmcows-sensor-behavior-20261007')


def grid_motion(x, valid, age, timestamps, starts, std):
    features, pair, dt = native_motion(x, valid, age, timestamps, starts, std)
    n, length, _ = x.shape
    grid = np.asarray(timestamps, float)[:, None] - (length - 1) + np.arange(length)
    source = grid - np.asarray(age[:, :, 0], float)
    full = np.asarray(valid).all(-1) & np.isfinite(source) & (source > np.asarray(timestamps)[:, None]-length)
    index = np.broadcast_to(np.arange(length), (n, length)); rows = np.arange(n)
    last = np.where(full, index, -1).max(1)
    latest = source[rows, np.maximum(last, 0)]
    previous = np.where(full & (source < latest[:, None]-1e-6), index, -1).max(1)
    prior = source[rows, np.maximum(previous, 0)]
    first_last = np.where(full & (np.abs(source-latest[:, None]) <= 1e-6), index, length).min(1)
    first_previous = np.where(full & (np.abs(source-prior[:, None]) <= 1e-6), index, length).min(1)
    grid_dt = (first_last-first_previous).astype(float)
    if np.any(pair & (grid_dt <= 0)): raise AssertionError('Positive first-appearance grid interval required')
    v = np.zeros((n, 3)); v[pair] = features[pair, :3] * (dt[pair]/grid_dt[pair])[:, None]
    result = features.copy(); result[:, :3] = v
    result[:, 3] = np.linalg.norm(v, axis=1); result[:, 4] = np.linalg.norm(v[:, :2], axis=1)
    return result, pair, grid_dt


def get_data(parent, proto):
    train, val, stats, trpool, vapool = dataset(parent, proto)
    norm = read(Path(parent['cache'])/proto/'normalizer.json')
    native, grid, pairs, raw, raw_grid = [], [], [], [], []
    for table, pool in [(train, trpool), (val, vapool)]:
        d = pool.arrays['uwb']['clean']; args=(d['x'], d['valid'], d['age'], table.timestamp.to_numpy(), table.start.to_numpy(), norm['uwb']['std'])
        nf, npair, _ = native_motion(*args); gf, gpair, _ = grid_motion(*args)
        np.testing.assert_array_equal(npair, gpair)
        raw.append(nf); raw_grid.append(gf); pairs.append(npair)
    ns=fit_motion_scaler(raw[0], pairs[0]); gs=fit_motion_scaler(raw_grid[0], pairs[0])
    native=[scale_motion(v,p,ns) for v,p in zip(raw,pairs)]
    grid=[scale_motion(v,p,gs) for v,p in zip(raw_grid,pairs)]
    return dict(train=train,val=val,stats=stats,trpool=trpool,vapool=vapool,native=native,grid=grid,pairs=pairs,
                raw=raw,native_scaler=ns,grid_scaler=gs)


def load_base(parent, proto, seed):
    old=Path(parent['output'])/proto/f'P0_seed{seed}'
    ckpt=torch.load(old/'model.pt',map_location='cpu',weights_only=False)
    model=EndpointFusion('gru',ckpt['prior']);model.load_state_dict(ckpt['state_dict'],strict=True)
    return model.cuda()


def setup():
    if Path.cwd()!=ROOT or not torch.cuda.is_available(): raise RuntimeError('Existing cloud CUDA only')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False


def frozen_features(base, pool, size, mask='Full', batch_size=256):
    base.eval();contexts=[];logits=[]
    with torch.no_grad():
        for start in range(0,size,batch_size):
            idx=np.arange(start,min(start+batch_size,size));x,b=base_outputs(base,pool.batch(idx),mask)
            contexts.append(x.cpu().numpy());logits.append(b.cpu().numpy())
    return np.concatenate(contexts),np.concatenate(logits)
