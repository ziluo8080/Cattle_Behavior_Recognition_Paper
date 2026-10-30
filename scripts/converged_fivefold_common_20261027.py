"""Pinned kernels and separate selection/refit convergence contracts."""
from pathlib import Path
import copy,hashlib,json
import numpy as np
import torch
from scripts import run_full_convergence_20261027 as native
from scripts.check_empirical_convergence_20261027 import strong_assess
from scripts.sensor_fusion_stability_20261027 import metric
from scripts.sensor_xgb_b_fusion_20261022 import temper,fuse

ROOT=Path('/q3/mmcows-sensor-behavior-20261007')
CONFIG=Path('configs/sensor_converged_fivefold_20261027_v1.json')
FIELDS=('dynamic','state','state_valid','context','current')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(obj,indent=2,allow_nan=False,ensure_ascii=False)+'\n'
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(data,encoding='utf-8');temp.replace(path)
def config():
    assert Path.cwd()==ROOT and (torch.__version__,np.__version__)==('2.4.1+cu121','1.24.4')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    cfg=read(CONFIG);assert sha(cfg['recipe'])==cfg['recipe_sha256']
    assert cfg['fusion']==dict(family='margin_average',XGB_temperature=1,B_temperature=1,XGB_weight=.7)
    return cfg
def source_files(cfg):
    paths={str(CONFIG),cfg['recipe'],cfg['environment'],cfg['parent_training_config'],cfg['architecture_config']}
    paths.update(str(p) for p in Path('scripts').glob('*converged_fivefold*20261027.py'))
    paths.update(read(cfg['recipe'])['architecture_code_sha256'])
    paths.update(str(p) for p in Path('q3_sensor').glob('*.py'))
    paths.update(str(p) for p in Path('third_party/convtran_context_20261022_v1').rglob('*.py'))
    paths.update(str(p) for p in Path('third_party/convtran_20261021_v1/Models').glob('*.py'))
    paths.update(['scripts/run_b_epoch_extension_20261024.py','scripts/run_convtran_endpoint_20261022.py','scripts/run_convtran_context_20261022.py','scripts/sensor_uwb_event_motion_20261020.py','scripts/sensor_xgb_b_fusion_20261022.py','scripts/sensor_native_immu_training_20261021.py','scripts/sensor_native_convtran_20261021.py','configs/sensor_convtran_context_20261022_v1.json'])
    paths.update(['scripts/run_full_convergence_20261027.py','scripts/run_b_low_lr_20261027.py','scripts/run_b_early_schedule_20261024.py','scripts/run_zero_head_replication_20261024.py','scripts/run_tree_convergence_20261027.py','scripts/check_empirical_convergence_20261027.py','scripts/sensor_state_average_20261027.py','scripts/sensor_bn_momentum_20261027.py','scripts/sensor_fusion_stability_20261027.py','scripts/sensor_learned_fusion_data_20261023.py','experiments/sensor_comparator_pilot_20261022_v2/cpu.py','.aris/compute/bitahub-root-converged-fivefold-20261027.md'])
    return {p:sha(p) for p in sorted(paths)}
def check_seal(cfg):
    s=read(Path(cfg['output'])/'TRAINING_SEAL.json');assert s['config']==cfg
    for p,h in s['code'].items():assert sha(p)==h,p
    for p,h in s['data'].items():
        if p.endswith('RECEIPT.json') or p.endswith('REFIT_SOURCE.json'):assert sha(p)==h,p
    return s
def data_dir(cfg,p,f):return Path(cfg['output'])/'data'/p/f
def case_dir(cfg,stage,p,f,seed=None):
    path=Path(cfg['output'])/stage/p/f
    return path if seed is None else path/f'seed{seed}'
def load_selection(cfg,p,f,device='cuda'):
    d=data_dir(cfg,p,f);r=read(d/'RECEIPT.json')
    for n,h in r['outputs'].items():assert sha(d/n)==h,n
    with np.load(d/'FIT.npz',allow_pickle=False) as fit,np.load(d/'DEV_INPUTS.npz',allow_pickle=False) as dev:
        assert 'labels' not in dev.files
        y=fit['labels'];counts=np.bincount(y,minlength=7);values={k:np.concatenate([fit[k],dev[k]]) for k in FIELDS}
        labels=np.concatenate([y,np.full(len(dev['dynamic']),-100,np.int64)])
    tensors={k:torch.as_tensor(v,device=device) for k,v in values.items()};tensors['labels']=torch.as_tensor(labels,device=device)
    truth=np.array([json.loads(s)['label']-1 for s in (d/'DEV_TARGETS.jsonl').read_text().splitlines()])
    return dict(tensors=tensors,weights=torch.tensor(np.sqrt(counts.sum()/(7*counts)),dtype=torch.float32,device=device),
        prior=counts/counts.sum(),train=np.arange(len(y)),evaluation=np.arange(len(y),len(labels)),fit_truth=y,dev_truth=truth)
def load_refit(cfg,p,f,device='cuda'):
    r=read(data_dir(cfg,p,f)/'REFIT_SOURCE.json');path=Path(r['FIT_path']);assert sha(path)==r['FIT_sha256']
    with np.load(path,allow_pickle=False) as fit:
        y=fit['labels'];counts=np.bincount(y,minlength=7);tensors={k:torch.as_tensor(fit[k],device=device) for k in FIELDS}
    tensors['labels']=torch.as_tensor(y,device=device)
    return dict(tensors=tensors,weights=torch.tensor(np.sqrt(counts.sum()/(7*counts)),dtype=torch.float32,device=device),
        prior=counts/counts.sum(),train=np.arange(len(y)),evaluation=np.array([],dtype=np.int64),fit_truth=y)
def new_state(cfg,data,seed,lr=None):
    parent=read(cfg['parent_training_config']);c=dict(cfg,learning_rate=cfg['learning_rate'] if lr is None else lr)
    return native.new(c,parent,data,seed,data['tensors']['dynamic'].device)
def update(state,data,ids):return native.update(state,data,ids)
def snapshot(state,local):return native.snapshot(state,local)
def evaluation(state,snaps,local):return native.clone(state,snaps,local)
def probabilities(model,data,ids):return native.base.infer(model,data,ids)
def score(prob,truth,data):return metric(prob,truth,data['weights'].double().cpu().numpy())[0]
def fused(px,pb,cfg):
    g=cfg['fusion'];return fuse(temper(px,g['XGB_temperature']),temper(pb,g['B_temperature']),g['family'],g['XGB_weight'])
def joint_assess(h):
    result=strong_assess(h)
    if len(h)<40:return result
    extra=[];minimum=min(r['fusion_ema']['weighted_CE'] for r in h)
    for w in [h[-20:-10],h[-10:]]:
        checks=[]
        for key in ['fusion_fit','fusion_ema']:
            v=np.array([r[key]['weighted_CE'] for r in w]);d=float((v[5:].mean()-v[:5].mean())/v[:5].mean());checks.append(dict(key=key,drift=d,passed=abs(d)<=.02))
        extra.append(dict(passed=bool(all(c['passed'] for c in checks) and np.mean([r['fusion_ema']['weighted_CE'] for r in w])<=minimum*1.05),checks=checks))
    result['fusion_CE_windows']=extra;result['passed']=bool(result['passed'] and all(w['passed'] for w in extra));return result
def fit_assess(h,minimum=40):
    if len(h)<minimum or len(h)<40:return dict(passed=False,reason='minimum_epochs')
    windows=[]
    for w in [h[-20:-10],h[-10:]]:
        checks=[]
        for key,tol in [('train_ce',.05),('fit_ema',.02),('fusion_fit',.02)]:
            v=np.array([r[key] if key=='train_ce' else r[key]['weighted_CE'] for r in w]);d=float((v[5:].mean()-v[:5].mean())/v[:5].mean());checks.append(dict(key=key,drift=d,passed=abs(d)<=tol))
        ok=all(z['passed'] for z in checks) and max(r['raw_relative_parameter_change'] for r in w)<=.002 and max(r['EMA_relative_parameter_change'] for r in w)<=.002 and min(r['learning_rate'] for r in w)>=3e-6
        windows.append(dict(passed=bool(ok),checks=checks))
    return dict(passed=all(w['passed'] for w in windows),windows=windows,scope='Final FIT stationarity only; held-outDEV convergence belongs to calibration stage')
def planned_rate(plan,epoch):return plan['base_learning_rates'][epoch-1] if epoch<=len(plan['base_learning_rates']) else 3e-6
def checkpoint(path,state,snaps,**meta):native.checkpoint(path,state,causal_ema_snapshots=snaps,**meta)
def restore(cfg,data,payload,device):return native.restore(cfg,read(cfg['parent_training_config']),data,payload,device)
