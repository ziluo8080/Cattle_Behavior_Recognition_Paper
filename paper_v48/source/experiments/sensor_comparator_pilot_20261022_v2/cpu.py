"""Fixed CPU recipes: no internal CV, early stopping or Test loader."""
import ast
import json
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from numba import njit, prange
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

from q3_sensor.endpoint_models import tokens
from q3_sensor.protocol import MODALITIES
from scripts.sensor_native_immu_training_20261021 import load_selection
from scripts.report_convtran_context_20261022 import scores


@njit(cache=True)
def numba_seed(seed):
    np.random.seed(seed)


def data_cpu(root, protocol):
    data = load_selection(root, protocol, 'fold_1')
    source = root / 'reports/sensor_architecture_comparison_20261011/data/selection' / protocol / 'fold_1'
    native = root / 'reports/sensor_native_immu_preflight_20261021_v2/data' / protocol / 'fold_1'
    assert (source/'targets.jsonl').read_bytes() == (native/'targets.jsonl').read_bytes()
    state, valid = [], []
    with np.load(source/'windows.npz', allow_pickle=False) as archive:
        for modality in MODALITIES:
            fields = {k: torch.as_tensor(archive[f'{modality}_{k}']) for k in ('x','valid','fresh','age')}
            fields['valid'] = fields['valid'].bool()
            fields['x'] = torch.where(fields['valid'], fields['x'], 0.)
            state.append(tokens(fields).numpy()); valid.append(fields['valid'].any(-1).numpy())
    data['state'] = np.concatenate(state, -1).astype(np.float32)
    data['state_valid'] = np.stack(valid, -1).any(-1)
    dynamic = data['native'].copy(); mask = dynamic[:, :, 6] > .5
    dynamic[:, :, :6] = np.where(mask[:, :, None], dynamic[:, :, :6], 0.)
    dynamic[:, :, 7] = np.where(mask, np.clip(dynamic[:, :, 7], 0, .2), .2)
    data['native'] = dynamic; data['current'] = data['state'][:, -1].copy()
    return data


def summary_features(data):
    """Fixed, label-free summaries; zero-filled payload and validity are explicit."""
    pieces = [data['context'], data['current']]
    for values, tail in ((data['native'], 10), (data['state'], 5)):
        for window in (values, values[:, -tail:]):
            for array in (window.mean(1), window.std(1), window.min(1), window.max(1),
                          np.quantile(window, .25, axis=1), np.median(window, axis=1),
                          np.quantile(window, .75, axis=1), window[:, -1],
                          window[:, -1]-window[:, 0], np.abs(np.diff(window, axis=1)).mean(1)):
                pieces.append(np.asarray(array, dtype=np.float32))
    signal = data['native'][:, :, :6]
    spectrum = np.abs(np.fft.rfft(signal-signal.mean(1,keepdims=True),axis=1))**2 / 100
    frequencies = np.fft.rfftfreq(100, .1)
    for lo, hi in ((0,.5),(.5,1),(1,2),(2,5.01)):
        pieces.append(spectrum[:, (frequencies>=lo)&(frequencies<hi)].mean(1).astype(np.float32))
    result = np.concatenate(pieces, -1).astype(np.float32)
    assert np.isfinite(result).all()
    return result


def mr_functions():
    source = Path(__file__).parent/'vendor/multirocket/multirocket/multirocket_multivariate.py'
    name = __package__+'.author_multirocket'
    module = types.ModuleType(name); module.__file__ = str(source)
    module.__dict__.update(np=np, njit=njit, prange=prange)
    sys.modules[name] = module
    tree = ast.parse(source.read_text())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    exec(compile(ast.Module(body=body,type_ignores=[]),str(source),'exec'),module.__dict__)
    return module


def mr_fit(module, train, requested, cfg):
    # Author wrapper: divide requested features between original/difference and four pooling operators.
    train=np.ascontiguousarray(train,dtype=np.float64)  # Author's explicit Numba input signature.
    kernels = int(requested / 2 / cfg['features_per_kernel'])
    return (module.fit(train, kernels, cfg['max_dilations_per_kernel']),
            module.fit(np.ascontiguousarray(np.diff(train,axis=2)), kernels, cfg['max_dilations_per_kernel']))


def mr_transform(module, values, parameters, cfg):
    rows = []
    for start in range(0,len(values),cfg['transform_batch_size']):
        x = np.ascontiguousarray(values[start:start+cfg['transform_batch_size']],dtype=np.float64)
        z = module.transform(x,np.ascontiguousarray(np.diff(x,axis=2)),*parameters,cfg['features_per_kernel'])
        if np.isinf(z).any(): raise ValueError('Infinite MultiRocket feature')
        # Author wrapper also replaces undefined pooling values; never normalize using VAL.
        rows.append(np.nan_to_num(z,nan=0.))
    return np.concatenate(rows)


def evaluate_predictions(truth, pred):
    assert pred.shape==truth.shape and np.isin(pred,range(1,8)).all()
    cm = np.zeros((7,7),np.int64); np.add.at(cm,(truth-1,pred-1),1)
    f1,macro=scores(cm)
    return dict(macro_f1=macro,f1=f1.tolist(),confusion_matrix=cm.tolist())


def worker(cfg, root, protocol, arm, save):
    out=root/cfg['output']/'groups'/protocol/arm; out.mkdir(parents=True,exist_ok=True)
    assert not (out/'STARTED.json').exists(), 'No automatic retries'
    save(out/'STARTED.json',dict(started_at_unix=time.time(),arm=arm,protocol=protocol))
    data=data_cpu(root,protocol); train,val=data['train'],data['evaluation']; y=data['labels']
    counts=np.bincount(y[train],minlength=7); class_weight=np.sqrt(counts.sum()/(7*counts))
    missing=~((data['native'][:,:,6]>.5).any(1)|data['state_valid'].any(1))
    prior=int(counts.argmax())+1; final={}; history={}; records=[]
    base=summary_features(data) if arm=='XGBoost_Statistics' else None
    module=mr_functions() if arm=='MultiRocket_Fusion' else None
    branch_arrays=[np.ascontiguousarray(data[k].transpose(0,2,1)) for k in ('native','state')]
    for seed in cfg['seeds']:
        np.random.seed(seed); random.seed(seed); numba_seed(seed); tick=time.perf_counter()
        if arm=='XGBoost_Statistics':
            import xgboost as xgb
            options=cfg['xgboost']; tr=xgb.DMatrix(base[train],label=y[train],weight=class_weight[y[train]])
            va=xgb.DMatrix(base[val])
            params=dict(objective='multi:softprob',num_class=7,tree_method='hist',device='cpu',
                nthread=cfg['threads'],eta=options['eta'],max_depth=options['max_depth'],seed=seed,
                subsample=1.,colsample_bytree=1.,disable_default_eval_metric=1)
            model=xgb.train(params,tr,num_boost_round=options['rounds'])
            pred=model.predict(va).argmax(1)+1
            model.save_model(out/f'seed{seed}_model.json')
            restored=xgb.Booster(); restored.load_model(out/f'seed{seed}_model.json')
            restored.set_param({'nthread':cfg['threads']})
            np.testing.assert_array_equal(pred,restored.predict(va).argmax(1)+1)
            extra=dict(features=base.shape[1],rounds=options['rounds'],checkpoint_replay=True)
        else:
            import joblib
            options=cfg['multirocket']; fitted=[]; train_parts=[]; val_parts=[]
            for values in branch_arrays:
                pars=mr_fit(module,values[train],options['requested_features_per_branch'],options)
                fitted.append(pars)
                train_parts.append(mr_transform(module,values[train],pars,options))
                val_parts.append(mr_transform(module,values[val],pars,options))
            train_parts += [data['context'][train],data['current'][train]]
            val_parts += [data['context'][val],data['current'][val]]
            xtr=np.concatenate(train_parts,-1); xv=np.concatenate(val_parts,-1)
            del train_parts,val_parts
            scaler=StandardScaler(copy=False); xtr=scaler.fit_transform(xtr); xv=scaler.transform(xv)
            model=RidgeClassifier(alpha=options['ridge_alpha'],solver=options['ridge_solver'],
                tol=options['ridge_tol'],max_iter=options['ridge_max_iter'])
            model.fit(xtr,y[train],sample_weight=class_weight[y[train]])
            if np.max(model.n_iter_)>=options['ridge_max_iter']: raise RuntimeError('Ridge iteration budget exhausted')
            pred=model.predict(xv)+1
            checkpoint=out/f'seed{seed}_model.joblib'
            joblib.dump(dict(parameters=fitted,scaler=scaler,classifier=model),checkpoint)
            restored=joblib.load(checkpoint)
            # Replay all predictions from the serialized classifier and scaling/kernel subset from raw inputs.
            np.testing.assert_array_equal(pred,restored['classifier'].predict(xv)+1)
            replay=[mr_transform(module,v[val[:256]],p,options) for v,p in zip(branch_arrays,restored['parameters'])]
            replay += [data['context'][val[:256]],data['current'][val[:256]]]
            replay=restored['scaler'].transform(np.concatenate(replay,-1))
            np.testing.assert_array_equal(pred[:256],restored['classifier'].predict(replay)+1)
            extra=dict(features=xtr.shape[1],ridge_iterations=model.n_iter_.tolist(),checkpoint_replay=True,
                       raw_input_kernel_replay_targets=min(256,len(val)))
            del xtr,xv,replay,restored,model,scaler,fitted
        pred=np.asarray(pred,dtype=np.int64);pred[missing[val]]=prior
        metric=evaluate_predictions(y[val]+1,pred)
        np.save(out/f'seed{seed}_raw_final_predictions.npy',pred)
        final[str(seed)]={'raw':metric}; history[str(seed)]=dict(elapsed_seconds=time.perf_counter()-tick,**extra)
        save(out/'HISTORY.json',history)
        print(json.dumps(dict(protocol=protocol,arm=arm,seed=seed,final_macro_f1=metric['macro_f1'],**extra)),flush=True)
    save(out/'COMPLETE.json',dict(status='PASS',final=final,history=history,Test_allowed=False,
        recipe='fixed CPU recipe, not equivalent to neural epoch20 compute',parameters=None))


def synthetic_check(cfg):
    import xgboost as xgb
    generator=np.random.default_rng(17); module=mr_functions()
    options=dict(cfg['multirocket']);options['transform_batch_size']=5
    x=generator.normal(size=(21,8,100)).astype(np.float64)
    np.random.seed(17);numba_seed(17);first=mr_fit(module,x,1344,options)
    np.random.seed(17);numba_seed(17);second=mr_fit(module,x,1344,options)
    for a,b in zip(first,second):
        for p,q in zip(a,b):np.testing.assert_array_equal(p,q)
    z=mr_transform(module,x,first,options)
    whole=module.transform(x,np.ascontiguousarray(np.diff(x,axis=2)),*first,4)
    np.testing.assert_array_equal(z,np.nan_to_num(whole,nan=0.))
    scaler=StandardScaler();scaled=scaler.fit_transform(z)
    ridge=RidgeClassifier(alpha=1.,solver='lsqr',max_iter=10000).fit(scaled,np.arange(21)%7)
    assert np.isfinite(ridge.decision_function(scaled)).all()
    features=generator.normal(size=(42,12)).astype(np.float32)
    booster=xgb.train(dict(objective='multi:softprob',num_class=7,nthread=4,tree_method='hist',
        device='cpu',max_depth=2,seed=17),xgb.DMatrix(features,label=np.arange(42)%7),num_boost_round=3)
    assert np.isfinite(booster.predict(xgb.DMatrix(features))).all()
    return dict(status='PASS',scope='synthetic CPU only, no scientific data or scores',
        multirocket_fit_seed_exact=True,multirocket_batching_exact=True,ridge_finite=True,xgboost_finite=True,
        multirocket_witness_features=z.shape[1])
