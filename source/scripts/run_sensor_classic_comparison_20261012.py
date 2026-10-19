"""Cloud-only classic baselines, reusing audited same-budget GRU anchor."""
import argparse
import fcntl
import gc
import json
import os
import resource
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from numba import set_num_threads

from q3_sensor.evaluation import metrics
from q3_sensor.protocol import MODALITIES, sha
from scripts import run_sensor_architecture_comparison_20261011 as neural
from scripts.run_sensor_history_controls_20261011 import read_json, write_json
from scripts.sensor_history_control_models_20261011 import CurrentRepeatGRU
from scripts.sensor_classic_features_20261012 import descriptors, sequence_features, rocket_fit, rocket

CONFIG = Path('configs/sensor_classic_comparison_20261012.json')
SOURCE_FILES = [CONFIG, Path(__file__), Path('configs/sensor_classic_env_20261012.json'),
    Path('scripts/sensor_classic_features_20261012.py'), Path('scripts/launch_sensor_classic_comparison_20261012.py'),
    Path('scripts/summarize_sensor_classic_comparison_20261012.py'),
    Path('scripts/sensor_history_control_models_20261011.py'),
    Path('third_party/minirocket_20261012/minirocket_multivariate.py'),
    Path('third_party/minirocket_20261012/LICENSE')]


def setup():
    cfg = read_json(CONFIG)
    if Path.cwd() != Path('/q3/mmcows-sensor-behavior-20261007'):
        raise RuntimeError('Cloud project required')
    # Validate the original source/env/parents rather than mutate the old batch.
    neural.setup()
    set_num_threads(cfg['threads'])
    root = Path(cfg['output'])
    root.mkdir(parents=True, exist_ok=True)
    parent = Path(cfg['reuse_architecture_root'])
    assert sha(parent/'run_identity.json') == cfg['reuse_architecture_identity_sha256']
    assert sha(parent/'METHOD_LOCK.json') == cfg['reuse_architecture_lock_sha256']
    sources = {str(p): sha(p) for p in SOURCE_FILES}
    inherited = read_json(parent/'run_identity.json')['source_sha256']
    for p, digest in inherited.items():
        assert sha(p) == digest, f'Frozen parent source changed: {p}'
    sources.update(inherited)
    for cow in range(1,11):
        path=Path(cfg['source'])/f'main_data/uwb/T{cow:02d}/T{cow:02d}_0725.csv'
        sources[str(path)]=sha(path)
    witness = read_json(root/'ENV_WITNESS.json')
    assert witness['status'] == 'PASS' and witness['versions']['numba'] == '0.60.0'
    assert witness['vendor_sha256'] == sources['third_party/minirocket_20261012/minirocket_multivariate.py']
    identity = dict(config=cfg, sources=sources,
        parent_identity_sha256=sha(parent/'run_identity.json'), parent_lock_sha256=sha(parent/'METHOD_LOCK.json'),
        environment_witness_sha256=sha(root/'ENV_WITNESS.json'))
    path = root/'run_identity.json'
    if path.exists():
        assert read_json(path) == identity, 'Immutable implementation changed'
    else:
        write_json(path, identity)
    jobs = [f'{phase}/{protocol}/{fold}/{kind}/{suffix}'
        for phase in ('selection','formal') for protocol in cfg['protocols'] for fold in cfg['folds']
        for kind in cfg['models'] for suffix in
        (['candidate0','candidate1'] if phase=='selection' else [f'seed{s}' for s in cfg['formal_seeds']])]
    manifest = dict(jobs=jobs, selection_jobs=80, formal_jobs=120, reused_GRU_jobs=50,
        identity_sha256=sha(path), max_parallel=1, no_automatic_hyperparameter_changes=True)
    if (root/'MANIFEST.json').exists():
        assert read_json(root/'MANIFEST.json') == manifest
    else:
        write_json(root/'MANIFEST.json', manifest)
    return cfg, root


def data_for(cfg, phase, protocol, fold):
    directory = Path(cfg['reuse_architecture_root'])/'data'/phase/protocol/fold
    receipt = read_json(directory/'DATA_RECEIPT.json')
    assert (receipt['phase'],receipt['protocol'],receipt['fold']) == (phase,protocol,fold)
    for p,digest in receipt['outputs'].items():
        assert sha(directory/p) == digest
    targets = pd.read_json(directory/'targets.jsonl',lines=True,convert_dates=False)
    assert set(targets.split) == ({'train','val'} if phase=='selection' else {'train','test'})
    arrays = dict(np.load(directory/'windows.npz',allow_pickle=False))
    values = {m:{k: arrays[f'{m}_{k}'] for k in ('x','valid','fresh','age')} for m in MODALITIES}
    return dict(targets=targets, labels=targets.label.to_numpy(dtype=int),
        train_ids=np.flatnonzero(targets.split.eq('train').to_numpy()),
        evaluation_ids=np.flatnonzero(~targets.split.eq('train').to_numpy()), arrays=values,
        normalizer=read_json(directory/'normalizer.json'),
        identity=dict(directory=str(directory),receipt_sha256=sha(directory/'DATA_RECEIPT.json'),outputs=receipt['outputs']))


def verify(directory, root, expected_id, data_identity):
    complete = read_json(directory/'COMPLETE.json')
    assert complete['job_id'] == expected_id and complete['identity_sha256'] == sha(root/'run_identity.json')
    assert complete['data_identity'] == data_identity
    inputs,result=read_json(directory/'INPUTS.json'),read_json(directory/'result.json')
    assert inputs['job_id']==result['job_id']==expected_id
    assert result['seed']==inputs['seed'] and result['learning_rate']==inputs['learning_rate']
    assert inputs['identity_sha256']==result['identity_sha256']==complete['identity_sha256']
    for name,digest in complete['outputs'].items():
        assert sha(directory/name)==digest, f'Output changed: {directory/name}'


def fallback(data, current):
    arrays=data['arrays']
    mask=np.stack([v['valid'][:,-1].any(-1) if current else v['valid'].reshape(len(v['valid']),-1).any(1)
                   for v in arrays.values()],axis=1).any(1)
    counts=np.bincount(data['labels'][data['train_ids']],minlength=8)[1:8]
    return mask,int(np.argmax(counts)+1)


def save_job(root, job_id, data, inputs, fitted, pred_train, pred_eval, elapsed, resources):
    directory=root/job_id
    table=data['targets'].iloc[data['evaluation_ids']][['cow_id','timestamp','label']].reset_index(drop=True)
    table['prediction']=pred_eval
    table.to_csv(directory/'predictions.csv',index=False)
    result=dict(inputs,model=job_id.split('/')[3],phase=job_id.split('/')[0],
        protocol=job_id.split('/')[1],fold=job_id.split('/')[2],
        selected_epoch=1,trained_epochs=1,selected_validation_macro_f1=None,
        metrics=metrics(data['labels'][data['evaluation_ids']],pred_eval),
        Fit_metrics=metrics(data['labels'][data['train_ids']],pred_train),seconds=elapsed,**resources)
    if result['phase']=='selection':
        result['selected_validation_macro_f1']=result['metrics']['macro_f1']
    write_json(directory/'result.json',result)
    write_json(directory/'history.json',[dict(fit_completed=True,no_Test_selection=True)])
    temporary=directory/'checkpoint.joblib.tmp'
    joblib.dump(dict(job_identity=inputs,model=fitted,normalizer=data['normalizer']),temporary)
    os.replace(temporary,directory/'checkpoint.joblib')
    write_json(directory/'COMPLETE.json',dict(job_id=job_id,identity_sha256=sha(root/'run_identity.json'),
        data_identity=data['identity'],outputs={p.name:sha(p) for p in directory.iterdir() if p.is_file() and p.name!='COMPLETE.json'}))
    neural.progress(root,job_id,state='completed',macro_f1=result['metrics']['macro_f1'],seconds=elapsed)
    print('CLASSIC_JOB_COMPLETE',job_id,result['metrics']['macro_f1'],elapsed,flush=True)
    return result


def classical_job(cfg,root,phase,protocol,fold,kind,suffix,data,index,seed):
    job_id=f'{phase}/{protocol}/{fold}/{kind}/{suffix}'
    directory=root/job_id
    if (directory/'COMPLETE.json').exists():
        verify(directory,root,job_id,data['identity'])
        return read_json(directory/'result.json')
    directory.mkdir(parents=True,exist_ok=True)
    inputs=dict(job_id=job_id,seed=seed,learning_rate=None,candidate_index=index,epochs_cap=1,
        data_identity=data['identity'],identity_sha256=sha(root/'run_identity.json'),
        method_lock_sha256=sha(root/'METHOD_LOCK.json') if phase=='formal' else None)
    if (directory/'INPUTS.json').exists():
        assert read_json(directory/'INPUTS.json')==inputs
    write_json(directory/'INPUTS.json',inputs)
    neural.progress(root,job_id,state='running',pid=os.getpid())
    start=time.monotonic()
    tr,ev=data['train_ids'],data['evaluation_ids']
    mask,prior=fallback(data,current=kind=='Stats-RF-CurrentOnly')
    if kind.startswith('Stats-RF'):
        feature_start=time.monotonic()
        features,names=descriptors(data['arrays'],data['normalizer'],kind.endswith('CurrentOnly'))
        feature_seconds=time.monotonic()-feature_start
        candidate=cfg['rf']['candidates'][index]
        clf=RandomForestClassifier(n_estimators=cfg['rf']['n_estimators'],class_weight=cfg['rf']['class_weight'],
            max_features=cfg['rf']['max_features'],n_jobs=cfg['threads'],random_state=seed,**candidate)
        fit_start=time.monotonic()
        clf.fit(features[tr],data['labels'][tr])
        fit_seconds=time.monotonic()-fit_start
        prediction=clf.predict(features)
        fitted=dict(classifier=clf,feature_names=names,current=kind.endswith('CurrentOnly'),prior_class=prior)
        extra=dict(feature_count=len(names),feature_seconds=feature_seconds,classifier_fit_seconds=fit_seconds,
                   hyperparameters=candidate)
    else:
        feature_start=time.monotonic()
        sequence=sequence_features(data['arrays'])
        parameters=rocket_fit(sequence[tr],seed,cfg['minirocket']['features_requested'])
        actual=len(parameters[-1])
        assert actual==cfg['minirocket']['features_actual']
        temporary=directory/'transformed.tmp.npy'
        transformed=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float32,shape=(len(sequence),actual))
        size=cfg['minirocket']['transform_batch_size']
        for at in range(0,len(sequence),size):
            transformed[at:at+size]=rocket.transform(sequence[at:at+size],parameters)
        transformed.flush()
        assert np.isfinite(transformed).all()
        feature_seconds=time.monotonic()-feature_start
        # Explicit TRAIN-only scaler. Selection evaluation never fits transform/scaler.
        scaler=StandardScaler().fit(transformed[tr])
        fit_start=time.monotonic()
        clf=RidgeClassifier(alpha=cfg['minirocket']['ridge_alphas'][index],solver='lsqr',
            tol=cfg['minirocket']['ridge_tol'],max_iter=cfg['minirocket']['ridge_max_iter'])
        clf.fit(scaler.transform(transformed[tr]),data['labels'][tr])
        if clf.n_iter_ is None or int(np.max(clf.n_iter_))>=cfg['minirocket']['ridge_max_iter']:
            raise RuntimeError('Ridge LSQR iteration limit: no automatic optimizer change')
        fit_seconds=time.monotonic()-fit_start
        prediction=np.concatenate([clf.predict(scaler.transform(transformed[at:at+size]))
                                   for at in range(0,len(sequence),size)])
        fitted=dict(parameters=parameters,scaler=scaler,classifier=clf,prior_class=prior,
                    vendor_sha256=sha('third_party/minirocket_20261012/minirocket_multivariate.py'))
        extra=dict(feature_count=actual,feature_seconds=feature_seconds,classifier_fit_seconds=fit_seconds,
                   hyperparameters=dict(alpha=cfg['minirocket']['ridge_alphas'][index]),lsqr_iterations=clf.n_iter_.tolist())
        del transformed,sequence
        # Only this known scratch file is removed, never checkpoints or failed logs.
        temporary.unlink()
    prediction[~mask]=prior
    return save_job(root,job_id,data,inputs,fitted,prediction[tr],prediction[ev],time.monotonic()-start,
        dict(max_process_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**extra))


def reuse_reference(cfg,root):
    parent=Path(cfg['reuse_architecture_root'])
    jobs={}
    for phase in ('selection','formal'):
        for protocol in cfg['protocols']:
            for fold in cfg['folds']:
                suffixes=['lr0','lr1'] if phase=='selection' else [f'seed{s}' for s in cfg['formal_seeds']]
                for suffix in suffixes:
                    directory=parent/phase/protocol/fold/'GRU'/suffix
                    c=read_json(directory/'COMPLETE.json')
                    assert c['job_id']==f'{phase}/{protocol}/{fold}/GRU/{suffix}'
                    assert c['identity_sha256']==cfg['reuse_architecture_identity_sha256']
                    for p,digest in c['outputs'].items():
                        assert sha(directory/p)==digest
                    jobs[c['job_id']]=dict(directory=str(directory),complete_sha256=sha(directory/'COMPLETE.json'))
    receipt=dict(reference_model=cfg['reference_model'],jobs=jobs,
        parent_method_lock_sha256=sha(parent/'METHOD_LOCK.json'),new_fits=0)
    if (root/'REUSED_GRU.json').exists():
        assert read_json(root/'REUSED_GRU.json')==receipt
    else:
        write_json(root/'REUSED_GRU.json',receipt)


def lock(cfg,root):
    selected,candidates={},[]
    for protocol in cfg['protocols']:
        for fold in cfg['folds']:
            data=data_for(cfg,'selection',protocol,fold)
            for kind in cfg['models']:
                results=[]
                for i in range(2):
                    job=f'selection/{protocol}/{fold}/{kind}/candidate{i}'
                    verify(root/job,root,job,data['identity'])
                    result=read_json(root/job/'result.json')
                    assert result['seed']==cfg['selection_seed']
                    results.append((i,result))
                    candidates.append(dict(job_id=job,complete_sha256=sha(root/job/'COMPLETE.json')))
                index,best=min(results,key=lambda item:(-item[1]['selected_validation_macro_f1'],item[1]['selected_epoch'],item[0]))
                selected[f'{protocol}/{fold}/{kind}']=dict(candidate_index=index,
                    learning_rate=cfg['learning_rates'][index] if kind=='CurrentRepeat-GRU-VAL' else None,
                    epochs=best['selected_epoch'],selection_job=best['job_id'],
                    validation_macro_f1=best['selected_validation_macro_f1'])
    payload=dict(identity_sha256=sha(root/'run_identity.json'),candidates=candidates,selected=selected,
        reference_receipt_sha256=sha(root/'REUSED_GRU.json'),all_selection_complete=True,
        historical_Test_exposure=True,new_formal_Test_predictions_before_lock=0)
    if (root/'METHOD_LOCK.json').exists():
        assert read_json(root/'METHOD_LOCK.json')==payload
    else:
        write_json(root/'METHOD_LOCK.json',payload)
    print('CLASSIC_ALL_METHODS_LOCKED',flush=True)


def run(cfg,root,phase):
    assert read_json(root/'SANITY.json')['status']=='PASS'
    recipe_lock=None
    if phase=='formal':
        recipe_lock=read_json(root/'METHOD_LOCK.json')
        assert recipe_lock['identity_sha256']==sha(root/'run_identity.json') and len(recipe_lock['candidates'])==80
        for candidate in recipe_lock['candidates']:
            assert sha(root/candidate['job_id']/'COMPLETE.json')==candidate['complete_sha256']
    original_factory,original_verify=neural.make_model,neural.verify_job
    neural.make_model=lambda kind,prior: CurrentRepeatGRU(prior)
    neural.verify_job=lambda directory,jobroot,data_identity: verify(directory,jobroot,str(directory.relative_to(jobroot)),data_identity)
    try:
        for protocol in cfg['protocols']:
            for fold in cfg['folds']:
                data=data_for(cfg,phase,protocol,fold)
                for kind in cfg['models']:
                    choices=[(i,cfg['selection_seed'],f'candidate{i}') for i in range(2)] if phase=='selection' else [
                        (recipe_lock['selected'][f'{protocol}/{fold}/{kind}']['candidate_index'],s,f'seed{s}') for s in cfg['formal_seeds']]
                    neural_data=None
                    if kind=='CurrentRepeat-GRU-VAL':
                        neural_data=neural.load_data(Path(data['identity']['directory']))
                    for index,seed,suffix in choices:
                        if neural_data is not None:
                            epochs=cfg['max_epochs'] if phase=='selection' else recipe_lock['selected'][f'{protocol}/{fold}/{kind}']['epochs']
                            neural.train_job(cfg,root,phase,protocol,fold,kind,suffix,neural_data,cfg['learning_rates'][index],seed,epochs)
                            verify(root/f'{phase}/{protocol}/{fold}/{kind}/{suffix}',root,
                                   f'{phase}/{protocol}/{fold}/{kind}/{suffix}',data['identity'])
                        else:
                            with threadpool_limits(limits=cfg['threads']):
                                classical_job(cfg,root,phase,protocol,fold,kind,suffix,data,index,seed)
                    del neural_data
                    gc.collect()
                    torch.cuda.empty_cache()
                del data
                gc.collect()
        write_json(root/f'{phase.upper()}_COMPLETE.json',dict(status='COMPLETE',
            expected_jobs=80 if phase=='selection' else 120,identity_sha256=sha(root/'run_identity.json')))
    finally:
        neural.make_model,neural.verify_job=original_factory,original_verify


def sanity(cfg,root):
    raw_receipt=read_json(Path(cfg['source'])/'source_receipt.json')
    partial_counts={}
    for cow in range(1,11):
        path=Path(cfg['source'])/f'main_data/uwb/T{cow:02d}/T{cow:02d}_0725.csv'
        member=f'sensor_data/main_data/uwb/T{cow:02d}/T{cow:02d}_0725.csv'
        record=next(r for r in raw_receipt['members'] if r['member']==member)
        assert sha(path)==record['sha256']
        raw=pd.read_csv(path)
        valid=np.isfinite(raw[['coord_x_cm','coord_y_cm','coord_z_cm']].to_numpy())
        partial_counts[str(cow)]=int((valid.any(1)&~valid.all(1)).sum())
        assert partial_counts[str(cow)]==0, 'Native reconstruction needs synchronous UWB field sources'
    data=data_for(cfg,'selection','TS','fold_1')
    chosen=data['train_ids'][:32]
    arrays={m:{k:v[chosen] for k,v in fields.items()} for m,fields in data['arrays'].items()}
    window,names=descriptors(arrays,data['normalizer'],False)
    current,names_current=descriptors(arrays,data['normalizer'],True)
    assert names==names_current and window.shape==current.shape==(32,110)
    altered={m:{k:v.copy() for k,v in fields.items()} for m,fields in arrays.items()}
    for fields in altered.values():
        fields['x'][:,:-1]+=999
        fields['valid'][:,:-1]=False
        fields['fresh'][:,:-1]=0
        fields['age'][:,:-1]=np.nan
    current_after,_=descriptors(altered,data['normalizer'],True)
    np.testing.assert_array_equal(current,current_after)
    assert np.all(current[:,-7:-1]==0) and np.all(current[:,-1]==1)
    batch={m:{k:torch.from_numpy(v).cuda() for k,v in fields.items()} for m,fields in arrays.items()}
    neural.seed_all(17)
    model=CurrentRepeatGRU([1/7]*7).cuda()
    assert sum(p.numel() for p in model.parameters())==86695
    output=model(batch)
    assert torch.isfinite(output).all()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    for _ in range(3):
        optimizer.zero_grad()
        loss=torch.nn.functional.cross_entropy(model(batch),torch.arange(32,device='cuda')%7)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    reuse_reference(cfg,root)
    write_json(root/'FEATURE_DICTIONARY.json',dict(names=names,width=110,
        UWB_source_reconstruction='grid_minus_oldest_component_age_fully_valid_constant_position_groups',
        no_extra_raw_source_observations=True,current_repeat_invariant_to_past=True))
    write_json(root/'SANITY.json',dict(status='PASS',synthetic_only_optimizer_steps=3,
        descriptors_no_GT=True,descriptor_width=110,current_past_invariant=True,
        raw_UWB_partial_field_records=partial_counts,
        GRU_parameters=86695,identity_sha256=sha(root/'run_identity.json')))
    print('CLASSIC_SANITY_PASS',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['sanity','selection','lock','formal'])
    args=parser.parse_args()
    cfg,root=setup()
    with (root/'.RUNNING.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if args.phase=='sanity': sanity(cfg,root)
            elif args.phase=='lock': lock(cfg,root)
            else: run(cfg,root,args.phase)
        except Exception:
            trace=traceback.format_exc()
            write_json(root/f'FAILURE_{time.time_ns()}.json',dict(phase=args.phase,traceback=trace))
            if (root/'QUEUE_STATE.json').exists():
                current=read_json(root/'QUEUE_STATE.json')['current_job']
                if current['state']=='running': neural.progress(root,current['job_id'],state='failed',traceback=trace)
            raise


if __name__=='__main__':
    main()
