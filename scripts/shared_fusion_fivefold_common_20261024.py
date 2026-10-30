"""Fixed method, source identities and runtime for the authorized fivefold refit."""
import hashlib
import json
import time
from pathlib import Path

CONFIG=Path('configs/sensor_shared_fusion_fivefold_20261024_v1.json')
ROOT=Path('/q3/mmcows-sensor-behavior-20261007')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)


def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):d.update(block)
    return d.hexdigest()


def configure(cuda=False):
    import numpy as np
    import torch,xgboost
    if Path.cwd()!=ROOT:raise RuntimeError('Formal scientific work is cloud-only')
    assert (torch.__version__,np.__version__,xgboost.__version__)==('2.4.1+cu121','1.24.4','2.1.4')
    if cuda:
        from scripts.run_convtran_context_20261022 import configure as existing_runtime
        existing_runtime()
    else:torch.set_num_threads(4)
    cfg=read(CONFIG)
    assert cfg['protocols']==['TS','OS'] and cfg['folds']==[f'fold_{i}' for i in range(1,6)]
    assert cfg['seeds']==[17,23,29] and cfg['epochs']==20 and cfg['primary_view']=='ema20'
    assert cfg['Test_allowed'] and cfg['formal_refit_allowed'] and cfg['additional_folds_allowed']
    assert not cfg['Test_selects_model_or_parameters'] and not cfg['early_stopping'] and not cfg['hyperparameter_search']
    recipe=read(cfg['incumbent_config'])['shared_recipe'];g=cfg['gate']
    assert recipe['view']=='ema'
    for k in ('family','XGB_temperature','B_temperature','XGB_weight'):assert g[k]==recipe[k]
    old=read(cfg['B_config'])
    for k in ('width','heads','ff','dropout','epochs','learning_rate','weight_decay','batch_size','ema_decay','ema_warmup_steps'):
        assert cfg[k]==old[k],k
    old_x=read(cfg['XGB_config'])['xgboost']
    for k in ('rounds','max_depth','eta','tree_method','device','nthread'):assert cfg['xgboost'][k]==old_x[k],k
    assert cfg['xgboost']['subsample']==cfg['xgboost']['colsample_bytree']==1. and cfg['xgboost']['seed']==17
    target=read(cfg['acceptance_targets']);assert sha(cfg['benchmark'])==target['frozen_benchmark_sha256']
    return cfg


def code_files(cfg):
    paths={CONFIG,Path(cfg['plan']),Path(cfg['environment']),Path(cfg['incumbent_config']),Path(cfg['B_config']),Path(cfg['XGB_config']),Path(cfg['acceptance_targets']),Path('configs/sensor_convtran_context_20261022_v1.json')}
    paths.update(Path('scripts').glob('*shared_fusion_fivefold*20261024.py'))
    paths.update(Path('tests').glob('*shared_fusion_fivefold*20261024.py'))
    for name in ('run_convtran_context_20261022','sensor_convtran_context_model_20261022',
        'sensor_convtran_endpoint_interaction_20261022','sensor_native_convtran_20261021',
        'sensor_native_immu_data_20261021','sensor_uwb_event_motion_20261020',
        'sensor_classic_features_20261012','sensor_xgb_b_fusion_20261022','report_convtran_context_20261022',
        'run_sensor_formal_ts','sensor_os_protocol','prepare_sensor_native_immu_20261021',
        'sensor_native_immu_training_20261021','sensor_native_immu_runner_20261021','sensor_native_immu_selection_policy_20261021'):
        paths.add(Path('scripts')/(name+'.py'))
    paths.add(Path('experiments/sensor_comparator_pilot_20261022_v2/cpu.py'))
    paths.update(Path('q3_sensor').glob('*.py'))
    paths.update(Path('third_party/convtran_20261021_v1/Models').glob('*.py'))
    return {p.as_posix():sha(p) for p in sorted(paths)}


def source_files(cfg):
    benchmark=read(cfg['benchmark']);known=benchmark['sources']
    paths={Path(cfg['benchmark']),Path('configs/official_ts.json'),Path('configs/official_os.json'),Path('configs/sensor_os_20261007.json')}
    native=Path(cfg['native_root']);inventory=native/'FULL_DAY_TRAIN_INVENTORY.json'
    inv=read(inventory);assert inv['status']=='FULL_DAY_TRAIN_INVENTORY_COMPLETE';paths.add(inventory)
    assert {r['cow_id'] for r in inv['cows']}==set(range(1,11))
    for record in inv['cows']:
        p=native/'raw_os_train_union'/f"T{record['cow_id']:02d}_0725.csv"
        assert sha(p)==record['cached_csv_sha256'];paths.add(p)
    for protocol in cfg['protocols']:
        for fold in cfg['folds']:
            directory=Path(cfg['state_cache'])/protocol/fold;receipt=directory/'DATA_RECEIPT.json'
            assert sha(receipt)==known[receipt.as_posix()];r=read(receipt)
            assert r['phase']=='formal' and r['normalizer_scope']=='TRAIN_plus_VAL'
            assert r['no_source_after_grid'] and r['source_within_own_role'];paths.add(receipt)
            for name,digest in r['outputs'].items():
                p=directory/name;assert sha(p)==digest;paths.add(p)
            assert sha(directory/'targets.jsonl')==known[(directory/'targets.jsonl').as_posix()]
            for seed in cfg['seeds']:
                historical=Path(cfg['historical_GRU'])/protocol/fold/'GRU'/f'seed{seed}'
                for name in ('COMPLETE.json','result.json','predictions.csv'):
                    p=historical/name;assert sha(p)==known[p.as_posix()];paths.add(p)
    return {p.as_posix():sha(p) for p in sorted(paths)}


def make_method_lock():
    cfg=configure();out=Path(cfg['output']);assert not (out/'METHOD_LOCK.json').exists()
    assert read(out/'CODE_REVIEW_GATE.json')['status']=='PASS'
    # Freeze the scientific method before CPU-only input preparation. The
    # original runtime is reused; a fresh CUDA witness is required again by the
    # training seal, once the already-running shared-GPU queue has finished.
    for name in ('INDEPENDENT_RUNTIME_WITNESS.json',):
        r=read(out/name);assert r['status']=='PASS'
        for p,h in r['implementation_files'].items():assert sha(p)==h
    save(out/'METHOD_LOCK.json',dict(status='PASS',config=cfg,code=code_files(cfg),sources=source_files(cfg),
        created_unix=time.time(),new_model_Test_predictions=0,new_model_Test_scores=0,
        user_authorization=cfg['user_authorization'],historical_Test_exposure=True,new_blind_evaluation=False,
        fresh_CUDA_witness_required_before_training_seal=True))
    print('SHARED_FUSION_FIVEFOLD_METHOD_LOCK_PASS',flush=True)


def check_method_lock(cfg,full_sources=False):
    assert Path.cwd()==ROOT
    lock=read(Path(cfg['output'])/'METHOD_LOCK.json');assert lock['config']==cfg
    for p,h in lock['code'].items():assert sha(p)==h,p
    if full_sources:
        for p,h in lock['sources'].items():assert sha(p)==h,p
    return lock


def case_dir(cfg,protocol,fold):
    assert protocol in cfg['protocols'] and fold in cfg['folds']
    return Path(cfg['output'])/'data'/protocol/fold


def check_data(cfg,protocol,fold):
    directory=case_dir(cfg,protocol,fold);r=read(directory/'DATA_RECEIPT.json')
    assert (r['protocol'],r['fold'],r['phase'])==(protocol,fold,'formal_refit')
    for name,h in r['outputs'].items():assert sha(directory/name)==h,(protocol,fold,name)
    return r


if __name__=='__main__':make_method_lock()
