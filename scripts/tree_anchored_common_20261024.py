"""Fixed scope and existing source identities for the TRAIN-internal pilot."""
from pathlib import Path
import json
import time

from scripts.sensor_learned_fusion_data_20261023 import read,save,sha,make_splits

ROOT=Path('/q3/mmcows-sensor-behavior-20261007')
CONFIG=Path('configs/sensor_tree_anchored_dynamic_residual_20261024_v1.json')


def configure(cuda=False):
    import torch,numpy as np,xgboost as xgb
    assert Path.cwd()==ROOT,'Scientific preparation/fitting is cloud-only'
    assert (torch.__version__,np.__version__,xgb.__version__)==('2.4.1+cu121','1.24.4','2.1.4')
    if cuda:
        from scripts.run_convtran_context_20261022 import configure as original_runtime
        original_runtime()
    else:torch.set_num_threads(4)
    cfg=read(CONFIG)
    assert cfg['protocols']==['TS','OS'] and cfg['parent_inner']==2 and cfg['nested_folds']==2
    assert cfg['seeds']==[17,23,29] and cfg['epochs']==20 and cfg['neural_trajectories']==18
    assert cfg['arms']==['ZeroHead_B','TreeResidual_CE','TreeResidual_KeepCorrect']
    assert cfg['tree_temperature']==2. and cfg['keep_correct_lambda']==.1
    assert all(cfg[k] is False for k in ('Test_allowed','outer_VAL_allowed','formal_refit_allowed','automatic_expansion','early_stopping','hyperparameter_search','additional_parent_inner_folds_allowed'))
    old=read(cfg['B_config'])
    for key in ['width','heads','ff','dropout','epochs','learning_rate','weight_decay','batch_size','ema_decay','ema_warmup_steps']:assert cfg[key]==old[key],key
    return cfg


def code_files(cfg):
    paths={CONFIG,Path(cfg['proposal']),Path('refine-logs/sensor_tree_anchored_dynamic_residual_20261024_v1/PROBLEM_ANCHOR.md'),
        Path('refine-logs/sensor_tree_anchored_dynamic_residual_20261024_v1/EXPERIMENT_PLAN.md'),
        Path('configs/sensor_tree_anchored_env_20261024_v1.json'),Path('.aris/compute/bitahub-root-tree-anchored-20261024.md'),
        Path(cfg['B_config']),Path(cfg['incumbent_config'])}
    paths.update(Path('scripts').glob('*tree_anchored*20261024.py'))
    paths.update(Path('tests').glob('*tree_anchored*20261024.py'))
    for name in ['sensor_learned_fusion_data_20261023','run_convtran_context_20261022','sensor_convtran_context_model_20261022',
                 'sensor_convtran_endpoint_interaction_20261022','sensor_uwb_event_motion_20261020','sensor_native_immu_data_20261021',
                 'sensor_xgb_b_fusion_20261022','sensor_classic_features_20261012','report_convtran_context_20261022']:
        paths.add(Path('scripts')/(name+'.py'))
    paths.update(Path('q3_sensor').glob('*.py'))
    paths.add(Path('experiments/sensor_comparator_pilot_20261022_v2/cpu.py'))
    return {str(p):sha(p) for p in sorted(paths)}


def source_files(cfg):
    seal=read(cfg['parent_seal']);sources={cfg['parent_seal']:sha(cfg['parent_seal'])}
    for protocol in cfg['protocols']:
        target=Path(cfg['outer_source'])/'data'/protocol/'targets.jsonl'
        assert sha(target)==seal['files'][str(target)];sources[str(target)]=sha(target)
        cache=Path(cfg['parent_cache'])/protocol/'inner2';receipt=cache/'DATA_RECEIPT.json'
        assert sha(receipt)==seal['files'][str(receipt)];sources[str(receipt)]=sha(receipt)
        r=read(receipt)
        for name,digest in r['outputs'].items():
            p=cache/name;assert sha(p)==digest; sources[str(p)]=digest
        for name,record in r['source_files'].items():
            assert sha(name)==record['sha256'];sources[name]=record['sha256']
        expert=Path(cfg['parent_experts'])/protocol/'inner2'
        names=['COMPLETE.json','XGB_seed17.json','XGB_probabilities.npy','XGB_RECEIPT.json','held_original_ids.npy','held_truth.npy']
        names += [f'B_seed{s}_latest.pt' for s in cfg['seeds']]+[f'B_ema_{s}_probabilities.npy' for s in cfg['seeds']]
        for name in names:
            p=expert/name;sources[str(p)]=sha(p)
    return sources


def parent_case(cfg,protocol):
    import numpy as np
    rows=[json.loads(line) for line in (Path(cfg['outer_source'])/'data'/protocol/'targets.jsonl').read_text().splitlines()]
    split=make_splits(rows,protocol,k=3)[cfg['parent_inner']]
    directory=Path(cfg['parent_cache'])/protocol/'inner2';receipt=read(directory/'DATA_RECEIPT.json')
    for name,digest in receipt['outputs'].items():assert sha(directory/name)==digest,name
    with np.load(directory/'DATA.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    for name in ['train','held','original_ids']:np.testing.assert_array_equal(data[name],split[name])
    np.testing.assert_array_equal(data['labels'],[r['label']-1 for r in split['rows']])
    fit_rows=[dict(split['rows'][int(i)],split='train') for i in split['train']]
    return data,split,fit_rows


def make_method_lock(cfg):
    out=Path(cfg['output']);assert not (out/'METHOD_LOCK.json').exists()
    assert read(out/'CODE_REVIEW_GATE.json')['status']=='PASS'
    assert read(out/'INDEPENDENT_RUNTIME_WITNESS.json')['status']=='PASS'
    for name,digest in read(out/'INDEPENDENT_RUNTIME_WITNESS.json')['implementation_files'].items():assert sha(name)==digest,name
    save(out/'METHOD_LOCK.json',dict(status='PASS',config=cfg,code=code_files(cfg),sources=source_files(cfg),created_unix=time.time(),
        Test_scores_computed=0,new_candidate_scores_computed=0))


def check_method_lock(cfg,full_sources=False):
    lock=read(Path(cfg['output'])/'METHOD_LOCK.json');assert lock['config']==cfg
    for name,digest in lock['code'].items():assert sha(name)==digest,name
    if full_sources:
        for name,digest in lock['sources'].items():assert sha(name)==digest,name
    return lock


def check_receipt(directory):
    directory=Path(directory);receipt=read(directory/'DATA_RECEIPT.json')
    for name,digest in receipt['outputs'].items():assert sha(directory/name)==digest,(directory,name)
    return receipt
