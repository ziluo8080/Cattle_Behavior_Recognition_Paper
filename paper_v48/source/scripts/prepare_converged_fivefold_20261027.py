"""Reuse hash-bound official caches; never open Test targets during training."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from scripts.converged_fivefold_common_20261027 import config,read,save,sha,data_dir,FIELDS
from scripts.sensor_native_immu_training_20261021 import load_selection,sha as source_sha
from q3_sensor.protocol import MODALITIES
from q3_sensor.endpoint_models import tokens
from experiments.sensor_comparator_pilot_20261022_v2.cpu import summary_features


def prepare(cfg,p,f):
    out=data_dir(cfg,p,f);assert not (out/'RECEIPT.json').exists();out.mkdir(parents=True,exist_ok=True)
    raw=load_selection(Path.cwd(),p,f)
    parent=Path(cfg['selection_state_root'])/p/f;r=read(parent/'DATA_RECEIPT.json');assert r['phase']=='selection'
    for n,digest in r['outputs'].items():assert sha(parent/n)==digest,n
    targets=parent/'targets.jsonl';native_targets=Path(cfg['selection_native_root'])/p/f/'targets.jsonl'
    assert targets.read_bytes()==native_targets.read_bytes()
    rows=[json.loads(s) for s in targets.read_text().splitlines()];assert {x['split'] for x in rows}=={'train','val'}
    parts=[];valid=[]
    with np.load(parent/'windows.npz',allow_pickle=False) as z:
        for m in MODALITIES:
            a={k:torch.from_numpy(np.array(z[f'{m}_{k}'],copy=True)) for k in ['x','valid','fresh','age']}
            a['valid']=a['valid'].bool();a['x']=torch.where(a['valid'],a['x'],0.);parts.append(tokens(a).numpy());valid.append(a['valid'].any(-1).numpy())
    state=np.concatenate(parts,-1).astype(np.float32);values=dict(dynamic=raw['native'],context=raw['context'],state=state,
        state_valid=np.stack(valid,-1).any(-1),current=state[:,-1].copy())
    values['features']=summary_features(dict(context=values['context'],current=values['current'],native=values['dynamic'],state=state))
    fit,dev=raw['train'],raw['evaluation'];labels=raw['labels'];assert len(labels)==len(rows)
    counts=np.bincount(labels[fit],minlength=7);assert (counts>0).all()
    for key in ['dynamic','context','state','current','features']:assert np.isfinite(values[key]).all()
    np.savez(out/'FIT.npz',**{k:v[fit] for k,v in values.items()},labels=labels[fit])
    np.savez(out/'DEV_INPUTS.npz',**{k:v[dev] for k,v in values.items()})
    for name,ids in [('FIT_TARGETS.jsonl',fit),('DEV_TARGETS.jsonl',dev)]:
        (out/name).write_text(''.join(json.dumps(rows[i])+'\n' for i in ids),encoding='utf-8')
    formal=Path(cfg['formal_data_root'])/p/f;fr=read(formal/'DATA_RECEIPT.json')
    assert fr['phase']=='formal_refit' and not fr['TEST_INPUTS_has_labels']
    assert sha(formal/'FIT.npz')==fr['outputs']['FIT.npz']
    assert sha(formal/'FIT_TARGETS.jsonl')==fr['outputs']['FIT_TARGETS.jsonl']
    fit_rows=[json.loads(s) for s in (formal/'FIT_TARGETS.jsonl').read_text().splitlines()]
    key=lambda r:(r['cow_id'],r['timestamp'],r['label'])
    assert sorted(map(key,rows))==sorted(map(key,fit_rows)), 'Selection TRAIN+VAL must equal final fit pool'
    save(out/'REFIT_SOURCE.json',dict(protocol=p,fold=f,FIT_path=str(formal/'FIT.npz'),FIT_sha256=fr['outputs']['FIT.npz'],
        FIT_targets_path=str(formal/'FIT_TARGETS.jsonl'),FIT_targets_sha256=fr['outputs']['FIT_TARGETS.jsonl'],
        TEST_INPUTS_path=str(formal/'TEST_INPUTS.npz'),TEST_INPUTS_sha256=fr['outputs']['TEST_INPUTS.npz'],
        TEST_TARGETS_path=str(formal/'TEST_TARGETS.jsonl'),TEST_TARGETS_sha256=fr['outputs']['TEST_TARGETS.jsonl'],
        fit_rows=fr['fit_targets'],test_rows=fr['test_targets'],Test_targets_opened=False,
        normalizer_scope='Original TRAIN+VAL fit-only normalizers; separate from TRAIN-only calibration normalizers'))
    names=['FIT.npz','DEV_INPUTS.npz','FIT_TARGETS.jsonl','DEV_TARGETS.jsonl','REFIT_SOURCE.json']
    receipt=dict(status='PASS',protocol=p,fold=f,fit_rows=len(fit),dev_rows=len(dev),final_fit_rows=len(fit_rows),
        fit_class_counts=counts.tolist(),calibration_scope='Official TRAIN fit; official VAL convergence; TRAIN-only preprocessing',
        selection_native_receipt_sha256=raw['receipt_sha256'],selection_state_receipt_sha256=sha(parent/'DATA_RECEIPT.json'),
        Test_targets_opened=False,selection_and_final_fit_ID_union_exact=True,outputs={n:sha(out/n) for n in names})
    save(out/'RECEIPT.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',choices=['TS','OS']);parser.add_argument('--fold');args=parser.parse_args();cfg=config()
    if args.protocol:prepare(cfg,args.protocol,args.fold)
    else:
        for p in cfg['protocols']:
            for f in cfg['folds']:prepare(cfg,p,f)
