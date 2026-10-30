"""Original-target formal input reconstruction, with Fit-only statistics."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np

from scripts.shared_fusion_fivefold_common_20261024 import configure,read,save,sha,case_dir,check_method_lock,check_data


def fit_roles_and_all_roles(cfg,protocol,fold):
    official=read('configs/official_'+protocol.lower()+'.json');f=official['folds'][fold]
    if protocol=='TS':
        from scripts.run_sensor_formal_ts import official_roles
        segments=official_roles(official,f);roles={c:segments for c in range(1,11)}
    else:
        from scripts.sensor_os_protocol import cow_roles,day_role
        groups,fit_cows,test_cows=cow_roles(f);day=day_role(read('configs/sensor_os_20261007.json'))
        roles={c:[dict(day,split='train' if c in fit_cows else 'test')] for c in range(1,11)}
    fit={c:[r for r in rs if r['split']=='train'] for c,rs in roles.items()}
    return {c:rs for c,rs in fit.items() if rs},roles


def validate_targets(rows,roles):
    assert rows and {r['split'] for r in rows}=={'train','test'}
    keys=[(int(r['cow_id']),int(r['timestamp'])) for r in rows];assert len(keys)==len(set(keys))
    for row in rows:
        assert row['timestamp']==int(row['timestamp']) and row['label'] in range(1,8)
        matches=[r for r in roles[int(row['cow_id'])] if r['start']<=row['timestamp']<r['end']]
        assert len(matches)==1
        role=matches[0]
        assert (row['split'],row['start'],row['end'])==(role['split'],role['start'],role['end'])


def native_windows(rows,sources,scaler):
    """The frozen native window rule, explicitly extended to authorized Test roles."""
    output=np.zeros((len(rows),100,8),np.float32);output[:,:,7]=.2
    provenance=np.full((len(rows),100),-1,np.int64)
    mean,std=np.asarray(scaler['mean']),np.asarray(scaler['std']);groups={}
    for i,r in enumerate(rows):
        if r['split'] not in ('train','test'):raise ValueError('Formal Fit/Test roles required')
        groups.setdefault((r['cow_id'],r['start'],r['end'],r['split']),[]).append(i)
    for (cow,start,end,role),ids in groups.items():
        ticks,values=sources[cow]
        keep=(ticks>=start*10)&(ticks<end*10)&np.isfinite(values).all(1)
        ticks,values=ticks[keep],values[keep]
        target=np.asarray([rows[i]['timestamp'] for i in ids],np.int64)*10
        assert ((target>=start*10)&(target<end*10)).all()
        query=target[:,None]+np.arange(-99,1)[None,:]
        if not len(ticks):continue
        pos=np.searchsorted(ticks,query,side='right')-1;safe=np.maximum(pos,0);picked=ticks[safe];age=query-picked
        valid=(pos>=0)&(age>=0)&(age<=2)&(picked>target[:,None]-100)
        physical=(values[safe]-mean)/std
        output[ids,:,:6]=np.where(valid[:,:,None],physical,0.)
        output[ids,:,6]=valid;output[ids,:,7]=np.where(valid,age/10,.2)
        provenance[ids]=np.where(valid,picked,-1)
        assert (picked[valid]<=query[valid]).all() and (picked[valid]>=start*10).all() and (picked[valid]<end*10).all()
    assert np.isfinite(output).all()
    return output,provenance


def normalize_context(unscaled,fit_ids):
    unscaled=np.asarray(unscaled,dtype=np.float32)
    values=unscaled[fit_ids].astype(np.float64);mean=values.mean(0);raw_std=values.std(0)
    std=np.where(raw_std<=1e-6,1.,raw_std)
    transformed=((unscaled-mean)/std).astype(np.float32)
    assert np.isfinite(transformed).all()
    return transformed,dict(mean=mean.tolist(),std=std.tolist(),raw_std=raw_std.tolist(),scope='Fit TRAIN+VAL target descriptors only')


def state_and_context(arrays,normalizer,fit_ids):
    import torch
    from q3_sensor.protocol import MODALITIES
    from q3_sensor.endpoint_models import tokens
    from scripts.sensor_classic_features_20261012 import descriptors
    desc,names=descriptors(arrays,normalizer)
    parts=[];masks=[]
    for modality in MODALITIES:
        fields={k:torch.from_numpy(np.array(arrays[modality][k],copy=True)) for k in ('x','valid','fresh','age')}
        fields['valid']=fields['valid'].bool();fields['x']=torch.where(fields['valid'],fields['x'],0.)
        parts.append(tokens(fields).numpy());masks.append(fields['valid'].any(-1).numpy())
    state=np.concatenate(parts,-1).astype(np.float32);current=state[:,-1].copy()
    context,scaler=normalize_context(np.concatenate([desc,current],-1),fit_ids)
    assert context.shape[1]==136 and state.shape[1:]==(60,26)
    return dict(state=state,state_valid=np.stack(masks,-1).any(-1),current=current,context=context),scaler


def native_sources(cfg):
    import pandas as pd
    from scripts.sensor_native_immu_data_20261021 import CHANNELS,canonical_source
    root=Path(cfg['native_root']);inventory=read(root/'FULL_DAY_TRAIN_INVENTORY.json');sources={}
    for record in inventory['cows']:
        c=int(record['cow_id']);p=root/'raw_os_train_union'/f'T{c:02d}_0725.csv'
        assert sha(p)==record['cached_csv_sha256']
        frame=pd.read_csv(p,usecols=['timestamp']+list(CHANNELS));sources[c]=canonical_source(frame)
    assert set(sources)==set(range(1,11))
    return sources


def prepare_case(cfg,protocol,fold,sources):
    import pandas as pd
    from q3_sensor.protocol import MODALITIES
    from scripts.sensor_native_immu_data_20261021 import fit_scaler
    from experiments.sensor_comparator_pilot_20261022_v2.cpu import summary_features
    out=case_dir(cfg,protocol,fold);out.mkdir(parents=True,exist_ok=True)
    if (out/'DATA_RECEIPT.json').exists():return check_data(cfg,protocol,fold)
    assert not (out/'FIT.npz').exists(),'Partial preparation must be inspected before retry'
    parent=Path(cfg['state_cache'])/protocol/fold;receipt=read(parent/'DATA_RECEIPT.json')
    assert receipt['phase']=='formal'
    for name,digest in receipt['outputs'].items():assert sha(parent/name)==digest
    rows=[json.loads(s) for s in (parent/'targets.jsonl').read_text().splitlines()]
    fit_roles,roles=fit_roles_and_all_roles(cfg,protocol,fold);validate_targets(rows,roles)
    fit=np.flatnonzero([r['split']=='train' for r in rows]);test=np.flatnonzero([r['split']=='test' for r in rows])
    assert len(fit)==receipt['train'] and len(test)==receipt['evaluation']
    with np.load(parent/'windows.npz',allow_pickle=False) as z:
        arrays={m:{k:z[f'{m}_{k}'] for k in ('x','valid','fresh','age')} for m in MODALITIES}
    normalizer=read(parent/'normalizer.json');data,context_norm=state_and_context(arrays,normalizer,fit)
    native_norm=fit_scaler(sources,fit_roles);dynamic,provenance=native_windows(rows,sources,native_norm)
    data['dynamic']=dynamic
    features=summary_features(dict(context=data['context'],current=data['current'],native=dynamic,state=data['state']))
    assert features.shape==(len(rows),866);data['features']=features
    fit_y=np.asarray([rows[i]['label']-1 for i in fit],np.int64);counts=np.bincount(fit_y,minlength=7)
    assert (counts>0).all();prior=counts/counts.sum();weights=np.sqrt(counts.sum()/(7*counts))
    test_rows=[rows[i] for i in test];fit_rows=[rows[i] for i in fit]
    historical=Path(cfg['historical_GRU'])/protocol/fold/'GRU'/'seed17'/'predictions.csv'
    old_ids=pd.read_csv(historical,usecols=['cow_id','timestamp','label'])
    np.testing.assert_array_equal(old_ids.to_numpy(),np.asarray([[r[k] for k in ('cow_id','timestamp','label')] for r in test_rows]))
    np.savez(out/'FIT.npz',**{k:v[fit] for k,v in data.items()},labels=fit_y,prior=prior,class_weights=weights)
    np.savez(out/'TEST_INPUTS.npz',**{k:v[test] for k,v in data.items()},prior=prior)
    for name,records in [('FIT_TARGETS.jsonl',fit_rows),('TEST_TARGETS.jsonl',test_rows)]:
        (out/name).write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in records),encoding='utf-8')
    np.save(out/'NATIVE_SOURCE_TICKS.npy',provenance)
    save(out/'NORMALIZERS.json',dict(state=normalizer,native=native_norm,context=context_norm,
        fit_source_roles=fit_roles,fit_scope=cfg['fit_scope'],Test_used_in_normalizers=False))
    outputs=['FIT.npz','TEST_INPUTS.npz','FIT_TARGETS.jsonl','TEST_TARGETS.jsonl','NATIVE_SOURCE_TICKS.npy','NORMALIZERS.json']
    r=dict(status='PASS',phase='formal_refit',protocol=protocol,fold=fold,fit_targets=len(fit),test_targets=len(test),
        fit_class_counts=counts.tolist(),parent_receipt_sha256=sha(parent/'DATA_RECEIPT.json'),
        parent_target_sha256=sha(parent/'targets.jsonl'),native_channels=8,statistics_features=866,
        TEST_INPUTS_has_labels=False,Test_identical_to_historical_GRU=True,role_bounded_past_sources=True,
        native_max_hold_seconds=.2,normalizer_fit_scope=cfg['fit_scope'],new_model_Test_scores=0,
        outputs={name:sha(out/name) for name in outputs})
    save(out/'DATA_RECEIPT.json',r);print(json.dumps({k:r[k] for k in ('protocol','fold','fit_targets','test_targets','status')}),flush=True)
    return r


def coverage(cfg):
    universes={};records=[]
    for protocol in cfg['protocols']:
        all_test=[];reference=None
        for fold in cfg['folds']:
            directory=case_dir(cfg,protocol,fold);r=check_data(cfg,protocol,fold)
            fit=[json.loads(x) for x in (directory/'FIT_TARGETS.jsonl').read_text().splitlines()]
            test=[json.loads(x) for x in (directory/'TEST_TARGETS.jsonl').read_text().splitlines()]
            f={(x['cow_id'],x['timestamp']) for x in fit};t={(x['cow_id'],x['timestamp']) for x in test}
            assert not f&t and len(f)==len(fit) and len(t)==len(test)
            universe=f|t
            if reference is None:reference=universe
            else:assert universe==reference
            all_test.extend(t);records.append(dict(protocol=protocol,fold=fold,fit=len(f),test=len(t)))
        assert len(all_test)==len(set(all_test))==len(reference)==46821
        assert set(all_test)==reference;universes[protocol]=reference
    assert universes['TS']==universes['OS']
    return dict(status='PASS',records=records,unique_Test_targets_per_protocol=46821,
        each_target_tested_once=True,TS_OS_target_universe_identical=True,Test_model_scores_computed=0)


def development_replay(cfg,sources):
    from scripts.prepare_sensor_native_immu_20261021 import fold_roles
    from scripts.sensor_native_immu_data_20261021 import fit_scaler
    from experiments.sensor_comparator_pilot_20261022_v2.cpu import summary_features
    records=[]
    for protocol in cfg['protocols']:
        parent=Path('reports/sensor_architecture_comparison_20261011/data/selection')/protocol/'fold_1'
        old=Path('reports/sensor_native_immu_preflight_20261021_v2/data')/protocol/'fold_1'
        receipt=read(old/'DATA_RECEIPT.json')
        for name,digest in receipt['outputs'].items():assert sha(old/name)==digest
        assert (parent/'targets.jsonl').read_bytes()==(old/'targets.jsonl').read_bytes()
        rows=[json.loads(x) for x in (old/'targets.jsonl').read_text().splitlines()]
        fit=np.flatnonzero([r['split']=='train' for r in rows]);assert {r['split'] for r in rows}=={'train','val'}
        roles=fold_roles(protocol,'fold_1');fit_roles={c:[r for r in rs if r['split']=='train'] for c,rs in roles.items()}
        fit_roles={c:rs for c,rs in fit_roles.items() if rs};scaler=fit_scaler(sources,fit_roles)
        # Only a synthetic role-name translation for legacy array equivalence;
        # these are the existing development TRAIN/VAL rows, not formal Test.
        translated=[dict(r,split='test' if r['split']=='val' else 'train') for r in rows]
        dynamic,provenance=native_windows(translated,sources,scaler)
        native_parent=Path(receipt['native_inputs_parent']);assert sha(native_parent)==receipt['native_inputs_parent_sha256']
        with np.load(native_parent,allow_pickle=False) as z:
            np.testing.assert_array_equal(dynamic,z['native']);np.testing.assert_array_equal(provenance,z['native_source_tick'])
        with np.load(parent/'windows.npz',allow_pickle=False) as z:
            arrays={m:{k:z[f'{m}_{k}'] for k in ('x','valid','fresh','age')} for m in ('uwb','hd','akl')}
        data,context_norm=state_and_context(arrays,read(parent/'normalizer.json'),fit)
        expected=np.load(old/'context.npy',allow_pickle=False)
        # Frozen before execution: v2's two-stage float32 scaling can differ by
        # floating-point roundoff from the algebraically equivalent direct form.
        np.testing.assert_allclose(data['context'],expected,rtol=1e-6,atol=1e-6)
        features=summary_features(dict(context=data['context'],current=data['current'],native=dynamic,state=data['state']))
        base=Path('reports/sensor_tree_expert_upgrade_20261022_v1/feature_cache')/protocol
        val=np.flatnonzero([r['split']=='val' for r in rows])
        for ids,name in ((fit,'TRAIN.npz'),(val,'VAL_INPUTS.npz')):
            with np.load(base/name,allow_pickle=False) as z:np.testing.assert_allclose(features[ids],z['features'],rtol=1e-6,atol=1e-6)
        records.append(dict(protocol=protocol,fold='fold_1',native_and_provenance_exact=True,
            context_max_absolute_difference=float(np.abs(data['context']-expected).max()),
            context_and_866_statistics_tolerance=dict(rtol=1e-6,atol=1e-6),scientific_scores_computed=0))
    result=dict(status='PASS',records=records,scope='old development inputs only; no new model scoring')
    save(Path(cfg['output'])/'DEVELOPMENT_INPUT_REPLAY.json',result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',choices=('TS','OS'));parser.add_argument('--fold');parser.add_argument('--development-replay',action='store_true')
    args=parser.parse_args();cfg=configure();check_method_lock(cfg,full_sources=True);sources=native_sources(cfg)
    if args.development_replay:development_replay(cfg,sources);print('FIVEFOLD_DEVELOPMENT_INPUT_REPLAY_PASS');return
    assert (args.protocol is None)==(args.fold is None)
    cases=[(args.protocol,args.fold)] if args.protocol else [(p,f) for p in cfg['protocols'] for f in cfg['folds']]
    for p,f in cases:prepare_case(cfg,p,f,sources);gc.collect()
    if not args.protocol:save(Path(cfg['output'])/'FIVEFOLD_COVERAGE.json',coverage(cfg))
    check_method_lock(cfg,full_sources=True)


if __name__=='__main__':main()
