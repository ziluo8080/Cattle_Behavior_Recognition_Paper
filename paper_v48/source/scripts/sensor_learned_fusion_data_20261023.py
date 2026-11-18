"""TRAIN-private cross-fit data, reconstructed from raw unlabeled sensor sources.

Inner folds never use the outer TRAIN normalizers. The saved source targets are
the only label source; no original behavior-label file is opened here.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

CLOUD_ROOT=Path('/q3/mmcows-sensor-behavior-20261007')
SOURCE='reports/sensor_xgb_b_fusion_20261022_v1'
RAW_STATE='data/sensor_formal_0725_20261007'
RAW_NATIVE='reports/sensor_native_immu_inventory_20261021_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(4*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def class_counts(rows,ids):
    return np.bincount([rows[int(i)]['label'] for i in ids],minlength=8)[1:8].astype(int).tolist()


def make_splits(rows,protocol,k=3):
    """Label-independent blocked TS / grouped OS partition of outer TRAIN only.

    TS divides each outer TRAIN segment into k equal wall-time chunks. Sources
    are bounded to these chunks. Excluding the first 60 seconds of each chunk
    ensures every retained historical window stays inside its own source domain.
    OS uses contiguous groups of sorted outer TRAIN cow ids; all its TRAIN rows
    are covered. Returned train/held indices are local to returned rows; the
    original_ids mapping always addresses the supplied full TRAIN/VAL roster.
    """
    if protocol not in ('TS','OS') or k<2:raise ValueError('Unsupported protocol/fold count')
    if not rows or not {r['split'] for r in rows}.issubset({'train','val'}):
        raise ValueError('Only outer TRAIN/VAL targets may be supplied')
    train_ids=[i for i,r in enumerate(rows) if r['split']=='train']
    if not train_ids:raise ValueError('No outer TRAIN targets')
    cow_ids=sorted({int(rows[i]['cow_id']) for i in train_ids})
    if protocol=='OS' and len(cow_ids)<k:raise ValueError('Fewer cows than inner folds')
    unique={(int(rows[i]['cow_id']),int(rows[i]['timestamp'])) for i in train_ids}
    if len(unique)!=len(train_ids):raise ValueError('Duplicate outer TRAIN target')
    cow_groups=[list(map(int,part)) for part in np.array_split(cow_ids,k)]
    cow_group={cow:g for g,cows in enumerate(cow_groups) for cow in cows}
    prepared=[];domains={}
    for original_id in train_ids:
        row=dict(rows[original_id]);cow=int(row['cow_id']);t=int(row['timestamp'])
        a,b=int(row['start']),int(row['end'])
        if not a<=t<b:raise ValueError('Target outside its outer source domain')
        if protocol=='TS':
            edges=np.linspace(a,b,k+1,dtype=np.int64)
            if np.any(np.diff(edges)<=60):raise ValueError('TS chunks must exceed 60 seconds')
            group=min(int(np.searchsorted(edges,t,side='right')-1),k-1)
            begin,end=int(edges[group]),int(edges[group+1])
            # Domain existence is determined by metadata, before target filtering.
            domain=dict(start=begin,end=end,segment=f'{row["segment"]}::inner{k}_part{group}',group=group)
            for g in range(k):
                d=dict(start=int(edges[g]),end=int(edges[g+1]),segment=f'{row["segment"]}::inner{k}_part{g}',group=g)
                domains[cow,d['start'],d['end'],g]=d
            if t<begin+60:continue
            row.update(start=begin,end=end,segment=domain['segment'])
        else:
            group=cow_group[cow]
            domain=dict(start=a,end=b,segment=row['segment'],group=group)
            domains[cow,a,b,group]=domain
        prepared.append((original_id,row,group))
    if not prepared:raise ValueError('No eligible cross-fit targets')
    original_ids=np.asarray([item[0] for item in prepared],np.int64)
    group_ids=np.asarray([item[2] for item in prepared],np.int64)
    output=[]
    for held_group in range(k):
        inner_rows=[dict(r,split='val' if g==held_group else 'train') for _,r,g in prepared]
        train=np.flatnonzero(group_ids!=held_group);held=np.flatnonzero(group_ids==held_group)
        if not len(train) or not len(held):raise ValueError('Empty inner TRAIN or held partition')
        fit_domains={};held_domains={}
        for (cow,_,_,g),d in sorted(domains.items()):
            target=held_domains if g==held_group else fit_domains
            target.setdefault(cow,[]).append(dict(start=d['start'],end=d['end'],segment=d['segment'],
                split='val' if g==held_group else 'train'))
        output.append(dict(inner_id=held_group,protocol=protocol,k=k,rows=inner_rows,
            original_ids=original_ids.copy(),group_ids=group_ids.copy(),train=train,held=held,
            fit_domains=fit_domains,held_domains=held_domains,cow_groups=cow_groups if protocol=='OS' else None,
            counts=dict(train=class_counts(inner_rows,train),held=class_counts(inner_rows,held)),
            coverage=dict(outer_train_targets=len(train_ids),OOF_targets=len(prepared),
                excluded_boundary_targets=len(train_ids)-len(prepared),fraction=len(prepared)/len(train_ids))))
    # Every retained outer TRAIN target is held exactly once, never outer VAL.
    all_held=np.concatenate([x['original_ids'][x['held']] for x in output])
    np.testing.assert_array_equal(np.sort(all_held),original_ids)
    return output


def inside(times,domains):
    keep=np.zeros(len(times),bool)
    for domain in domains:keep|=(times>=domain['start'])&(times<domain['end'])
    return keep


def outer_domains(rows):
    result={}
    for r in rows:
        if r['split']!='train':continue
        result.setdefault(int(r['cow_id']),set()).add((int(r['start']),int(r['end'])))
    return {cow:[dict(start=a,end=b,split='train') for a,b in sorted(intervals)]
        for cow,intervals in result.items()}


def raw_sources(root,rows):
    """Hash/read only selected unlabeled source files, then trim outer TRAIN."""
    import pandas as pd
    from q3_sensor.protocol import FEATURES
    from scripts.sensor_native_immu_data_20261021 import canonical_source,CHANNELS
    domains=outer_domains(rows);state_root=root/RAW_STATE;native_root=root/RAW_NATIVE
    source_receipt=read(state_root/'source_receipt.json')
    members={Path(item['member']).relative_to('sensor_data').as_posix():item['sha256']
        for item in source_receipt['members']}
    native_receipt=read(native_root/'FULL_DAY_TRAIN_INVENTORY.json')
    assert native_receipt['status']=='FULL_DAY_TRAIN_INVENTORY_COMPLETE'
    native_hashes={int(item['cow_id']):item['cached_csv_sha256'] for item in native_receipt['cows']}
    frames={};sources={};receipts={}
    for cow in sorted(domains):
        tag,name=f'T{cow:02d}',f'C{cow:02d}'
        files=dict(uwb=f'main_data/uwb/{tag}/{tag}_0725.csv',
            hd=f'sub_data/head_direction/{tag}/{tag}_0725.csv',akl=f'main_data/ankle/{name}/{name}_0725.csv')
        frames[cow]={}
        for modality,relative in files.items():
            path=state_root/relative;digest=sha(path)
            assert digest==members[relative],f'Raw sensor source changed: {relative}'
            frame=pd.read_csv(path,usecols=['timestamp']+list(FEATURES[modality]))
            frame=frame.loc[inside(frame.timestamp.to_numpy(),domains[cow])].copy()
            assert np.isfinite(frame.timestamp.to_numpy()).all() and frame.timestamp.is_monotonic_increasing
            frames[cow][modality]=frame
            receipts[path.relative_to(root).as_posix()]=dict(sha256=digest,
                outer_train_rows=len(frame),outer_domains=domains[cow])
        path=native_root/'raw_os_train_union'/f'{tag}_0725.csv';digest=sha(path)
        assert digest==native_hashes[cow],f'Raw native source changed: {path}'
        frame=pd.read_csv(path,usecols=['timestamp']+list(CHANNELS))
        frame=frame.loc[inside(frame.timestamp.to_numpy(),domains[cow])].copy()
        sources[cow]=canonical_source(frame)
        receipts[path.relative_to(root).as_posix()]=dict(sha256=digest,
            outer_train_rows=len(frame),outer_domains=domains[cow])
    return frames,sources,receipts


def inner_state_normalizer(frames,fit_domains):
    """Prefilter per-cow source membership before shared normalizer helper."""
    from q3_sensor.protocol import fit_normalizer,MODALITIES
    fit_frames={}
    limits=[];membership={}
    for cow,domains in sorted(fit_domains.items()):
        fit_frames[cow]={};membership[str(cow)]={};limits.extend((d['start'],d['end']) for d in domains)
        for modality in MODALITIES:
            frame=frames[cow][modality]
            chosen=frame.loc[inside(frame.timestamp.to_numpy(),domains)].copy()
            fit_frames[cow][modality]=chosen
            membership[str(cow)][modality]=dict(rows=len(chosen),domains=domains)
    bounds=[dict(start=min(a for a,b in limits),end=max(b for a,b in limits),split='train')]
    return fit_normalizer(fit_frames,bounds),membership


def context_from_inner(arrays,normalizer,train):
    import torch
    from q3_sensor.endpoint_models import tokens
    from scripts.sensor_classic_features_20261012 import descriptors
    context,names=descriptors(arrays,normalizer)
    current=np.concatenate([tokens({k:torch.from_numpy(np.array(v[:,-1:],copy=True))
        for k,v in arrays[m].items()}).numpy()[:,0] for m in ('uwb','hd','akl')],axis=1)
    unscaled=np.concatenate([context,current],axis=1).astype(np.float32)
    mean=unscaled[train].astype(np.float64).mean(0)
    raw_std=unscaled[train].astype(np.float64).std(0)
    std=np.where(raw_std<=1e-6,1.,raw_std)
    normalized=((unscaled-mean)/std).astype(np.float32)
    assert normalized.shape[1]==136 and np.isfinite(normalized).all()
    return normalized,current,dict(mean=mean.tolist(),std=std.tolist(),raw_std=raw_std.tolist(),
        scope='inner TRAIN target descriptors only',feature_names=names+[f'current_token_{i}' for i in range(26)],
        constant_or_near_constant_policy='inner TRAIN std <=1e-6 uses scale1')


def prepare_inner(root,protocol,inner_id,out_dir):
    """Create one inner cache in the existing cloud runtime. No model fitting."""
    root=Path(root).resolve();out=Path(out_dir)
    if root!=CLOUD_ROOT or Path.cwd().resolve()!=root:raise RuntimeError('Scientific input preparation is cloud-only')
    if protocol not in ('TS','OS') or inner_id not in (0,1,2):raise ValueError('Outside fixed3 fold_1 scope')
    target_path=root/SOURCE/'data'/protocol/'targets.jsonl'
    inference=read(root/SOURCE/'INFERENCE_COMPLETE.json')
    assert inference['status']=='PASS'
    expected=next(r['target_sha256'] for r in inference['receipts'] if r['protocol']==protocol)
    assert sha(target_path)==expected,'Frozen source targets changed'
    existing=out/'DATA_RECEIPT.json'
    if existing.exists():
        result=read(existing)
        assert result['protocol']==protocol and result['inner_id']==inner_id and result['outer_target_sha256']==expected
        assert result['data_code_sha256']==sha(__file__),'Preparation code changed'
        for name,digest in result['outputs'].items():assert sha(out/name)==digest,name
        return result
    assert not (out/'DATA.npz').exists(),'Partial inner cache exists; do not overwrite'
    from q3_sensor.protocol import build_windows,MODALITIES
    from q3_sensor.endpoint_models import tokens
    from scripts.sensor_native_immu_data_20261021 import fit_scaler,windows
    from experiments.sensor_comparator_pilot_20261022_v2.cpu import summary_features
    import torch
    source_rows=[json.loads(x) for x in target_path.read_text().splitlines()]
    split=make_splits(source_rows,protocol,k=3)[inner_id]
    rows=split['rows'];train,held=split['train'],split['held']
    frames,sources,source_receipts=raw_sources(root,source_rows)
    state_norm,membership=inner_state_normalizer(frames,split['fit_domains'])
    native_norm=fit_scaler(sources,split['fit_domains'])
    state_windows=build_windows(rows,frames,state_norm)
    native,native_ticks=windows(rows,sources,native_norm)
    bounds=[]
    for modality in MODALITIES:
        win=state_windows[modality];valid=win['valid'].astype(bool)
        src=win['source'];target=np.asarray([r['timestamp'] for r in rows])[:,None,None]
        grid=target+np.arange(-59,1)[None,:,None]
        assert np.array_equal(np.isfinite(src),valid)
        assert np.all((src<=grid)[valid]) and np.all((src>target-60)[valid])
        for cow in sorted({r['cow_id'] for r in rows}):
            for role,indices,domains in (('train',train,split['fit_domains']),('val',held,split['held_domains'])):
                chosen=[int(i) for i in indices if rows[int(i)]['cow_id']==cow]
                if not chosen:continue
                used=src[chosen];used=used[np.isfinite(used)]
                assert inside(used,domains[cow]).all(),'State source outside inner role'
                opposite=split['held_domains'] if role=='train' else split['fit_domains']
                assert not inside(used,opposite.get(cow,[])).any(),'State source shared with opposite role'
        bounds.append(dict(modality=modality,valid_source_elements=int(valid.sum()),inner_source_membership=True))
    for role,indices,domains in (('train',train,split['fit_domains']),('val',held,split['held_domains'])):
        for cow in sorted({rows[int(i)]['cow_id'] for i in indices}):
            chosen=[int(i) for i in indices if rows[int(i)]['cow_id']==cow]
            ticks=native_ticks[chosen];ticks=ticks[ticks>=0]
            assert inside(ticks/10,domains[cow]).all(),'Native source outside inner role'
            opposite=split['held_domains'] if role=='train' else split['fit_domains']
            assert not inside(ticks/10,opposite.get(cow,[])).any(),'Native source shared with opposite role'
    arrays={m:{k:state_windows[m][k] for k in ('x','valid','fresh','age')} for m in MODALITIES}
    context,current,context_norm=context_from_inner(arrays,state_norm,train)
    state=np.concatenate([tokens({k:torch.from_numpy(np.array(v,copy=True)) for k,v in arrays[m].items()}).numpy()
        for m in MODALITIES],axis=-1).astype(np.float32)
    state_valid=np.stack([arrays[m]['valid'].astype(bool).any(-1) for m in MODALITIES],axis=-1).any(-1)
    # Match the sealed CPU expert's finite invalid-payload/age convention.
    dynamic=native.copy();mask=dynamic[:,:,6]>.5
    dynamic[:,:,:6]=np.where(mask[:,:,None],dynamic[:,:,:6],0.)
    dynamic[:,:,7]=np.where(mask,np.clip(dynamic[:,:,7],0,.2),.2)
    features=summary_features(dict(context=context,current=current,native=dynamic,state=state))
    labels=np.asarray([r['label']-1 for r in rows],np.int64)
    counts=np.bincount(labels[train],minlength=7)
    assert len(counts)==7 and (counts>0).all()
    payload=dict(dynamic=dynamic,context=context,state=state,state_valid=state_valid,current=current,
        features=features,labels=labels,train=train,held=held,original_ids=split['original_ids'],prior=counts/counts.sum())
    assert dynamic.shape==(len(rows),100,8) and state.shape==(len(rows),60,26) and features.shape==(len(rows),866)
    for key in ('dynamic','context','state','current','features'):assert np.isfinite(payload[key]).all(),key
    np.testing.assert_array_equal(current,state[:,-1])
    assert all(source_rows[int(i)]['split']=='train' for i in split['original_ids'])
    out.mkdir(parents=True,exist_ok=True)
    np.savez(out/'DATA.npz',**payload)
    save(out/'NORMALIZERS.json',dict(state=state_norm,native=native_norm,context=context_norm,
        inner_fit_class_counts=counts.tolist(),inner_fit_prior=payload['prior'].tolist()))
    save(out/'DOMAINS.json',dict(fit=split['fit_domains'],held=split['held_domains']))
    result=dict(status='PASS',protocol=protocol,outer_fold='fold_1',inner_id=inner_id,inner_folds=3,
        normalizer_scope='inner TRAIN only; raw sensor sources, no outer normalizers',
        outer_target_sha256=expected,targets=len(rows),train_targets=len(train),held_targets=len(held),
        counts=split['counts'],coverage=split['coverage'],state_normalizer_membership=membership,
        source_files=source_receipts,source_checks=bounds,original_ids_match_outer_TRAIN=True,
        no_outer_VAL_in_inner_data=True,no_shared_source_between_fit_and_held=True,
        raw_sensor_reconstruction=True,Test_labels_or_predictions_read=False,model_fits=0,
        data_code_sha256=sha(__file__),outputs={name:sha(out/name) for name in ('DATA.npz','NORMALIZERS.json','DOMAINS.json')})
    save(existing,result)
    return result
