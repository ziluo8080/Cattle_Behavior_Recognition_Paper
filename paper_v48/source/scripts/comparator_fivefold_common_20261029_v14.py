"""V14 immutable data/code contract; Test targets belong only to final scorer."""
import ast
import hashlib
import json
from pathlib import Path

CONFIG=Path('configs/sensor_comparator_fivefold_retrain_20261029_v14.json')
ROOT=Path('/q3/mmcows-sensor-behavior-20261007')
FIELDS=('dynamic','state','state_valid','context','current')

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def config():
    assert Path.cwd()==ROOT
    from scripts import run_comparator_convergence_20261027_v2 as parent
    parent.setup(False)
    cfg=read(CONFIG)
    assert cfg['models']==['GRU_Fusion','Transformer_Fusion','ModernTCN_Fusion']
    assert cfg['protocols']==['TS','OS'] and cfg['seeds']==[17,23,29]
    assert cfg['folds']==['fold_1','fold_2','fold_3','fold_4','fold_5']
    assert cfg['alpha_by_protocol']=={'TS':20000,'OS':100000}
    assert cfg['refit_epoch_transfer']=='same_epochs' and cfg['weight_decay']==.03
    assert cfg['calibration_stopping']==dict(minimum_epochs=40,maximum_epochs=160,patience_epochs=20,relative_improvement=.001)
    return cfg
def case_dir(cfg,stage,p,f,model,seed):
    assert stage in ['calibration','refit','prediction'] and p in cfg['protocols'] and f in cfg['folds'] and seed in cfg['seeds']
    assert model in cfg['models']+[cfg['rocket_model']]
    return Path(cfg['output'])/stage/p/f/model/f'seed{seed}'
def code(cfg):
    from scripts import run_comparator_convergence_20261027_v2 as parent
    result=parent.code(read(parent.CONFIG))
    paths=[CONFIG,Path(__file__),Path(cfg['ledger']),
           Path('scripts/run_comparator_fivefold_retrain_20261029_v14.py'),
           Path('scripts/comparator_fivefold_neural_20261029_v14.py'),
           Path('scripts/comparator_fivefold_rocket_20261029_v14.py'),
           Path('scripts/report_comparator_fivefold_retrain_20261029_v14.py'),
           Path('tests/test_comparator_fivefold_neural_20261029_v14.py'),
           Path('tests/test_comparator_fivefold_rocket_20261029_v14.py'),
           Path('tests/test_comparator_fivefold_queue_20261029_v14.py')]
    pending=paths+[Path(p) for p in result if p.endswith('.py')];seen=set()
    while pending:
        p=pending.pop()
        if p.is_absolute():p=p.relative_to(Path.cwd())
        if p in seen:continue
        seen.add(p);result[p.as_posix()]=sha(p)
        if p.suffix!='.py':continue
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            modules=[]
            if isinstance(node,ast.Import):modules=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                prefix=list(p.parent.parts)[:len(p.parent.parts)-node.level+1] if node.level else []
                base='.'.join(prefix+([node.module] if node.module else []))
                if base:modules=[base]+[base+'.'+a.name for a in node.names]
            for name in modules:
                base=Path(*name.split('.'))
                for item in [base.with_suffix('.py'),base/'__init__.py']:
                    if item.is_file():pending.append(item)
    return result
def check(cfg):
    seal=read(Path(cfg['output'])/'SEAL.json')
    assert seal['config']==cfg and seal['code']==code(cfg)
    for path,digest in seal['sources'].items():
        if path.endswith(('RECEIPT.json','REFIT_SOURCE.json','TRAINING_SEAL.json','/SEAL.json')):
            assert sha(path)==digest,path
    return seal
def _receipt(cfg,p,f):
    path=Path(cfg['selection_parent'])/'data'/p/f/'RECEIPT.json'
    r=read(path)
    assert r['status']=='PASS' and r['selection_and_final_fit_ID_union_exact']
    return path,r
def load_selection(cfg,p,f,device='cuda'):
    from scripts.converged_fivefold_common_20261027 import load_selection as load
    return load(dict(output=cfg['selection_parent']),p,f,device)
def _formal_receipt(cfg,p,f):
    root=Path(cfg['formal_parent'])/'data'/p/f
    r=read(root/'DATA_RECEIPT.json')
    assert r['phase']=='formal_refit' and not r['TEST_INPUTS_has_labels']
    return root,r
def raw_refit(cfg,p,f):
    import numpy as np
    root,r=_formal_receipt(cfg,p,f)
    assert sha(root/'FIT.npz')==r['outputs']['FIT.npz']
    with np.load(root/'FIT.npz',allow_pickle=False) as z:
        result={k:z[k].copy() for k in FIELDS+('labels','prior','class_weights')}
    counts=np.bincount(result['labels'],minlength=7)
    np.testing.assert_array_equal(counts,r['fit_class_counts'])
    np.testing.assert_array_equal(result['prior'],counts/counts.sum())
    np.testing.assert_array_equal(result['class_weights'],np.sqrt(counts.sum()/(7*counts)))
    assert len(result['labels'])==r['fit_targets'] and (counts>0).all()
    return result
def load_refit(cfg,p,f,device='cuda'):
    import numpy as np
    import torch
    raw=raw_refit(cfg,p,f);y=raw['labels']
    tensors={k:torch.as_tensor(raw[k],device=device) for k in FIELDS}
    tensors['labels']=torch.as_tensor(y,device=device)
    return dict(tensors=tensors,weights=torch.as_tensor(raw['class_weights'],dtype=torch.float32,device=device),
        prior=raw['prior'],train=np.arange(len(y)),evaluation=np.array([],dtype=np.int64),fit_truth=y)
def raw_test(cfg,p,f):
    import numpy as np
    check_training_lock(cfg)
    root,r=_formal_receipt(cfg,p,f)
    assert sha(root/'TEST_INPUTS.npz')==r['outputs']['TEST_INPUTS.npz']
    with np.load(root/'TEST_INPUTS.npz',allow_pickle=False) as z:
        assert 'labels' not in z.files
        result={k:z[k].copy() for k in FIELDS};result['prior']=z['prior'].copy()
    counts=np.asarray(r['fit_class_counts'])
    np.testing.assert_array_equal(result['prior'],counts/counts.sum())
    result['class_weights']=np.sqrt(counts.sum()/(7*counts))
    assert len(result['dynamic'])==r['test_targets']
    return result
def load_test(cfg,p,f,device='cuda'):
    import numpy as np
    import torch
    raw=raw_test(cfg,p,f);n=len(raw['dynamic'])
    tensors={k:torch.as_tensor(raw[k],device=device) for k in FIELDS}
    tensors['labels']=torch.full((n,),-100,dtype=torch.long,device=device)
    return dict(tensors=tensors,weights=torch.as_tensor(raw['class_weights'],device=device,dtype=torch.float32),
        prior=raw['prior'],train=np.array([],dtype=np.int64),evaluation=np.arange(n))
def gpu_lock(cfg,slot,kind='neural'):
    import fcntl
    import time
    from scripts.accelerate_comparator_queue_20261028_v8b import slot_gpu
    from scripts.run_convtran_context_20261022 import configure
    assert kind in ['neural','rocket'] and (slot==2 if kind=='rocket' else slot in [0,1,2])
    with (ROOT/'reports/sensor_motion_sync_validation_20261020_v1/SINGLE_GPU.lock').open('a+b') as guard:
        try:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:pass
        else:
            fcntl.flock(guard,fcntl.LOCK_UN)
            raise RuntimeError('V14 root controller must hold the global GPU lock')
    settings=cfg['runtime']
    runtime=dict(runtime_output=cfg['runtime_output'],GPU_processes=3,
        Torch_memory_fraction_per_process=settings[kind+'_fraction'],minimum_free_GPU_MiB=settings[kind+'_minimum_free_MiB'])
    deadline=time.monotonic()+120
    while True:
        try:lock=slot_gpu(runtime,slot);break
        except AssertionError as exc:
            if 'Resource admission failed before scientific STARTED' not in str(exc) or time.monotonic()>=deadline:raise
            time.sleep(5)
    try:configure();return lock
    except BaseException:lock.close();raise
def source_manifest(cfg):
    """Hash only TRAIN/VAL and training FIT; Test byte streams stay closed."""
    selection_seal=Path(cfg['selection_parent'])/'TRAINING_SEAL.json'
    formal_seal=Path(cfg['formal_parent'])/'SEAL.json'
    original=read(selection_seal);original_formal=read(formal_seal)
    files={str(selection_seal):sha(selection_seal),str(formal_seal):sha(formal_seal)};cases=[]
    for p in cfg['protocols']:
        for f in cfg['folds']:
            receipt_path,r=_receipt(cfg,p,f);root=receipt_path.parent
            assert sha(receipt_path)==original['data'][str(receipt_path)]
            files[str(receipt_path)]=sha(receipt_path)
            for name,digest in r['outputs'].items():
                assert sha(root/name)==digest
                assert digest==original['data'][str(root/name)]
                files[str(root/name)]=digest
            reference=read(root/'REFIT_SOURCE.json');formal,fr=_formal_receipt(cfg,p,f)
            files[str(formal/'DATA_RECEIPT.json')]=sha(formal/'DATA_RECEIPT.json')
            assert files[str(formal/'DATA_RECEIPT.json')]==original_formal['receipts'][str(formal/'DATA_RECEIPT.json')]
            assert reference['FIT_sha256']==fr['outputs']['FIT.npz']
            assert reference['TEST_INPUTS_sha256']==fr['outputs']['TEST_INPUTS.npz']
            for name in ['FIT.npz','FIT_TARGETS.jsonl']:
                assert sha(formal/name)==fr['outputs'][name]
                assert fr['outputs'][name]==original_formal['data_files'][str(formal/name)]
                files[str(formal/name)]=fr['outputs'][name]
            key=lambda row:(row['cow_id'],row['timestamp'],row['label'])
            selection=[]
            for name in ['FIT_TARGETS.jsonl','DEV_TARGETS.jsonl']:
                selection += [key(json.loads(line)) for line in (root/name).read_text().splitlines()]
            fit=[key(json.loads(line)) for line in (formal/'FIT_TARGETS.jsonl').read_text().splitlines()]
            assert sorted(selection)==sorted(fit)
            cases.append(dict(protocol=p,fold=f,calibration_fit_rows=r['fit_rows'],calibration_val_rows=r['dev_rows'],
                final_fit_rows=fr['fit_targets'],test_rows=fr['test_targets'],Test_targets_opened=False))
    return files,cases
def check_training_lock(cfg):
    lock=read(Path(cfg['output'])/'TRAINING_LOCK.json')
    assert lock['seal_sha256']==sha(Path(cfg['output'])/'SEAL.json')
    assert lock['status']=='FROZEN_TRAINING' and lock['eligible_models']
    for path,digest in lock['files'].items():
        if path.endswith('COMPLETE.json'):assert sha(path)==digest,path
    return lock
