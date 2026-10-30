"""Two bounded new-initialization TimesNet development pilots; no expansion."""
import argparse
import ast
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

# Must precede importing numerical libraries in each fresh worker.
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

CONFIG = Path('configs/sensor_timesnet_pilot_20261029_v17.json')
MODULE = 'scripts.run_timesnet_pilot_20261029_v17'
MODEL = 'TimesNet_DS'
ADAPTER = Path('experiments/sensor_timesnet_pilot_20261029_v17')
DATA_NAMES = ('FIT.npz', 'DEV_INPUTS.npz', 'DEV_TARGETS.jsonl')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def runtime_imports():
    global np, torch, parent, base, core, metric, optimization, generalization, training_components, old_stopping, selected, make_model
    import numpy as np
    import torch
    from scripts import run_comparator_convergence_20261027_v2 as parent
    from scripts.sensor_fusion_stability_20261027 import metric
    from scripts.comparator_convergence_criteria_20261027_v2 import assess as optimization
    from scripts.comparator_generalization_gate_20261027 import assess as generalization
    from scripts.run_comparator_stability_extension_20261028_v9 import training_components
    from scripts.comparator_repair_policy_20261028 import stopping as old_stopping, selected
    from experiments.sensor_timesnet_pilot_20261029_v17.model import make_model
    base, core = parent.base, parent.core


def validate_config(cfg):
    assert cfg['models'] == [MODEL] and cfg['protocols'] == ['TS', 'OS']
    assert cfg['inners'] == [0] and cfg['seeds'] == [17]
    assert cfg['data_parent'] == 'reports/sensor_zero_head_replication_20261024_v1'
    assert cfg['timesnet'] == dict(d_model=16,d_ff=32,e_layers=2,top_k=3,num_kernels=4,dropout=.1)
    assert cfg['learning_rate'] == .0003 and cfg['weight_decay'] == .03
    assert cfg['batch_size'] == cfg['eval_batch_size'] == 256
    assert cfg['EMA_decay'] == .998 and cfg['EMA_warmup_steps'] == 200 and cfg['evaluation_window'] == 3
    assert cfg['stopping'] == dict(minimum_epochs=40,maximum_epochs=160,patience_epochs=20,relative_improvement=.001)
    assert cfg['scheduler'] == dict(mode='min',factor=.2,patience=2,threshold=.0001,threshold_mode='rel',cooldown=0,min_lr=.000003,eps=0)
    assert cfg['Torch_memory_fraction_per_process'] == .125 and cfg['minimum_free_GPU_MiB'] == 8192
    assert cfg['threads'] == 1 and cfg['admission_seconds'] == 120
    assert cfg['budget']['worker_seconds'] == 10800 and cfg['budget']['queue_seconds'] == 28800
    assert cfg['budget']['gpu_parallel'] == 1 and cfg['budget']['max_attempts'] == 1
    assert not any(cfg[k] for k in ('Test_allowed','outer_VAL_allowed','formal_refit_allowed','new_paid_resources','automatic_expansion','automatic_fivefold_expansion'))


def setup(cuda=False):
    runtime_imports(); cfg = read(CONFIG); validate_config(cfg)
    assert Path.cwd() == core.ROOT, 'Witnesses and workloads use the verified existing cloud workspace'
    assert (torch.__version__, np.__version__) == ('2.4.1+cu121','1.24.4')
    if cuda:
        core.configure()
        torch.cuda.set_per_process_memory_fraction(cfg['Torch_memory_fraction_per_process'])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    return cfg


def code(cfg):
    """Recursive local-import closure plus definition-loaded vendored source.

    Receipts are deliberately outside this closure, avoiding report hash cycles.
    """
    paths = {CONFIG, Path(__file__).resolve().relative_to(Path.cwd()), Path(cfg['ledger']),
             Path('tests/test_timesnet_runner_20261029_v17.py'),
             Path('configs/sensor_comparator_joint_acceptance_20261027_v1.json')}
    paths.update(p for p in ADAPTER.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    paths.update(Path('tests').glob('test_timesnet*20261029_v17.py'))
    pending = list(paths); scanned = set()
    while pending:
        path = pending.pop()
        assert path.is_file(), str(path)
        if path in scanned or path.suffix != '.py': continue
        scanned.add(path)
        tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): modules += [x.name for x in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                modules += [node.module] + [node.module+'.'+x.name for x in node.names]
        for module in modules:
            candidate = Path(*module.split('.')).with_suffix('.py')
            package = Path(*module.split('.'))/'__init__.py'
            for target in (candidate, package):
                if target.is_file() and target not in paths:
                    paths.add(target); pending.append(target)
    # The inherited adapter dynamically imports these local vendor packages.
    for directory in ('experiments/sensor_comparator_pilot_20261022_v2/vendor', 'third_party/convtran_context_20261022_v1'):
        paths.update(Path(directory).rglob('*.py'))
    return {p.as_posix(): sha(p) for p in sorted(paths)}


def verify(files):
    for path, digest in files.items():
        assert sha(path) == digest, 'Pinned source changed: '+path


def data_files(cfg):
    root = Path(cfg['data_parent']); seal_path = root/'SEAL.json'; seal = read(seal_path)
    result = {seal_path.as_posix(): sha(seal_path)}
    for protocol in cfg['protocols']:
        folder = root/'data'/protocol/'inner0'; receipt_path = folder/'RECEIPT.json'
        assert sha(receipt_path) == seal['files'][receipt_path.as_posix()]
        result[receipt_path.as_posix()] = sha(receipt_path)
        receipt = read(receipt_path)
        assert set(receipt['files']) == set(DATA_NAMES), 'Restricted loader must not read additional files'
        for name in DATA_NAMES:
            path = folder/name; assert sha(path) == receipt['files'][name]
            result[path.as_posix()] = sha(path)
    return result


def load_data(cfg, protocol, expected):
    assert protocol in ('TS', 'OS'); verify(expected)
    root = Path(cfg['data_parent'])
    parent_cfg = read(root/'SEAL.json')['config']
    assert parent_cfg['output'] == cfg['data_parent']
    # The inherited loader reads only this inner0 FIT / DEV_INPUTS and its receipt.
    data = base.load_data(parent_cfg, protocol, 0)
    target = root/'data'/protocol/'inner0'/'DEV_TARGETS.jsonl'
    truth = np.asarray([json.loads(line)['label']-1 for line in target.read_text().splitlines()])
    assert len(truth) == len(data['evaluation'])
    verify(expected)
    return data, truth


def private_admission(cfg):
    """Only our own advisory lock. Never touches the formal-controller lock."""
    import fcntl
    handle = Path(cfg['private_GPU_lock']).open('a+b'); deadline = time.monotonic()+cfg['admission_seconds']
    while True:
        locked = False
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB); locked = True
            free = int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
            if free >= cfg['minimum_free_GPU_MiB']: return handle
        except BlockingIOError:
            pass
        except Exception:
            handle.close(); raise
        if locked: fcntl.flock(handle, fcntl.LOCK_UN)
        if time.monotonic() >= deadline:
            handle.close(); raise RuntimeError('HOLD_RESOURCE_ADMISSION: no scientific STARTED or retry')
        time.sleep(min(5,max(0,deadline-time.monotonic())))


def new(cfg, prior, seed, device):
    core.seed_all(seed); model = make_model(cfg, prior).to(device)
    opt = base.optimizer_for(model,cfg)
    ema = base.ParameterEMA(model,cfg['EMA_decay'],cfg['EMA_warmup_steps'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,**cfg['scheduler'])
    return dict(model=model,opt=opt,ema=ema,scheduler=scheduler,step=0)


def restore(cfg, prior, seed, payload, device):
    state = new(cfg,prior,seed,device)
    state['model'].load_state_dict(payload['model']); state['ema'].model.load_state_dict(payload['ema'])
    state['opt'].load_state_dict(copy.deepcopy(payload['optimizer']))
    state['scheduler'].load_state_dict(copy.deepcopy(payload['scheduler']))
    state['ema'].updates=payload['ema_updates']; state['step']=payload['global_step']
    core.restore_rng(payload['rng'])
    return state


def bounded_stop(history,cfg):
    result=old_stopping(history,cfg['stopping'])
    stable=training_components(optimization(history,cfg['convergence']),cfg['convergence'])
    accepted=stable['passed'] and result['validation_plateau']
    exhausted=history[-1]['epoch'] >= cfg['stopping']['maximum_epochs']
    result.update(stop=accepted or exhausted,training_components=stable,
                  reason='TRAINING_STABLE_AND_DEV_PLATEAU' if accepted else 'BOUNDED_BUDGET_HOLD' if exhausted else 'CONTINUE')
    return result


def batch_contract(cfg,data,seed):
    def digest(values): return hashlib.sha256(np.asarray(values,dtype='<i8').tobytes()).hexdigest()
    n=len(data['train'])
    return dict(train_batch_size=256,eval_batch_size=256,drop_last=False,train_tail_size=n%256 or 256,
                FIT_eval_tail_size=n%256 or 256,DEV_eval_tail_size=len(data['evaluation'])%256 or 256,
                train_order='numpy default_rng(SeedSequence([seed,epoch,0])).permutation(train_ids)',seed=seed,
                FIT_eval_ids_sha256=digest(data['train']),DEV_eval_ids_sha256=digest(data['evaluation']),
                FIT_eval_rows=n,DEV_eval_rows=len(data['evaluation']),eval_order='exact loader ids order',
                FFT_periods_depend_on_batch_membership=True,batch_size_invariance_claimed=False)


def witness(cfg,cpu):
    if cpu: assert not torch.cuda.is_available(), 'Fresh CPU witness must hide CUDA'
    lock=None if cpu else private_admission(cfg)
    try:
        if cpu:
            subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_timesnet*20261029_v17.py'],check=True)
        folder=Path(cfg['output'])/('cpu_witness_work' if cpu else 'cuda_witness_work'); folder.mkdir(parents=True,exist_ok=False)
        records=[]; device='cpu' if cpu else 'cuda'
        for index,protocol in enumerate(cfg['protocols']):
            core.seed_all(1700+index)
            batch=parent.old.synthetic_batch(device,21)
            data=dict(prior=np.arange(1,8)/28,train=np.arange(21),evaluation=np.arange(21),weights=torch.arange(1,8,device=device).float(),
                      tensors=dict(**batch,labels=torch.arange(21,device=device)%7))
            state=new(cfg,data['prior'],17,device); snaps=[]
            for epoch in range(9,12):
                parent.update(state,data,data['train']); state['scheduler'].step(.5)
                snaps.append(parent.snapshot(state,epoch))
            path=folder/(protocol+'.pt'); parent.checkpoint(path,state,epoch=11,causal_ema_snapshots=snaps)
            payload=torch.load(path,map_location='cpu',weights_only=False)
            expected=parent.update(state,data,data['train']); state['scheduler'].step(.51)
            snaps.append(parent.snapshot(state,12)); snaps=snaps[-3:]; expected_rng=core.rng()
            other=restore(cfg,data['prior'],17,payload,device)
            actual=parent.update(other,data,data['train']); other['scheduler'].step(.51)
            other_snaps=copy.deepcopy(payload['causal_ema_snapshots']); other_snaps.append(parent.snapshot(other,12)); other_snaps=other_snaps[-3:]
            base.equal_tree(expected,actual); base.equal_tree(expected_rng,core.rng()); base.equal_tree(snaps,other_snaps)
            for key in ('model','opt','scheduler'): base.equal_tree(state[key].state_dict(),other[key].state_dict())
            base.equal_tree(state['ema'].model.state_dict(),other['ema'].model.state_dict())
            assert state['ema'].updates==other['ema'].updates and state['step']==other['step']
            raw=copy.deepcopy(state['ema'].model.state_dict()); before_rng=core.rng()
            evaluation=parent.evaluation_model(state,snaps,12)
            assert not any(p.requires_grad for p in evaluation.parameters())
            prob=base.infer(evaluation,data,data['evaluation'],batch_size=256)
            base.equal_tree(raw,state['ema'].model.state_dict()); base.equal_tree(before_rng,core.rng())
            replay=make_model(cfg,data['prior']).to(device).eval().requires_grad_(False)
            replay.load_state_dict(evaluation.state_dict())
            np.testing.assert_array_equal(prob,base.infer(replay,data,data['evaluation'],batch_size=256))
            records.append(dict(protocol=protocol,synthetic_only=True,complete_next_update_exact=True,
                                model_optimizer_EMA_scheduler_RNG_global_step_causal_snapshots_exact=True,
                                evaluation_clone_preserves_live_EMA_RNG=True,requires_grad_false_replay_exact=True,
                                parameters=sum(p.numel() for p in evaluation.parameters())))
            del state,other,payload,data,batch,evaluation,replay,snaps,other_snaps; gc.collect()
            if not cpu: torch.cuda.empty_cache()
        save(Path(cfg['output'])/('CPU_WITNESS.json' if cpu else 'CUDA_WITNESS.json'),
             dict(status='PASS',real_data_rows_read=0,CUDA_validated=not cpu,records=records,code=code(cfg)))
    finally:
        if lock: lock.close()


def preflight(cfg):
    out=Path(cfg['output']); assert not (out/'PREFLIGHT.json').exists()
    expected=data_files(cfg); source=code(cfg); lock=private_admission(cfg); records=[]
    try:
        for protocol in cfg['protocols']:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            data,truth=load_data(cfg,protocol,expected); state=new(cfg,data['prior'],17,'cuda')
            times=[]
            for i in range(3):
                tick=time.monotonic(); parent.update(state,data,data['train'][:256]); torch.cuda.synchronize(); times.append(time.monotonic()-tick)
            # Keep live optimizer/EMA, full data, update clones, causal snapshots and eval clone resident together.
            before={n:p.detach().clone() for n,p in state['model'].named_parameters()}
            ebefore={n:p.detach().clone() for n,p in state['ema'].model.named_parameters()}
            optimizer_clone=copy.deepcopy(state['opt'].state_dict())
            snaps=[parent.snapshot(state,e) for e in (9,10,11)]
            # Exercise a real full update with the worker's raw/EMA delta clones,
            # all optimizer moments and causal snapshots already live.
            tick=time.monotonic(); parent.update(state,data,data['train'][:256]); torch.cuda.synchronize()
            times.append(time.monotonic()-tick)
            evaluation=parent.evaluation_model(state,snaps,11)
            tick=time.monotonic()
            for ids in (data['train'],data['evaluation']): base.infer(evaluation,data,ids,batch_size=256)
            torch.cuda.synchronize(); inference=time.monotonic()-tick
            parent.checkpoint(out/'preflight_work'/f'{protocol}.pt',state,epoch=11,causal_ema_snapshots=snaps)
            estimate=(max(times[1:])*math.ceil(len(data['train'])/256)+inference*1.5+1)*160*1.5+60
            peak=torch.cuda.max_memory_allocated(); reserved=torch.cuda.max_memory_reserved()
            limit=torch.cuda.get_device_properties(0).total_memory*.125
            records.append(dict(protocol=protocol,FIT_rows=len(data['train']),DEV_rows=len(truth),batch_size=256,
                train_step_seconds=times,full_FIT_DEV_inference_seconds=inference,peak_cuda_bytes=peak,
                peak_reserved_bytes=reserved,memory_cap_bytes=limit,optimizer_EMA_snapshots_and_clones_resident=True,
                conservative_maximum_worker_seconds=estimate,passed=peak<limit and reserved<=limit and estimate<10800))
            del data,truth,state,before,ebefore,optimizer_clone,snaps,evaluation; gc.collect(); torch.cuda.empty_cache()
        verify(expected); assert source==code(cfg)
        status='PASS' if all(r['passed'] for r in records) else 'HOLD_CAPACITY_OR_RUNTIME'
        save(out/'PREFLIGHT.json',dict(status=status,records=records,code=source,data_files=expected,
             official_Test_rows_read=0,batch_changed=False,planned_maximum_cases=2))
        assert status=='PASS',status
    except Exception as exc:
        if not (out/'PREFLIGHT.json').exists():
            save(out/'PREFLIGHT.json',dict(status='HOLD_PREFLIGHT',error=repr(exc),records=records,code=source,data_files=expected,batch_changed=False))
        raise
    finally: lock.close()


def prepare(cfg):
    out=Path(cfg['output']); assert not (out/'SEAL.json').exists()
    source=code(cfg); data_pins=data_files(cfg); files=dict(data_pins)
    for name in ('CPU_WITNESS.json','CUDA_WITNESS.json','PREFLIGHT.json'):
        receipt=read(out/name); assert receipt['status']=='PASS' and receipt['code']==source
        if name=='PREFLIGHT.json': assert receipt['data_files']==data_pins
        files[(out/name).as_posix()]=sha(out/name)
    assert read(out/'CPU_WITNESS.json')['real_data_rows_read']==0
    assert read(out/'CUDA_WITNESS.json')['CUDA_validated']
    import shutil
    assert shutil.disk_usage('.').free/2**30>=cfg['budget']['minimum_free_disk_GiB']
    save(out/'SEAL.json',dict(config=cfg,code=source,files=files,created_unix=time.time(),maximum_cases=2,
        automatic_expansion=False,Test_rows=0,outer_VAL_rows=0,formal_refit=False))


def check(cfg):
    seal=read(Path(cfg['output'])/'SEAL.json')
    assert seal['config']==cfg and seal['code']==code(cfg); verify(seal['files'])
    return seal


def directory(cfg,protocol):
    assert protocol in ('TS','OS')
    return Path(cfg['output'])/'trajectories'/protocol/'inner0'/MODEL/'seed17'


def worker(cfg,protocol):
    seal=check(cfg); lock=private_admission(cfg)
    folder=directory(cfg,protocol)
    try:
        assert not folder.exists(), 'No skipping or scientific retry'
        folder.mkdir(parents=True)
        started=time.monotonic(); save(folder/'STARTED.json',dict(pid=os.getpid(),unix=time.time(),source_pin_verified=True))
        data,truth=load_data(cfg,protocol,seal['files'])
        fy=data['tensors']['labels'][torch.as_tensor(data['train'],device='cuda')].cpu().numpy()
        weights=data['weights'].double().cpu().numpy(); state=new(cfg,data['prior'],17,'cuda')
        contract=batch_contract(cfg,data,17); save(folder/'BATCH_CONTRACT.json',contract)
        history=[]; snaps=[]; best=math.inf
        for epoch in range(1,161):
            assert time.monotonic()-started<10800, 'Worker budget exhausted'
            tick=time.monotonic(); rate=state['opt'].param_groups[0]['lr']
            before={n:v.detach().clone() for n,v in state['model'].named_parameters()}
            ebefore={n:v.detach().clone() for n,v in state['ema'].model.named_parameters()}
            order=np.random.default_rng(np.random.SeedSequence([17,epoch,0])).permutation(data['train']); num=den=0.
            for offset in range(0,len(order),256):
                assert time.monotonic()-started<10800, 'Worker budget exhausted'
                loss,mass=parent.update(state,data,order[offset:offset+256]); num+=loss*mass; den+=mass
            raw_delta=parent.relative_delta(state['model'],before); ema_delta=parent.relative_delta(state['ema'].model,ebefore)
            snaps.append(parent.snapshot(state,epoch)); snaps=snaps[-3:]
            rng=core.rng(); evaluation=parent.evaluation_model(state,snaps,epoch)
            fit=base.infer(evaluation,data,data['train'],batch_size=256)
            dev=base.infer(evaluation,data,data['evaluation'],batch_size=256)
            fm,_=metric(fit,fy,weights); dm,_=metric(dev,truth,weights)
            row=dict(epoch=epoch,stage='source',stage_epoch=epoch,learning_rate=rate,train_ce=num/den,
                     fit_ema=fm,dev_ema=dm,FIT_DEV_F1_gap_pp=fm['macro_f1_percent']-dm['macro_f1_percent'],
                     raw_relative_parameter_change=raw_delta,EMA_relative_parameter_change=ema_delta,
                     seconds=time.monotonic()-tick,global_step=state['step'])
            history.append(row)
            if dm['weighted_CE']<best:
                best=dm['weighted_CE']
                torch.save(dict(model=evaluation.state_dict(),epoch=epoch,model_name=MODEL,
                    prior=data['prior'],adapter='experiments.sensor_timesnet_pilot_20261029_v17.model.make_model',
                    batch_contract=contract),folder/'selected_best_DEV_CE.pt')
                np.save(folder/'selected_DEV_EMA.npy',dev); np.save(folder/'selected_FIT_EMA.npy',fit)
            state['scheduler'].step(dm['weighted_CE']); row['next_learning_rate']=state['opt'].param_groups[0]['lr']
            core.restore_rng(rng); save(folder/'history.json',history)
            parent.checkpoint(folder/'latest.pt',state,epoch=epoch,causal_ema_snapshots=snaps,prior=data['prior'],batch_contract=contract)
            stop=bounded_stop(history,cfg)
            save(folder/'PROGRESS.json',dict(protocol=protocol,epoch=epoch,selected_epoch=stop['selected_epoch'],
                DEV=dm,FIT_DEV_F1_gap_pp=row['FIT_DEV_F1_gap_pp'],stopping=stop,elapsed_seconds=time.monotonic()-started))
            print(dict(protocol=protocol,epoch=epoch,DEV=dm,stopping=stop),flush=True)
            if stop['stop']:
                torch.save(dict(model=evaluation.state_dict(),epoch=epoch,prior=data['prior'],batch_contract=contract),folder/'terminal_eval.pt')
                np.save(folder/'terminal_DEV_EMA.npy',dev)
            del evaluation,before,ebefore
            if stop['stop']: break
        chosen=selected(history)
        payload=torch.load(folder/'selected_best_DEV_CE.pt',map_location='cpu',weights_only=False)
        assert payload['epoch']==chosen['epoch']; np.testing.assert_array_equal(payload['prior'],data['prior'])
        replay=make_model(cfg,payload['prior']).to('cuda').eval().requires_grad_(False)
        replay.load_state_dict(payload['model'])
        for ids,name in ((data['evaluation'],'selected_DEV_EMA.npy'),(data['train'],'selected_FIT_EMA.npy')):
            np.testing.assert_array_equal(base.infer(replay,data,ids,batch_size=256),np.load(folder/name))
        opt=optimization(history,cfg['convergence'])
        gen=generalization(history,read('configs/sensor_comparator_joint_acceptance_20261027_v1.json')['neural'])
        check(cfg)
        assert time.monotonic()-started<10800, 'Worker budget exhausted before completion'
        names=['history.json','selected_best_DEV_CE.pt','selected_DEV_EMA.npy','selected_FIT_EMA.npy','latest.pt','terminal_eval.pt','terminal_DEV_EMA.npy','BATCH_CONTRACT.json']
        save(folder/'COMPLETE.json',dict(status='COMPLETE_DEVELOPMENT_REVIEW_REQUIRED',model=MODEL,protocol=protocol,inner=0,seed=17,
            epochs=len(history),stopping=stop,selected_checkpoint=chosen,terminal_checkpoint=history[-1],
            best_vs_terminal=dict(DEV_CE_change=history[-1]['dev_ema']['weighted_CE']-chosen['dev_ema']['weighted_CE'],
                DEV_F1_change_pp=history[-1]['dev_ema']['macro_f1_percent']-chosen['dev_ema']['macro_f1_percent'],
                selected_FIT_DEV_gap_pp=chosen['FIT_DEV_F1_gap_pp'],terminal_FIT_DEV_gap_pp=history[-1]['FIT_DEV_F1_gap_pp']),
            legacy_terminal_optimization=opt,legacy_terminal_generalization=gen,
            legacy_terminal_joint_pass=opt['passed'] and gen['passed'],
            scientific_training_stability_pass=stop['training_components']['passed'] and stop['validation_plateau'],
            selected_checkpoint_replay_exact=True,early_selected_checkpoint_stationarity_claimed=False,
            no_overfitting_claimed=False,independent_generalization=False,automatic_expansion=False,
            official_Test_rows_read=0,outer_VAL_rows_read=0,formal_adoption_allowed=False,batch_contract=contract,
            elapsed_seconds=time.monotonic()-started,files={(folder/n).as_posix():sha(folder/n) for n in names}))
    finally: lock.close()


def kill_own_process(proc):
    if proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL); proc.wait(timeout=10)


def run(cfg):
    check(cfg); out=Path(cfg['output'])
    # Exclusive marker: one controller only, no skip/retry, even after failures.
    marker=out/'LAUNCH.json'
    with marker.open('x',encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(),unix=time.time(),maximum_cases=2,attempts_per_case=1),stream)
    started=time.monotonic(); jobs=[]; active=None
    def interrupted(signum,frame):
        raise RuntimeError('Controller interrupted; drain own worker only, signal '+str(signum))
    previous={sig:signal.signal(sig,interrupted) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        for protocol in cfg['protocols']:
            assert not directory(cfg,protocol).exists(), 'No skipping or scientific retry'
            remaining=28800-(time.monotonic()-started); assert remaining>0
            log=out/(protocol+'_inner0_seed17.log')
            job=dict(protocol=protocol,inner=0,seed=17,status='RUNNING',log=log.as_posix()); jobs.append(job)
            save(out/'QUEUE_STATE.json',dict(status='RUNNING',jobs=jobs))
            with log.open('xb') as stream:
                active=subprocess.Popen([sys.executable,'-s','-u','-m',MODULE,'--mode','worker','--protocol',protocol],
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                try: active.wait(timeout=min(10800+120+30,remaining))
                except subprocess.TimeoutExpired:
                    kill_own_process(active); raise RuntimeError('HOLD_HARD_TIMEOUT_NO_RETRY')
            assert active.returncode==0, 'Worker failure, inspect '+str(log)
            receipt=read(directory(cfg,protocol)/'COMPLETE.json'); verify(receipt['files'])
            job.update(status='COMPLETE',epochs=receipt['epochs'],scientific_training_stability_pass=receipt['scientific_training_stability_pass'])
            active=None
        result=dict(status='REVIEW_REQUIRED',jobs=jobs,completed=2,automatic_expansion=False,
                    official_Test_rows_read=0,outer_VAL_rows_read=0,formal_adoption_allowed=False)
        save(out/'COMPLETE.json',result); save(out/'QUEUE_STATE.json',result)
    except BaseException as exc:
        if active: kill_own_process(active)
        save(out/'QUEUE_STATE.json',dict(status='HOLD_NO_RETRY',jobs=jobs,error=repr(exc)))
        raise
    finally:
        for sig,handler in previous.items(): signal.signal(sig,handler)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',required=True,choices=['cpu_witness','cuda_witness','preflight','prepare','run','worker'])
    parser.add_argument('--protocol',choices=['TS','OS'])
    args=parser.parse_args(); cfg=setup(args.mode in ('cuda_witness','preflight','worker'))
    if args.mode in ('cpu_witness','cuda_witness'): witness(cfg,args.mode=='cpu_witness')
    elif args.mode=='preflight': preflight(cfg)
    elif args.mode=='prepare': prepare(cfg)
    elif args.mode=='worker':
        assert args.protocol; worker(cfg,args.protocol)
    else: run(cfg)


if __name__=='__main__': main()
