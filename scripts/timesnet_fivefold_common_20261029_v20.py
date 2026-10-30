"""V20 own seals and data contract; only the frozen scorer opens Test targets."""
import ast
import hashlib
import json
from pathlib import Path

CONFIG = Path('configs/sensor_timesnet_fivefold_20261029_v20.json')
ROOT = Path('/q3/mmcows-sensor-behavior-20261007')
FIELDS = ('dynamic', 'state', 'state_valid', 'context', 'current')
SELECTION_NAMES = {'FIT.npz', 'DEV_INPUTS.npz', 'FIT_TARGETS.jsonl', 'DEV_TARGETS.jsonl', 'REFIT_SOURCE.json'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def config():
    """Configure this run without executing an old run's setup/check/config."""
    import numpy as np
    import torch
    assert Path.cwd() == ROOT
    assert (torch.__version__, np.__version__) == ('2.4.1+cu121', '1.24.4')
    cfg = read(CONFIG)
    assert cfg['models'] == ['TimesNet_DS'] and cfg['protocols'] == ['TS', 'OS']
    assert cfg['seeds'] == [17, 23, 29] and cfg['folds'] == ['fold_'+str(i) for i in range(1, 6)]
    assert cfg['selection_parent'] == 'reports/sensor_converged_fivefold_20261027_v1'
    assert cfg['formal_parent'] == 'reports/sensor_shared_fusion_fivefold_20261024_v1'
    assert cfg['timesnet'] == dict(d_model=16, d_ff=32, e_layers=2, top_k=3, num_kernels=4, dropout=.1)
    assert cfg['head_dropout'] == .1 and cfg['parameter_count'] == 403047
    assert cfg['learning_rate'] == .0003 and cfg['weight_decay'] == .03
    assert cfg['batch_size'] == cfg['eval_batch_size'] == 256
    assert cfg['EMA_decay'] == .998 and cfg['EMA_warmup_steps'] == 200 and cfg['evaluation_window'] == 3
    assert cfg['BN_momentum'] == .01 and cfg['refit_epoch_transfer'] == 'same_epochs'
    assert cfg['calibration_stopping'] == dict(minimum_epochs=40, maximum_epochs=160, patience_epochs=20, relative_improvement=.001)
    assert cfg['scheduler'] == dict(mode='min', factor=.2, patience=2, threshold=.0001, threshold_mode='rel', cooldown=0, min_lr=.000003, eps=0)
    assert cfg['threads'] == 4 and cfg['GPU_processes'] == cfg['budget']['gpu_parallel'] == 3
    assert cfg['runtime'] == dict(neural_minimum_free_MiB=6144, neural_fraction=.15, GPU_processes=3)
    assert cfg['Torch_memory_fraction_per_process'] == .15 and cfg['minimum_free_GPU_MiB'] == 6144
    assert cfg['budget']['worker_seconds'] == 10800 and cfg['budget']['prediction_worker_seconds'] == 3600
    assert cfg['budget']['queue_seconds'] == 172800 and cfg['budget']['max_attempts'] == 1
    assert cfg['budget']['cleanup_seconds'] == 120 and not cfg['new_paid_resources']
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    return cfg


def case_dir(cfg, stage, protocol, fold, model, seed):
    assert stage in ('calibration', 'refit', 'prediction')
    assert protocol in cfg['protocols'] and fold in cfg['folds'] and seed in cfg['seeds'] and model in cfg['models']
    return Path(cfg['output'])/stage/protocol/fold/model/f'seed{seed}'


def code(cfg):
    """Required own entrypoints plus recursively imported and dynamic vendor code.

    Output receipts are sources, never code: no hash refers back to a new receipt.
    Missing mandatory entrypoints fail closed even before they have been created.
    """
    paths = {CONFIG, Path(__file__).resolve().relative_to(Path.cwd()), Path(cfg['ledger']),
             Path(cfg['environment']), Path(cfg['model_config']),
             Path('scripts/run_timesnet_fivefold_20261029_v20.py'),
             Path('scripts/timesnet_fivefold_neural_20261029_v20.py'),
             Path('scripts/report_timesnet_fivefold_20261029_v20.py'),
             Path('tests/test_timesnet_fivefold_neural_20261029_v20.py'),
             Path('tests/test_timesnet_fivefold_queue_20261029_v20.py'),
             Path('scripts/verify_timesnet_pilot_results_20261029_v17.py'),
             Path('configs/sensor_comparator_joint_acceptance_20261027_v1.json')}
    for directory in ('experiments/sensor_timesnet_pilot_20261029_v17',
                      'experiments/sensor_comparator_pilot_20261022_v2/vendor',
                      'third_party/convtran_context_20261022_v1'):
        assert Path(directory).is_dir(), directory
        paths.update(p for p in Path(directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    pending = list(paths); scanned = set()
    while pending:
        path = pending.pop()
        assert path.is_file(), str(path)
        if path in scanned or path.suffix != '.py':
            continue
        scanned.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))):
            modules = []
            if isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = list(path.parent.parts)[:len(path.parent.parts)-node.level+1] if node.level else []
                base = '.'.join(prefix+([node.module] if node.module else []))
                if base:
                    modules = [base]+[base+'.'+item.name for item in node.names]
            for module in modules:
                base = Path(*module.split('.'))
                for candidate in (base.with_suffix('.py'), base/'__init__.py'):
                    if candidate.is_file() and candidate not in paths:
                        paths.add(candidate); pending.append(candidate)
    return {path.as_posix(): sha(path) for path in sorted(paths)}


def check(cfg):
    seal = read(Path(cfg['output'])/'SEAL.json')
    assert seal['config'] == cfg and seal['code'] == code(cfg)
    assert seal['manifest_sha256'] == sha(Path(cfg['output'])/'JOB_MANIFEST.json')
    expected_gates = {'CPU_WITNESS.json', 'CUDA_WITNESS.json', 'PREFLIGHT.json'}
    assert set(seal['gates']) == expected_gates
    for name, digest in seal['gates'].items():
        assert sha(Path(cfg['output'])/name) == digest, name
    for path, digest in seal['sources'].items():
        if not path.endswith(('.npz', '.pt')):
            assert sha(path) == digest, path
    return seal


def _receipt(cfg, protocol, fold):
    path = Path(cfg['selection_parent'])/'data'/protocol/fold/'RECEIPT.json'
    receipt = read(path)
    assert receipt['status'] == 'PASS' and receipt['selection_and_final_fit_ID_union_exact']
    assert receipt['protocol'] == protocol and receipt['fold'] == fold
    assert set(receipt['outputs']) == SELECTION_NAMES
    return path, receipt


def load_selection(cfg, protocol, fold, device='cuda'):
    import numpy as np
    import torch
    path, receipt = _receipt(cfg, protocol, fold); folder = path.parent
    for name, digest in receipt['outputs'].items():
        assert sha(folder/name) == digest, name
    with np.load(folder/'FIT.npz', allow_pickle=False) as fit, np.load(folder/'DEV_INPUTS.npz', allow_pickle=False) as dev:
        assert 'labels' not in dev.files
        truth = fit['labels'].copy(); counts = np.bincount(truth, minlength=7)
        np.testing.assert_array_equal(counts, receipt['fit_class_counts'])
        assert (counts > 0).all() and len(truth) == receipt['fit_rows']
        values = {key: np.concatenate([fit[key], dev[key]]) for key in FIELDS}
        assert len(dev['dynamic']) == receipt['dev_rows']
        labels = np.concatenate([truth, np.full(len(dev['dynamic']), -100, np.int64)])
    dev_truth = np.asarray([json.loads(line)['label']-1 for line in (folder/'DEV_TARGETS.jsonl').read_text().splitlines()])
    assert len(dev_truth) == receipt['dev_rows']
    tensors = {key: torch.as_tensor(value, device=device) for key, value in values.items()}
    tensors['labels'] = torch.as_tensor(labels, device=device)
    return dict(tensors=tensors, weights=torch.as_tensor(np.sqrt(counts.sum()/(7*counts)), dtype=torch.float32, device=device),
                prior=counts/counts.sum(), train=np.arange(len(truth)), evaluation=np.arange(len(truth), len(labels)),
                fit_truth=truth, dev_truth=dev_truth)


def _formal_receipt(cfg, protocol, fold):
    folder = Path(cfg['formal_parent'])/'data'/protocol/fold
    receipt = read(folder/'DATA_RECEIPT.json')
    assert receipt['phase'] == 'formal_refit' and not receipt['TEST_INPUTS_has_labels']
    return folder, receipt


def raw_refit(cfg, protocol, fold):
    import numpy as np
    folder, receipt = _formal_receipt(cfg, protocol, fold)
    assert sha(folder/'FIT.npz') == receipt['outputs']['FIT.npz']
    with np.load(folder/'FIT.npz', allow_pickle=False) as source:
        result = {key: source[key].copy() for key in FIELDS+('labels', 'prior', 'class_weights')}
    counts = np.bincount(result['labels'], minlength=7)
    np.testing.assert_array_equal(counts, receipt['fit_class_counts'])
    np.testing.assert_array_equal(result['prior'], counts/counts.sum())
    np.testing.assert_array_equal(result['class_weights'], np.sqrt(counts.sum()/(7*counts)))
    assert len(result['labels']) == receipt['fit_targets'] and (counts > 0).all()
    return result


def load_refit(cfg, protocol, fold, device='cuda'):
    import numpy as np
    import torch
    raw = raw_refit(cfg, protocol, fold); truth = raw['labels']
    tensors = {key: torch.as_tensor(raw[key], device=device) for key in FIELDS}
    tensors['labels'] = torch.as_tensor(truth, device=device)
    return dict(tensors=tensors, weights=torch.as_tensor(raw['class_weights'], dtype=torch.float32, device=device),
                prior=raw['prior'], train=np.arange(len(truth)), evaluation=np.array([], dtype=np.int64), fit_truth=truth)


def raw_test(cfg, protocol, fold):
    import numpy as np
    check_training_lock(cfg)
    folder, receipt = _formal_receipt(cfg, protocol, fold)
    assert sha(folder/'TEST_INPUTS.npz') == receipt['outputs']['TEST_INPUTS.npz']
    with np.load(folder/'TEST_INPUTS.npz', allow_pickle=False) as source:
        assert 'labels' not in source.files
        result = {key: source[key].copy() for key in FIELDS}; result['prior'] = source['prior'].copy()
    counts = np.asarray(receipt['fit_class_counts'])
    np.testing.assert_array_equal(result['prior'], counts/counts.sum())
    result['class_weights'] = np.sqrt(counts.sum()/(7*counts))
    assert len(result['dynamic']) == receipt['test_targets']
    return result


def load_test(cfg, protocol, fold, device='cuda'):
    import numpy as np
    import torch
    raw = raw_test(cfg, protocol, fold); count = len(raw['dynamic'])
    tensors = {key: torch.as_tensor(raw[key], device=device) for key in FIELDS}
    tensors['labels'] = torch.full((count,), -100, dtype=torch.long, device=device)
    return dict(tensors=tensors, weights=torch.as_tensor(raw['class_weights'], device=device, dtype=torch.float32),
                prior=raw['prior'], train=np.array([], dtype=np.int64), evaluation=np.arange(count))


def controller_gpu_lock(cfg):
    """Own the original global lock when all three caps plus reserve fit.

    Existing independently authorized GPU work may remain resident. Admission
    only reads capacity; it never signals another process or changes its limits.
    """
    import fcntl
    import math
    import subprocess
    handle = (ROOT/'reports/sensor_motion_sync_validation_20261020_v1/SINGLE_GPU.lock').open('a+b')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        line = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.total,memory.free', '--format=csv,noheader,nounits'],
            text=True).splitlines()[0]
        total, free = [int(value.strip()) for value in line.split(',')]
        assert cfg['Torch_memory_fraction_per_process'] == .15 and cfg['GPU_processes'] == 3
        assert cfg['minimum_free_GPU_MiB'] == 6144
        required = math.ceil(total * cfg['Torch_memory_fraction_per_process'] * cfg['GPU_processes']) + cfg['minimum_free_GPU_MiB']
        assert free >= required, f'Resource admission failed before scientific STARTED: free {free} MiB < required {required} MiB'
        return handle
    except BaseException:
        handle.close()
        raise


def gpu_lock(cfg, slot, kind='neural'):
    import fcntl
    import time
    from scripts.accelerate_comparator_queue_20261028_v8b import slot_gpu
    assert kind == 'neural' and slot in (0, 1, 2)
    with (ROOT/'reports/sensor_motion_sync_validation_20261020_v1/SINGLE_GPU.lock').open('a+b') as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(guard, fcntl.LOCK_UN)
            raise RuntimeError('V20 root controller must hold the global GPU lock')
    runtime = dict(runtime_output=cfg['runtime_output'], GPU_processes=3,
                   Torch_memory_fraction_per_process=.15, minimum_free_GPU_MiB=6144)
    deadline = time.monotonic()+cfg['admission_seconds']
    while True:
        try:
            return slot_gpu(runtime, slot)
        except AssertionError as exc:
            if 'Resource admission failed before scientific STARTED' not in str(exc) or time.monotonic() >= deadline:
                raise
            time.sleep(5)


def source_manifest(cfg):
    """Validate sealed TRAIN/VAL union=FIT; never open or hash Test byte streams."""
    selection_seal = Path(cfg['selection_parent'])/'TRAINING_SEAL.json'
    formal_seal = Path(cfg['formal_parent'])/'SEAL.json'
    old_selection, old_formal = read(selection_seal), read(formal_seal)
    assert old_selection['config']['output'] == cfg['selection_parent']
    assert old_formal['config']['output'] == cfg['formal_parent']
    files = {path.as_posix(): sha(path) for path in (selection_seal, formal_seal)}; cases = []
    for protocol in cfg['protocols']:
        for fold in cfg['folds']:
            path, receipt = _receipt(cfg, protocol, fold); folder = path.parent
            assert sha(path) == old_selection['data'][path.as_posix()]
            files[path.as_posix()] = sha(path)
            for name, digest in receipt['outputs'].items():
                source = folder/name
                assert sha(source) == digest == old_selection['data'][source.as_posix()]
                files[source.as_posix()] = digest
            reference = read(folder/'REFIT_SOURCE.json'); formal, fr = _formal_receipt(cfg, protocol, fold)
            frpath = formal/'DATA_RECEIPT.json'; digest = sha(frpath)
            assert digest == old_formal['receipts'][frpath.as_posix()]
            files[frpath.as_posix()] = digest
            assert reference['FIT_sha256'] == fr['outputs']['FIT.npz']
            assert reference['TEST_INPUTS_sha256'] == fr['outputs']['TEST_INPUTS.npz']
            for name in ('FIT.npz', 'FIT_TARGETS.jsonl'):
                source = formal/name; digest = fr['outputs'][name]
                assert sha(source) == digest == old_formal['data_files'][source.as_posix()]
                files[source.as_posix()] = digest
            key = lambda row: (row['cow_id'], row['timestamp'], row['label'])
            selected = []
            for name in ('FIT_TARGETS.jsonl', 'DEV_TARGETS.jsonl'):
                selected += [key(json.loads(line)) for line in (folder/name).read_text().splitlines()]
            fit = [key(json.loads(line)) for line in (formal/'FIT_TARGETS.jsonl').read_text().splitlines()]
            assert sorted(selected) == sorted(fit)
            assert len(fit) == fr['fit_targets'] == receipt['final_fit_rows'] == receipt['fit_rows']+receipt['dev_rows']
            cases.append(dict(protocol=protocol, fold=fold, calibration_fit_rows=receipt['fit_rows'],
                              calibration_val_rows=receipt['dev_rows'], final_fit_rows=fr['fit_targets'],
                              test_rows=fr['test_targets'], Test_targets_opened=False, Test_inputs_opened=False))
    return files, cases


def evidence_gate(cfg):
    """Recompute the retained V17 training checks; retain legacy HOLD as metadata."""
    from scripts.comparator_convergence_criteria_20261027_v2 import assess
    from scripts.comparator_repair_policy_20261028 import stopping, selected
    from scripts.run_comparator_stability_extension_20261028_v9 import training_components
    folder = Path(cfg['reference_output']); seal_path = folder/'SEAL.json'; seal = read(seal_path)
    old_cfg = seal['config']
    assert old_cfg['timesnet'] == cfg['timesnet'] and old_cfg['models'] == cfg['models']
    assert old_cfg['batch_size'] == old_cfg['eval_batch_size'] == 256
    for key in ('learning_rate', 'weight_decay', 'EMA_decay', 'EMA_warmup_steps',
                'BN_momentum', 'evaluation_window', 'scheduler', 'convergence'):
        assert old_cfg[key] == cfg[key], key
    assert old_cfg['stopping'] == cfg['calibration_stopping']
    files = {seal_path.as_posix(): sha(seal_path)}
    for section in ('code', 'files'):
        for name, digest in seal[section].items():
            assert sha(name) == digest, name
            files[name] = digest
    summary_path = folder/'VERIFIED_PILOT_SUMMARY.json'
    confirmation_path = folder/'CONVERGENCE_CONFIRMATION.json'
    summary, confirmation = read(summary_path), read(confirmation_path)
    assert summary['status'] == 'VERIFIED_PILOT_RESULTS' and summary['Test_rows_read'] == 0
    assert confirmation['status'] == 'RECOMPUTED_ORIGINAL_CRITERIA'
    files.update({path.as_posix(): sha(path) for path in (summary_path, confirmation_path)})
    records = []
    for protocol, expected_epoch, selected_epoch in (('TS', 40, 10), ('OS', 43, 14)):
        trajectory = folder/'trajectories'/protocol/'inner0'/'TimesNet_DS'/'seed17'
        complete_path, history_path = trajectory/'COMPLETE.json', trajectory/'history.json'
        receipt, history = read(complete_path), read(history_path)
        assert (receipt['protocol'], receipt['seed'], receipt['inner'], receipt['model']) == (protocol, 17, 0, 'TimesNet_DS')
        assert receipt['status'] == 'COMPLETE_DEVELOPMENT_REVIEW_REQUIRED'
        for path, digest in receipt['files'].items():
            assert sha(path) == digest, path
            files[path] = digest
        assert sha(history_path) == receipt['files'][history_path.as_posix()]
        assert len(history) == receipt['epochs'] == expected_epoch
        assert receipt['selected_checkpoint'] == selected(history) and selected(history)['epoch'] == selected_epoch
        assert receipt['terminal_checkpoint'] == history[-1]
        training = training_components(assess(history, old_cfg['convergence']), old_cfg['convergence'])
        plateau = stopping(history, old_cfg['stopping'])
        assert len(training['checks']) == 12 and all(training['checks'].values()) and training['passed']
        assert plateau['validation_plateau'] and receipt['stopping']['training_components'] == training
        recomputed = dict(plateau, stop=True, training_components=training,
                          reason='TRAINING_STABLE_AND_DEV_PLATEAU')
        assert receipt['stopping'] == recomputed
        assert receipt['scientific_training_stability_pass'] and receipt['selected_checkpoint_replay_exact']
        assert not receipt['early_selected_checkpoint_stationarity_claimed'] and not receipt['no_overfitting_claimed']
        assert receipt['official_Test_rows_read'] == receipt['outer_VAL_rows_read'] == 0
        verified = next(row for row in summary['records'] if row['protocol'] == protocol)
        proof = next(row for row in confirmation['records'] if row['protocol'] == protocol)
        assert verified['training_stability_pass'] and verified['validation_plateau'] and verified['stopping_recomputed_exact']
        assert proof['checks_passed'] == proof['checks_total'] == 12 and proof['empirical_training_stability']
        assert proof['epochs'] == expected_epoch and proof['selected_epoch'] == selected_epoch
        for path in (complete_path, history_path):
            recorded = {Path(name.replace('\\', '/')).name: digest for name, digest in proof['source_SHA256'].items()}
            assert sha(path) == recorded[protocol+('_COMPLETE.json' if path == complete_path else '_history.json')]
            files[path.as_posix()] = sha(path)
        records.append(dict(protocol=protocol, epochs=expected_epoch, selected_epoch=selected_epoch, checks_passed=12,
                            checks_total=12, validation_plateau=True, empirical_training_stability=True,
                            legacy_terminal_generalization_pass=receipt['legacy_terminal_generalization']['passed'],
                            early_selected_checkpoint_stationarity_claimed=False, no_overfitting_claimed=False))
    return dict(status='PASS', files=files, records=records, later_dropout_variants_adopted=False,
                boundary='Two V17 pilots only; training stability and DEV plateau, with early CE selection. Legacy generalization HOLD retained; no unseen-fold or no-overfit certificate.')


def check_training_lock(cfg):
    lock = read(Path(cfg['output'])/'TRAINING_LOCK.json')
    assert lock['seal_sha256'] == sha(Path(cfg['output'])/'SEAL.json')
    assert lock['status'] == 'FROZEN_TRAINING' and lock['eligible_models'] == cfg['models']
    for path, digest in lock['files'].items():
        assert sha(path) == digest, path
    expected = {case_dir(cfg, stage, p, f, model, seed).joinpath('COMPLETE.json').as_posix()
                for stage in ('calibration', 'refit') for p in cfg['protocols'] for f in cfg['folds']
                for model in cfg['models'] for seed in cfg['seeds']}
    assert expected.issubset(lock['files']), 'All 30 calibrations and 30 refits must be frozen before Test inputs'
    return lock
