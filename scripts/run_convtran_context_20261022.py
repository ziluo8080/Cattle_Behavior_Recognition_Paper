"""Cloud-only fixed20 two-rate ConvTran validation, gated before any launch."""
import argparse
import copy
import fcntl
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import numpy as np
import torch
from torch.nn import functional as F

from q3_sensor.endpoint_models import tokens
from q3_sensor.evaluation import metrics
from q3_sensor.protocol import MODALITIES
from scripts.sensor_convtran_context_model_20261022 import make_model
from scripts.sensor_native_immu_training_20261021 import load_selection, sha
from scripts.sensor_uwb_event_motion_20261020 import ParameterEMA

ROOT = Path('/q3/mmcows-sensor-behavior-20261007')
CONFIG = Path('configs/sensor_convtran_context_20261022_v1.json')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def configure():
    if Path.cwd() != ROOT or not torch.cuda.is_available():
        raise RuntimeError('Use the existing verified cloud CUDA workspace')
    if (torch.__version__, np.__version__) != ('2.4.1+cu121', '1.24.4'):
        raise RuntimeError('Warm runtime changed')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    cfg = read(CONFIG)
    assert cfg['epochs'] == 20 and cfg['fold'] == 'fold_1' and cfg['seeds'] == [17, 23, 29]
    assert cfg['arms'] == ['Original_ConvTran', 'TwoRate_Concat', 'TwoRate_Interaction']
    assert all(cfg[k] is False for k in ('Test_allowed', 'additional_folds_allowed', 'formal_refit_allowed', 'automatic_expansion'))
    return cfg


def acquire_gpu():
    handle = (ROOT / 'reports/sensor_motion_sync_validation_20261020_v1/SINGLE_GPU.lock').open('a+b')
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    memory = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
        '--format=csv,noheader,nounits'], text=True).splitlines()[0])
    if memory >= 500:
        raise RuntimeError('GPU occupied; no unrelated process will be stopped')
    return handle


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])


def data_for(cfg, protocol):
    if protocol not in cfg['protocols']:
        raise ValueError('Protocol outside locked pilot')
    data = load_selection(ROOT, protocol, cfg['fold'])
    parent = Path('reports/sensor_architecture_comparison_20261011/data/selection') / protocol / cfg['fold']
    receipt = read(parent / 'DATA_RECEIPT.json')
    assert receipt['phase'] == 'selection'
    freeze = read('refine-logs/sensor_motion_backbone_selection_20261020/REFERENCE_FREEZE.json')['selection_cache_SHA256']
    for name in ('targets.jsonl', 'normalizer.json', 'windows.npz', 'DATA_RECEIPT.json'):
        path = parent / name
        assert sha(path) == freeze[path.as_posix()]
    native_directory = Path('reports/sensor_native_immu_preflight_20261021_v2/data') / protocol / cfg['fold']
    assert (parent / 'targets.jsonl').read_bytes() == (native_directory / 'targets.jsonl').read_bytes()
    rows = [json.loads(line) for line in (parent / 'targets.jsonl').read_text().splitlines()]
    assert {r['split'] for r in rows} == {'train', 'val'}
    copies = Path(cfg['output']) / 'data' / protocol
    copies.mkdir(parents=True, exist_ok=True)
    target_file = copies / 'targets.jsonl'
    if target_file.exists():
        assert target_file.read_bytes() == (parent / 'targets.jsonl').read_bytes()
    else:
        target_file.write_bytes((parent / 'targets.jsonl').read_bytes())
    physical_masks, state_parts = [], []
    with np.load(parent / 'windows.npz', allow_pickle=False) as archive:
        for modality in MODALITIES:
            fields = {k: torch.as_tensor(archive[f'{modality}_{k}']) for k in ('x', 'valid', 'fresh', 'age')}
            fields['valid'] = fields['valid'].bool()
            fields['x'] = torch.where(fields['valid'], fields['x'], 0.)
            state_parts.append(tokens(fields).numpy())
            physical_masks.append(fields['valid'].any(-1).numpy())
    state = np.concatenate(state_parts, -1).astype(np.float32)
    state_valid = np.stack(physical_masks, -1).any(-1)
    assert state.shape == (len(rows), 60, 26) and np.isfinite(state).all()
    arrays = dict(dynamic=data['native'], context=data['context'], state=state,
                  state_valid=state_valid, current=state[:, -1].copy())
    tensors = {k: torch.as_tensor(v, device='cuda') for k, v in arrays.items()}
    counts = np.bincount(data['labels'][data['train']], minlength=7)
    assert len(counts) == 7 and (counts > 0).all()
    tensors['labels'] = torch.as_tensor(data['labels'], device='cuda')
    data.update(tensors=tensors, prior=counts / counts.sum(),
                weights=torch.tensor(np.sqrt(counts.sum() / (7 * counts)), dtype=torch.float32, device='cuda'))
    save(copies / 'DATA_RECEIPT.json', dict(phase='selection', protocol=protocol, fold=cfg['fold'],
        targets=len(rows), train=len(data['train']), val=len(data['evaluation']),
        target_sha256=sha(target_file), native_parent_receipt_sha256=data['receipt_sha256'],
        state_parent_receipt_sha256=sha(parent / 'DATA_RECEIPT.json'),
        state_valid_rule='any of ten physical field-valid flags; held values remain available',
        current_matches_last_state=True, state_shape=list(state.shape), Test_allowed=False))
    return data


def minibatch(data, ids):
    index = torch.as_tensor(ids, device='cuda')
    return {k: v[index] for k, v in data['tensors'].items() if k != 'labels'}


def update(model, optimizer, ema, data, ids):
    model.train(); optimizer.zero_grad(set_to_none=True)
    labels = data['tensors']['labels'][torch.as_tensor(ids, device='cuda')]
    loss = F.cross_entropy(model(minibatch(data, ids)), labels, weight=data['weights'])
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite CE')
    loss.backward()
    parameters = list(model.parameters())
    assert all(p.grad is not None for p in parameters), 'Disconnected gradient'
    assert torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all(), 'Nonfinite gradient'
    optimizer.step(); ema.update(model)
    assert torch.stack([torch.isfinite(p).all() for p in parameters]).all(), 'Nonfinite weight'
    return float(loss.detach()), float(data['weights'][labels].sum())


@torch.no_grad()
def evaluate(model, data, arm, ids, batch_size):
    model.eval(); predictions = []; numerator = denominator = 0.
    for offset in range(0, len(ids), batch_size):
        chosen = ids[offset:offset + batch_size]
        logits = model(minibatch(data, chosen))
        assert torch.isfinite(logits).all()
        labels = data['tensors']['labels'][torch.as_tensor(chosen, device='cuda')]
        numerator += float(F.cross_entropy(logits, labels, weight=data['weights'], reduction='sum'))
        denominator += float(data['weights'][labels].sum())
        predictions.append(logits.argmax(1).cpu().numpy() + 1)
    prediction = np.concatenate(predictions)
    score = metrics(data['labels'][ids] + 1, prediction)
    score['weighted_ce'] = numerator / denominator
    return prediction, score


def tracked_files():
    files = [CONFIG, Path(__file__), Path('scripts/sensor_convtran_context_model_20261022.py'),
             Path('scripts/report_convtran_context_20261022.py'),
             Path('scripts/sensor_native_convtran_20261021.py'),
             Path('scripts/sensor_native_immu_training_20261021.py'),
             Path('scripts/sensor_native_immu_runner_20261021.py'),
             Path('scripts/sensor_native_immu_selection_policy_20261021.py'),
             Path('scripts/sensor_uwb_event_motion_20261020.py'),
             Path('configs/sensor_gru_margin_targets_20261020_v3.json')]
    files += list(Path('q3_sensor').glob('*.py'))
    files += list(Path('third_party/convtran_20261021_v1/Models').glob('*.py'))
    return {Path(p).resolve().relative_to(ROOT).as_posix(): sha(p) for p in files}


def preflight(cfg):
    lock = acquire_gpu(); out = Path(cfg['output'])
    assert not (out / 'SEAL.json').exists()
    records = []
    for protocol in cfg['protocols']:
        data = data_for(cfg, protocol)
        counts = len(data['train'])
        for arm in cfg['arms']:
            seed_all(20261022)
            model = make_model(arm, cfg, data['prior']).cuda()
            opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'], eps=1e-8)
            ema = ParameterEMA(model, cfg['ema_decay'], cfg['ema_warmup_steps'])
            started = time.perf_counter(); losses = []
            for i in range(3):
                losses.append(update(model, opt, ema, data, data['train'][i*256:(i+1)*256])[0])
            path = out / 'preflight' / f'{protocol}_{arm}.pt'; path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), ema=ema.model.state_dict(),
                ema_updates=ema.updates, rng=rng()), path)
            expected_loss = update(model, opt, ema, data, data['train'][:256])[0]
            payload = torch.load(path, weights_only=False)
            other = make_model(arm, cfg, data['prior']).cuda(); other.load_state_dict(payload['model'])
            other_opt = torch.optim.AdamW(other.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'], eps=1e-8)
            other_opt.load_state_dict(payload['optimizer'])
            other_ema = ParameterEMA(other, cfg['ema_decay'], cfg['ema_warmup_steps'])
            other_ema.model.load_state_dict(payload['ema']); other_ema.updates = payload['ema_updates']
            restore_rng(payload['rng'])
            assert update(other, other_opt, other_ema, data, data['train'][:256])[0] == expected_loss
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, other.state_dict()[key], rtol=0, atol=0)
            for key, value in ema.model.state_dict().items():
                torch.testing.assert_close(value, other_ema.model.state_dict()[key], rtol=0, atol=0)
            records.append(dict(protocol=protocol, arm=arm, parameters=sum(p.numel() for p in model.parameters()),
                train_ce=losses, serialized_resume_bitwise=True, short_steps_elapsed=time.perf_counter()-started,
                train_targets=counts, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()))
            del model, opt, ema, other, other_opt, other_ema, payload
        del data
    save(out / 'CUDA_PREFLIGHT.json', dict(status='PASS', records=records, files=tracked_files(),
         scientific_VAL_scores_computed=0, Test_allowed=False, gpu=torch.cuda.get_device_name(0)))
    lock.close()


def check_seal(cfg):
    seal = read(Path(cfg['output']) / 'SEAL.json')
    assert seal['config'] == cfg and seal['files'] == tracked_files()
    for name, digest in seal['data'].items():
        assert sha(name) == digest


def seal(cfg):
    out = Path(cfg['output'])
    assert not (out / 'SEAL.json').exists()
    witness = read(out / 'CUDA_PREFLIGHT.json')
    assert witness['status'] == 'PASS' and witness['files'] == tracked_files()
    assert read(out / 'CODE_REVIEW_GATE.json')['status'] == 'PASS'
    assert read(out / 'HOST_TRUST_RECEIPT.json')['trusted_fingerprint_verified'] is True
    data = {}
    for protocol in cfg['protocols']:
        directory = Path('reports/sensor_native_immu_preflight_20261021_v2/data') / protocol / cfg['fold']
        receipt = read(directory / 'DATA_RECEIPT.json')
        for name in ('DATA_RECEIPT.json', 'context.npy', 'normalizer.json', 'targets.jsonl'):
            path = directory / name; data[path.as_posix()] = sha(path)
        path = Path(receipt['native_inputs_parent']); data[path.as_posix()] = sha(path)
        parent = Path('reports/sensor_architecture_comparison_20261011/data/selection') / protocol / cfg['fold']
        for name in ('DATA_RECEIPT.json', 'windows.npz', 'normalizer.json', 'targets.jsonl'):
            path = parent / name; data[path.as_posix()] = sha(path)
    for name in ('CUDA_PREFLIGHT.json', 'CODE_REVIEW_GATE.json', 'HOST_TRUST_RECEIPT.json'):
        path = out / name; data[path.as_posix()] = sha(path)
    save(out / 'SEAL.json', dict(config=cfg, files=tracked_files(), data=data,
         created_at_unix=time.time(), Test_allowed=False))


def worker(cfg, protocol, arm):
    check_seal(cfg)
    assert protocol in cfg['protocols'] and arm in cfg['arms']
    lock = acquire_gpu()
    directory = Path(cfg['output']) / 'groups' / protocol / arm
    directory.mkdir(parents=True, exist_ok=True)
    assert not (directory / 'STARTED.json').exists(), 'No automatic retries'
    save(directory / 'STARTED.json', dict(pid=os.getpid(), started_at_unix=time.time()))
    data = data_for(cfg, protocol); history = {}; final = {}; started = time.perf_counter()
    for seed in cfg['seeds']:
        seed_all(seed); model = make_model(arm, cfg, data['prior']).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'], eps=1e-8)
        ema = ParameterEMA(model, cfg['ema_decay'], cfg['ema_warmup_steps']); history[str(seed)] = []
        for epoch in range(1, cfg['epochs'] + 1):
            order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(data['train'])
            numerator = denominator = 0.; tick = time.perf_counter()
            for i in range(0, len(order), cfg['batch_size']):
                loss, weight = update(model, opt, ema, data, order[i:i+cfg['batch_size']])
                numerator += loss * weight; denominator += weight
            record = dict(epoch=epoch, train_ce=numerator / denominator, validation={})
            for view, network in (('raw', model), ('ema', ema.model)):
                pred, score = evaluate(network, data, arm, data['evaluation'], cfg['batch_size'])
                record['validation'][view] = score
                if epoch == 20:
                    np.save(directory / f'seed{seed}_{view}_final20_predictions.npy', pred)
            record['elapsed_seconds'] = time.perf_counter() - tick
            history[str(seed)].append(record); save(directory / 'HISTORY.json', history)
            temp = directory / f'seed{seed}_latest.tmp'
            torch.save(dict(epoch=epoch, seed=seed, arm=arm, protocol=protocol, model=model.state_dict(),
                ema=ema.model.state_dict(), optimizer=opt.state_dict(), ema_updates=ema.updates, rng=rng()), temp)
            temp.replace(directory / f'seed{seed}_latest.pt')
            print(json.dumps(dict(protocol=protocol, arm=arm, seed=seed, epoch=epoch,
                raw_f1=record['validation']['raw']['macro_f1'], train_ce=record['train_ce'], seconds=record['elapsed_seconds'])), flush=True)
        payload = torch.load(directory / f'seed{seed}_latest.pt', weights_only=False)
        for view, key in (('raw', 'model'), ('ema', 'ema')):
            other = make_model(arm, cfg, data['prior']).cuda(); other.load_state_dict(payload[key])
            pred, score = evaluate(other, data, arm, data['evaluation'], cfg['batch_size'])
            np.testing.assert_array_equal(pred, np.load(directory / f'seed{seed}_{view}_final20_predictions.npy'))
            assert score == history[str(seed)][-1]['validation'][view]
            if arm == 'Original_ConvTran':
                old = Path('reports/sensor_native_immu_pilot_20261021_v1/groups') / protocol / 'ConvTran_Native_FeatureFusion'
                np.testing.assert_array_equal(pred, np.load(old / f'seed{seed}_{view}_final20_predictions.npy'))
            del other
        final[str(seed)] = history[str(seed)][-1]['validation']; del model, opt, ema, payload
    save(directory / 'COMPLETE.json', dict(status='PASS', final=final, checkpoint_prediction_replays=6,
         baseline_replay_passed=arm == 'Original_ConvTran', elapsed_seconds=time.perf_counter()-started,
         parameters=sum(p.numel() for p in make_model(arm, cfg, data['prior']).parameters()), Test_allowed=False))
    lock.close()


def supervise(cfg):
    check_seal(cfg); out = Path(cfg['output']); lock = (out / 'QUEUE.lock').open('a+b')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (out / 'RUN_START.json').exists()
    started = time.time(); jobs = [dict(protocol=p, arm=a, status='pending') for a in cfg['arms'] for p in cfg['protocols']]
    save(out / 'RUN_START.json', dict(pid=os.getpid(), started_at_unix=started))
    for i, job in enumerate(jobs):
        remaining = cfg['max_wall_seconds'] - (time.time() - started)
        if remaining <= 0: raise RuntimeError('Fixed wall budget exhausted')
        job.update(status='running', attempts=1)
        save(out / 'queue_state.json', dict(jobs=jobs, max_attempts=1))
        save(out / 'PUBLIC_PROGRESS.json', dict(status='RUNNING', completed_groups=i, total_groups=len(jobs), current=job))
        log = out / 'logs' / f'{job["protocol"]}_{job["arm"]}.log'; log.parent.mkdir(exist_ok=True)
        try:
            with log.open('wb') as stream:
                result = subprocess.run([sys.executable, '-s', '-m', 'scripts.run_convtran_context_20261022', 'worker',
                    '--protocol', job['protocol'], '--arm', job['arm']], stdout=stream, stderr=subprocess.STDOUT, timeout=remaining)
            if result.returncode:
                raise RuntimeError(f'Worker failed: {job}; see {log}')
        except BaseException:
            job['status'] = 'failed'; save(out / 'queue_state.json', dict(jobs=jobs, max_attempts=1)); raise
        assert read(out / 'groups' / job['protocol'] / job['arm'] / 'COMPLETE.json')['status'] == 'PASS'
        job['status'] = 'completed'; save(out / 'queue_state.json', dict(jobs=jobs, max_attempts=1))
    from scripts.report_convtran_context_20261022 import report
    report(out, cfg)
    save(out / 'PUBLIC_PROGRESS.json', dict(status='COMPLETE_STOP_FOR_REVIEW', completed_groups=6, total_groups=6,
                                          Test_allowed=False, automatic_expansion=False))


def launch(cfg):
    check_seal(cfg); out = Path(cfg['output'])
    assert not (out / 'LAUNCH.json').exists()
    with (out / 'SUPERVISOR.log').open('wb') as stream:
        process = subprocess.Popen([sys.executable, '-s', '-m', 'scripts.run_convtran_context_20261022', 'supervise'],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = dict(pid=process.pid, launched_at_unix=time.time(), Test_allowed=False, automatic_expansion=False)
    save(out / 'LAUNCH.json', receipt); print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['preflight', 'seal', 'worker', 'supervise', 'launch'])
    parser.add_argument('--protocol', choices=['TS', 'OS']); parser.add_argument('--arm')
    args = parser.parse_args(); cfg = configure()
    try:
        if args.phase == 'preflight': preflight(cfg)
        elif args.phase == 'seal': seal(cfg)
        elif args.phase == 'worker': worker(cfg, args.protocol, args.arm)
        elif args.phase == 'launch': launch(cfg)
        else: supervise(cfg)
    except BaseException:
        save(Path(cfg['output']) / f'FAILURE_{time.time_ns()}.json', dict(traceback=traceback.format_exc(), automatic_retry=False))
        save(Path(cfg['output']) / 'PUBLIC_PROGRESS.json', dict(status='STOPPED_FAILURE', Test_allowed=False))
        raise
