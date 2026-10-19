"""Versioned cloud queue: official inner selection, locked refits and saved predictions."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
import random
import resource
import socket
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch

from q3_sensor.evaluation import metrics
from q3_sensor.protocol import MODALITIES, sha
from scripts.sensor_architecture_models_20261011 import make_model
from scripts.sensor_architecture_data_20261011 import FIELDS, frames_from_source, make_cache, parent_hashes
from scripts.run_sensor_history_controls_20261011 import read_json, write_json

CONFIG = Path('configs/sensor_architecture_comparison_20261011.json')
SOURCES = [CONFIG, Path(__file__), Path('scripts/sensor_architecture_models_20261011.py'),
    Path('scripts/sensor_architecture_data_20261011.py'),
    Path('scripts/summarize_sensor_architecture_comparison_20261011.py'),
    Path('scripts/launch_sensor_architecture_comparison_20261011.py'),
    Path('scripts/run_sensor_history_controls_20261011.py'), Path('scripts/train_sensor_baselines.py'),
    Path('scripts/run_sensor_formal_ts.py'), Path('scripts/sensor_os_protocol.py')] + sorted(Path('q3_sensor').glob('*.py'))


def setup():
    cfg = read_json(CONFIG)
    if Path.cwd() != Path('/q3/mmcows-sensor-behavior-20261007') or not torch.cuda.is_available():
        raise RuntimeError('Validated cloud project and CUDA required')
    if (torch.__version__, np.__version__, sklearn.__version__) != ('2.4.1+cu121', '1.24.4', '1.3.2'):
        raise RuntimeError('Warm environment changed')
    canonical = json.dumps(read_json(cfg['environment_spec']), sort_keys=True, separators=(',', ':')).encode()
    if hashlib.sha256(canonical).hexdigest() != cfg['environment_canonical_sha256']:
        raise RuntimeError('Environment declaration changed')
    torch.set_num_threads(cfg['threads'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.use_deterministic_algorithms(True)
    root = Path(cfg['output'])
    root.mkdir(parents=True, exist_ok=True)
    identity = dict(config=cfg, source_sha256={str(p): sha(p) for p in SOURCES},
        parent_sha256=parent_hashes(cfg), environment_spec_sha256=sha(cfg['environment_spec']),
        hostname=socket.gethostname(), gpu=torch.cuda.get_device_name(0),
        versions=dict(torch=torch.__version__, numpy=np.__version__, sklearn=sklearn.__version__))
    identity_path = root / 'run_identity.json'
    if identity_path.exists():
        if read_json(identity_path) != identity:
            raise RuntimeError('Immutable implementation or parents changed')
    else:
        write_json(identity_path, identity)
    jobs = []
    for phase in ('selection', 'formal'):
        for protocol in cfg['protocols']:
            for fold in cfg['folds']:
                for kind in cfg['models']:
                    suffixes = [f'lr{i}' for i in range(len(cfg['learning_rates']))] if phase == 'selection' else [f'seed{s}' for s in cfg['formal_seeds']]
                    jobs.extend(f'{phase}/{protocol}/{fold}/{kind}/{s}' for s in suffixes)
    manifest = dict(jobs=jobs, selection_jobs=80, formal_jobs=120, max_parallel=1,
        all_selection_complete_before_any_formal_model_Test_evaluation=True,
        identity_sha256=sha(identity_path), automatic_hyperparameter_change=False)
    path = root / 'MANIFEST.json'
    if path.exists() and read_json(path) != manifest:
        raise RuntimeError('Manifest differs')
    if not path.exists():
        write_json(path, manifest)
    return cfg, root


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_torch(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def cpu_state(model):
    return {k: value.detach().cpu().clone() for k, value in model.state_dict().items()}


def progress(root, job_id, **values):
    path = root / 'JOB_STATES.json'
    states = read_json(path) if path.exists() else {j: dict(state='pending') for j in read_json(root / 'MANIFEST.json')['jobs']}
    states[job_id] = dict(job_id=job_id, **values)
    write_json(path, states)
    counts = {s: sum(v['state'] == s for v in states.values()) for s in ('pending', 'running', 'completed', 'failed')}
    write_json(root / 'QUEUE_STATE.json', dict(current_job=states[job_id], counts=counts))


def batch_at(tensors, ids):
    index = torch.as_tensor(ids, dtype=torch.long, device='cuda')
    return {m: {k: value.index_select(0, index) for k, value in data.items()} for m, data in tensors.items()}


def predict(model, tensors, ids, size):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(batch_at(tensors, ids[at:at + size])).argmax(1).cpu().numpy() + 1
                               for at in range(0, len(ids), size)])


def load_data(directory):
    targets = pd.read_json(directory / 'targets.jsonl', lines=True, convert_dates=False)
    train = targets.split.eq('train').to_numpy()
    arrays = dict(np.load(directory / 'windows.npz', allow_pickle=False))
    tensors = {m: {k: torch.from_numpy(arrays[f'{m}_{k}']).to('cuda', dtype=torch.bool if k == 'valid' else torch.float32)
                   for k in FIELDS} for m in MODALITIES}
    receipt = read_json(directory / 'DATA_RECEIPT.json')
    return dict(targets=targets, train_ids=np.flatnonzero(train), evaluation_ids=np.flatnonzero(~train),
                tensors=tensors, labels=targets.label.to_numpy(dtype=int),
                normalizer=read_json(directory / 'normalizer.json'),
                identity=dict(directory=str(directory), receipt_sha256=sha(directory / 'DATA_RECEIPT.json'),
                              outputs=receipt['outputs']))


def sanity(cfg, root):
    seed_all(17)
    tensors = {}
    for m, dims in zip(MODALITIES, (3, 6, 1)):
        tensors[m] = dict(x=torch.randn(14, 60, dims, device='cuda'),
                          valid=torch.ones(14, 60, dims, dtype=torch.bool, device='cuda'),
                          fresh=torch.ones(14, 60, 1, device='cuda'),
                          age=torch.rand(14, 60, 1, device='cuda') * 59)
        tensors[m]['valid'][0] = False
        tensors[m]['valid'][1, :20] = False
        tensors[m]['x'][0] = 0
        tensors[m]['age'][0] = float('nan')
    records = []
    for kind in cfg['models']:
        seed_all(17)
        model = make_model(kind, [1 / 7] * 7).cuda().eval()
        count = sum(p.numel() for p in model.parameters())
        if count != cfg['expected_parameters'][kind]:
            raise AssertionError(f'{kind} actual parameters {count}')
        value = model(tensors)
        if not torch.isfinite(value).all():
            raise AssertionError('All-missing/prefix-missing output is nonfinite')
        torch.testing.assert_close(value[0], model.prior_logits, rtol=0, atol=0)
        prefix_error = None
        if kind != 'GRU':
            before = model.encoders['uwb'].sequence(tensors['uwb'])
            changed = {k: v.clone() for k, v in tensors['uwb'].items()}
            changed['x'][:, 30:] += 100
            changed['age'][:, 30:] = 0
            changed['valid'][:, 30:] = False
            after = model.encoders['uwb'].sequence(changed)
            prefix_error = float((before[:, :30] - after[:, :30]).abs().max())
            torch.testing.assert_close(before[:, :30], after[:, :30], rtol=1e-6, atol=1e-6)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        initial = cpu_state(model)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(tensors), torch.arange(14, device='cuda') % 7)
            if not torch.isfinite(loss):
                raise AssertionError('Nonfinite synthetic loss')
            loss.backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                raise AssertionError('Inactive or nonfinite gradient')
            optimizer.step()
        if not any(not torch.equal(value.cpu(), initial[key]) for key, value in model.state_dict().items()):
            raise AssertionError('No synthetic update')
        records.append(dict(model=kind, parameters=count, steps=3, finite_gradients=True,
                            all_missing_prior_exact=True, causal_prefix_max_error=prefix_error))
        del model, optimizer, initial
    write_json(root / 'SANITY.json', dict(status='PASS', synthetic_only=True, scientific_F1=False,
                                          records=records, identity_sha256=sha(root / 'run_identity.json')))
    print('ARCHITECTURE_SANITY_PASS', json.dumps(records), flush=True)
    gc.collect()
    torch.cuda.empty_cache()


def verify_job(directory, root, data_identity):
    complete = read_json(directory / 'COMPLETE.json')
    if complete['identity_sha256'] != sha(root / 'run_identity.json') or complete['data_identity'] != data_identity:
        raise AssertionError('Completed job identity differs')
    for name, digest in complete['outputs'].items():
        if sha(directory / name) != digest:
            raise AssertionError('Completed result changed')


def train_job(cfg, root, phase, protocol, fold, kind, suffix, data, learning_rate, seed, epochs):
    job_id = f'{phase}/{protocol}/{fold}/{kind}/{suffix}'
    directory = root / job_id
    if (directory / 'COMPLETE.json').exists():
        verify_job(directory, root, data['identity'])
        return read_json(directory / 'result.json')
    directory.mkdir(parents=True, exist_ok=True)
    job_identity = dict(job_id=job_id, learning_rate=learning_rate, seed=seed, epochs_cap=epochs,
                        data_identity=data['identity'], identity_sha256=sha(root / 'run_identity.json'))
    if (directory / 'INPUTS.json').exists() and read_json(directory / 'INPUTS.json') != job_identity:
        raise AssertionError('Partial job differs')
    write_json(directory / 'INPUTS.json', job_identity)
    seed_all(seed)
    rng = np.random.default_rng(seed)
    tr, ev, labels = data['train_ids'], data['evaluation_ids'], data['labels']
    counts = np.bincount(labels[tr], minlength=8)[1:8]
    prior = counts / counts.sum()
    weights = torch.as_tensor(np.sqrt(counts.sum() / (7 * counts)), dtype=torch.float32, device='cuda')
    y = torch.from_numpy(labels - 1).long().cuda()
    model = make_model(kind, prior).cuda()
    if sum(p.numel() for p in model.parameters()) != cfg['expected_parameters'][kind]:
        raise AssertionError('Parameter count changed')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=cfg['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    history, best, best_epoch, stale, first_epoch, elapsed = [], -1., 0, 0, 1, 0.
    resume_path = directory / 'RESUME.pt'
    if resume_path.exists():
        saved = torch.load(resume_path, map_location='cpu', weights_only=False)
        if saved['job_identity'] != job_identity:
            raise AssertionError('Resume checkpoint changed identity')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        history, best, best_epoch, stale = saved['history'], saved['best'], saved['best_epoch'], saved['stale']
        first_epoch, elapsed = saved['epoch'] + 1, saved['elapsed']
        rng.bit_generator.state = saved['numpy_generator']
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        if phase == 'selection' and len(history) >= cfg['minimum_training_epochs'] and stale >= cfg['patience']:
            first_epoch = epochs + 1
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    progress(root, job_id, state='running', pid=os.getpid(), epoch=first_epoch - 1)
    for epoch in range(first_epoch, epochs + 1):
        model.train()
        order = rng.permutation(tr)
        numerator = denominator = 0.
        epoch_start = time.monotonic()
        for at in range(0, len(order), cfg['batch_size']):
            idx = order[at:at + cfg['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_at(data['tensors'], idx))
            loss = criterion(logits, y[torch.as_tensor(idx, dtype=torch.long, device='cuda')])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            if epoch == 1 and at == 0 and not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError('Inactive/nonfinite first-batch gradient')
            optimizer.step()
            weight_sum = float(weights[y[torch.as_tensor(idx, dtype=torch.long, device='cuda')]].sum())
            numerator += float(loss.detach()) * weight_sum
            denominator += weight_sum
        record = dict(epoch=epoch, train_loss=numerator / denominator, epoch_seconds=time.monotonic() - epoch_start)
        if phase == 'selection':
            prediction = predict(model, data['tensors'], ev, cfg['batch_size'])
            score = metrics(labels[ev], prediction)['macro_f1']
            record['val_macro_f1'] = score
            if score > best:
                best, best_epoch, stale = score, epoch, 0
                save_torch(directory / 'BEST.pt', dict(state_dict=cpu_state(model), epoch=epoch, val_macro_f1=score))
            else:
                stale += 1
        history.append(record)
        elapsed_now = elapsed + time.monotonic() - start
        write_json(directory / 'history.json', history)
        save_torch(resume_path, dict(job_identity=job_identity, epoch=epoch, model=cpu_state(model),
            optimizer=optimizer.state_dict(), history=history, best=best, best_epoch=best_epoch, stale=stale,
            elapsed=elapsed_now, numpy_generator=rng.bit_generator.state, python_rng=random.getstate(),
            numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()))
        progress(root, job_id, state='running', pid=os.getpid(), epoch=epoch, train_loss=record['train_loss'],
                 val_macro_f1=record.get('val_macro_f1'), best_epoch=best_epoch if phase == 'selection' else None)
        print('ARCHITECTURE_EPOCH', job_id, json.dumps(record), flush=True)
        if phase == 'selection' and epoch >= cfg['minimum_training_epochs'] and stale >= cfg['patience']:
            break
    if phase == 'selection':
        selected = torch.load(directory / 'BEST.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(selected['state_dict'])
    else:
        best_epoch = epochs
    model.eval()
    evaluation_prediction = predict(model, data['tensors'], ev, cfg['batch_size'])
    fitted_prediction = predict(model, data['tensors'], tr, cfg['batch_size'])
    table = data['targets'].iloc[ev][['cow_id', 'timestamp', 'label']].reset_index(drop=True)
    table['prediction'] = evaluation_prediction
    table.to_csv(directory / 'predictions.csv', index=False)
    result = dict(job_identity, model=kind, protocol=protocol, fold=fold, phase=phase,
        parameters=cfg['expected_parameters'][kind], trained_epochs=len(history), selected_epoch=best_epoch,
        selected_validation_macro_f1=best if phase == 'selection' else None,
        budget_limited=bool(phase == 'selection' and len(history) == epochs and best_epoch >= epochs - cfg['patience'] + 1),
        stopped_by_patience=bool(phase == 'selection' and len(history) < epochs),
        prior=prior.tolist(), CE_weights=weights.cpu().tolist(), fit_class_counts=counts.tolist(),
        metrics=metrics(labels[ev], evaluation_prediction), Fit_metrics=metrics(labels[tr], fitted_prediction),
        seconds=elapsed + time.monotonic() - start, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        max_process_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    save_torch(directory / 'checkpoint.pt', dict(kind=kind, state_dict=cpu_state(model), prior=prior,
        normalizer=data['normalizer'], CE_weights=weights.cpu(), job_identity=job_identity,
        selected_epoch=best_epoch, config=cfg))
    write_json(directory / 'result.json', result)
    outputs = {p.name: sha(p) for p in directory.iterdir() if p.is_file() and p.name != 'COMPLETE.json'}
    write_json(directory / 'COMPLETE.json', dict(job_id=job_id, identity_sha256=sha(root / 'run_identity.json'),
        data_identity=data['identity'], outputs=outputs))
    progress(root, job_id, state='completed', trained_epochs=len(history), selected_epoch=best_epoch,
             macro_f1=result['metrics']['macro_f1'], seconds=result['seconds'])
    print('ARCHITECTURE_JOB_COMPLETE', job_id, result['metrics']['macro_f1'], best_epoch, result['seconds'], flush=True)
    del model, optimizer, weights, y
    gc.collect()
    return result


def lock_method(cfg, root):
    selected = {}
    candidates = []
    for protocol in cfg['protocols']:
        for fold in cfg['folds']:
            for kind in cfg['models']:
                results = []
                for i, lr in enumerate(cfg['learning_rates']):
                    path = root / 'selection' / protocol / fold / kind / f'lr{i}'
                    result = read_json(path / 'result.json')
                    verify_job(path, root, result['data_identity'])
                    if result['learning_rate'] != lr or result['seed'] != cfg['selection_seed']:
                        raise AssertionError('Wrong candidate learning rate or seed')
                    candidates.append(dict(job_id=result['job_id'], complete_sha256=sha(path / 'COMPLETE.json')))
                    results.append((i, result))
                index, best = min(results, key=lambda item: (-item[1]['selected_validation_macro_f1'], item[1]['selected_epoch'], item[0]))
                selected[f'{protocol}/{fold}/{kind}'] = dict(learning_rate=cfg['learning_rates'][index],
                    epochs=best['selected_epoch'], selection_job=best['job_id'],
                    validation_macro_f1=best['selected_validation_macro_f1'], budget_limited=best['budget_limited'])
    lock = dict(identity_sha256=sha(root / 'run_identity.json'), selected=selected, candidates=candidates,
                selection_jobs=80, formal_jobs=120, new_Test_predictions_before_lock=0,
                historical_Test_exposure=True, created_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    path = root / 'METHOD_LOCK.json'
    if path.exists():
        previous = read_json(path)
        if {k: v for k, v in previous.items() if k != 'created_UTC'} != {k: v for k, v in lock.items() if k != 'created_UTC'}:
            raise AssertionError('Method lock changed')
    else:
        write_json(path, lock)
    print('ARCHITECTURE_ALL_METHODS_LOCKED', flush=True)


def run_phase(cfg, root, phase):
    if read_json(root / 'SANITY.json')['status'] != 'PASS':
        raise RuntimeError('Numerical sanity gate missing')
    locked = None
    if phase == 'formal':
        locked = read_json(root / 'METHOD_LOCK.json')
        if locked['identity_sha256'] != sha(root / 'run_identity.json') or len(locked['candidates']) != 80:
            raise RuntimeError('Full selection lock required')
    frames = frames_from_source(dict(cfg, formal_source=cfg['source']))
    for protocol in cfg['protocols']:
        for fold in cfg['folds']:
            directory = make_cache(cfg, root, phase, protocol, fold, frames)
            data = load_data(directory)
            print('ARCHITECTURE_DATA_READY', phase, protocol, fold, len(data['train_ids']), len(data['evaluation_ids']), flush=True)
            for kind in cfg['models']:
                if phase == 'selection':
                    for i, lr in enumerate(cfg['learning_rates']):
                        train_job(cfg, root, phase, protocol, fold, kind, f'lr{i}', data, lr, cfg['selection_seed'], cfg['max_epochs'])
                else:
                    recipe = locked['selected'][f'{protocol}/{fold}/{kind}']
                    for seed in cfg['formal_seeds']:
                        train_job(cfg, root, phase, protocol, fold, kind, f'seed{seed}', data,
                                  recipe['learning_rate'], seed, recipe['epochs'])
            del data
            gc.collect()
            torch.cuda.empty_cache()
    write_json(root / f'{phase.upper()}_COMPLETE.json', dict(status='COMPLETE',
        expected_jobs=80 if phase == 'selection' else 120, identity_sha256=sha(root / 'run_identity.json')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['sanity', 'selection', 'lock', 'formal'])
    args = parser.parse_args()
    cfg, root = setup()
    with (root / '.RUNNING.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.phase == 'sanity':
                sanity(cfg, root)
            elif args.phase == 'lock':
                lock_method(cfg, root)
            else:
                run_phase(cfg, root, args.phase)
        except Exception:
            if (root / 'QUEUE_STATE.json').exists():
                current = read_json(root / 'QUEUE_STATE.json')['current_job']
                if current['state'] == 'running':
                    progress(root, current['job_id'], state='failed', traceback=traceback.format_exc())
            write_json(root / f'FAILURE_{time.time_ns()}.json', dict(phase=args.phase, traceback=traceback.format_exc()))
            raise


if __name__ == '__main__':
    main()
