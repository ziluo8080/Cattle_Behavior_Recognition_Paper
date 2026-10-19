"""Resumable cloud-only history controls with immutable parents and job receipts."""
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

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.ensemble import RandomForestClassifier

from q3_sensor.endpoint_models import EndpointFusion
from q3_sensor.evaluation import metrics
from q3_sensor.protocol import MODALITIES, build_windows, fit_normalizer, sha
from scripts.run_sensor_formal_ts import official_roles
from scripts.sensor_history_control_models_20261011 import (
    FIELDS, CurrentRepeatGRU, available_rows, repeat_current, window_rf_features)
from scripts.sensor_os_protocol import assert_window_sources, cow_roles, day_role, os_normalizer
from scripts.train_sensor_baselines import tensor_batch

CONFIG = Path('configs/sensor_history_controls_20261011.json')
TRACKED = [CONFIG, Path(__file__), Path('scripts/sensor_history_control_models_20261011.py'),
           Path('scripts/run_sensor_formal_ts.py'), Path('scripts/sensor_os_protocol.py'),
           Path('scripts/train_sensor_baselines.py')] + sorted(Path('q3_sensor').glob('*.py'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def setup():
    cfg = read_json(CONFIG)
    if Path.cwd() != Path('/q3/mmcows-sensor-behavior-20261007') or not torch.cuda.is_available():
        raise RuntimeError('Validated cloud project and CUDA required')
    if (torch.__version__, np.__version__, sklearn.__version__) != ('2.4.1+cu121', '1.24.4', '1.3.2'):
        raise RuntimeError('Environment differs from warm-reuse contract')
    digest = hashlib.sha256(json.dumps(read_json(cfg['environment_spec']), sort_keys=True,
                                      separators=(',', ':')).encode()).hexdigest()
    if digest != cfg['environment_canonical_sha256']:
        raise RuntimeError('Environment declaration changed')
    torch.set_num_threads(cfg['threads'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    root = Path(cfg['output'])
    root.mkdir(parents=True, exist_ok=True)
    identity = dict(config=cfg, sources={str(p): sha(p) for p in TRACKED},
                    hostname=socket.gethostname(), versions=dict(torch=torch.__version__,
                    numpy=np.__version__, sklearn=sklearn.__version__, joblib=joblib.__version__),
                    gpu=torch.cuda.get_device_name(0), environment_spec_sha256=sha(cfg['environment_spec']))
    identity_path = root / 'run_identity.json'
    if identity_path.exists() and read_json(identity_path) != identity:
        raise RuntimeError('Run identity changed; cannot overwrite or resume')
    if not identity_path.exists():
        write_json(identity_path, identity)
    jobs = [f'{scope}/{fold}/{system}_seed{seed}'
            for scope, folds in [('development', ['fold_1'])] + [(p, cfg['folds']) for p in cfg['protocols']]
            for fold in folds for system in cfg['systems'] for seed in cfg['seeds']]
    manifest = dict(identity_sha256=sha(identity_path), jobs=jobs, max_parallel=1,
                    dependent_formal_phase_requires_successful_development_lock=True,
                    OOM_max_attempts=1, hyperparameter_changes=False)
    if (root / 'MANIFEST.json').exists() and read_json(root / 'MANIFEST.json') != manifest:
        raise RuntimeError('Queue manifest changed')
    if not (root / 'MANIFEST.json').exists():
        write_json(root / 'MANIFEST.json', manifest)
    return cfg, root


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def predict(model, batch, indices, size):
    with torch.no_grad():
        return np.concatenate([model(tensor_batch(batch, indices[i:i + size], 'cuda'))
                               .argmax(1).cpu().numpy() + 1 for i in range(0, len(indices), size)])


def sanity(cfg, root):
    seed_all(17)
    batch = {}
    for m, dims in zip(MODALITIES, (3, 6, 1)):
        batch[m] = dict(x=torch.randn(14, 60, dims), valid=torch.ones(14, 60, dims, dtype=torch.bool),
                        fresh=torch.ones(14, 60, 1), age=torch.rand(14, 60, 1) * 50)
        batch[m]['valid'][0, -1] = False
    altered = {m: {k: v.clone() for k, v in d.items()} for m, d in batch.items()}
    for m in MODALITIES:
        altered[m]['x'][:, :-1] *= 100
        altered[m]['valid'][:, :-1] = False
        altered[m]['fresh'][:, :-1] = 0
        altered[m]['age'][:, :-1] = float('nan')
    model = CurrentRepeatGRU([1 / 7] * 7).cuda().eval()
    if sum(p.numel() for p in model.parameters()) != cfg['expected_parameters']:
        raise AssertionError('Parameter mismatch')
    with torch.no_grad():
        a = model(tensor_batch(batch, np.arange(14), 'cuda'))
        b = model(tensor_batch(altered, np.arange(14), 'cuda'))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(a[0], model.prior_logits, rtol=0, atol=0)
    expanded = repeat_current(batch)
    for m in MODALITIES:
        for k in FIELDS:
            expected = batch[m][k][:, -1:, :].expand(-1, 60, -1)
            torch.testing.assert_close(expanded[m][k], expected, rtol=0, atol=0, equal_nan=True)
    features = window_rf_features(batch)
    if np.array_equal(features, window_rf_features(altered)):
        raise AssertionError('RF did not receive full history')
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'])
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(tensor_batch(batch, np.arange(14), 'cuda')),
                                                torch.arange(14, device='cuda') % 7)
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
            raise AssertionError('Inactive or nonfinite parameter gradient')
        for m in MODALITIES:
            if not any(p.grad.abs().sum() > 0 for p in model.encoders[m].gru.parameters()):
                raise AssertionError('Inactive recurrent branch')
        optimizer.step()
    if not any(not torch.equal(v, initial[k]) for k, v in model.state_dict().items()):
        raise AssertionError('No optimizer update')
    write_json(root / 'SANITY.json', dict(status='PASS', synthetic_only=True, scientific_F1=False,
               steps=3, parameters=cfg['expected_parameters'], history_invariance_exact=True,
               endpoint_only_fallback=True, RF_features=features.shape[1], all_gradients_finite=True,
               identity_sha256=sha(root / 'run_identity.json')))
    print('HISTORY_CONTROLS_SANITY_PASS', flush=True)


def to_tensors(windows):
    return {m: {k: torch.from_numpy(windows[m][k]).bool() if k == 'valid'
                else torch.from_numpy(windows[m][k]) for k in FIELDS} for m in MODALITIES}


def developmental(cfg):
    source = Path(cfg['development_source'])
    arrays = dict(np.load(source / 'windows.npz', allow_pickle=False))
    targets = pd.read_json(source / 'targets.jsonl', lines=True, convert_dates=False)
    for key in ('cow_id', 'timestamp', 'label'):
        np.testing.assert_array_equal(targets[key].to_numpy(), arrays[key])
    train = arrays['train'].astype(bool)
    if set(targets.split) != {'train', 'val'}:
        raise AssertionError('Development contains non TRAIN/VAL targets')
    np.testing.assert_array_equal(train, targets.split.eq('train').to_numpy())
    tensors = to_tensors({m: {k: arrays[f'{m}_{k}'] for k in FIELDS} for m in MODALITIES})
    return dict(targets=targets, train=train, tensors=tensors, normalizer=read_json(source / 'normalizer.json'),
                parent_files={str(p): sha(p) for p in [source / 'windows.npz', source / 'targets.jsonl',
                                                     source / 'normalizer.json', source / 'build_receipt.json']})


def raw_frames(cfg):
    source = Path(cfg['formal_source'])
    receipt = read_json(source / 'source_receipt.json')
    for record in receipt['members']:
        member = source / Path(record['member']).relative_to('sensor_data')
        if sha(member) != record['sha256']:
            raise AssertionError('Raw source receipt mismatch')
    frames = {}
    for cow in range(1, 11):
        tag, name = f'T{cow:02d}', f'C{cow:02d}'
        frames[cow] = dict(uwb=pd.read_csv(source / f'main_data/uwb/{tag}/{tag}_0725.csv'),
                          hd=pd.read_csv(source / f'sub_data/head_direction/{tag}/{tag}_0725.csv'),
                          akl=pd.read_csv(source / f'main_data/ankle/{name}/{name}_0725.csv'))
    return frames


def formal_data(cfg, protocol, fold, frames):
    parent = Path(cfg['parent_roots'][protocol]) / fold
    complete = read_json(parent / 'COMPLETE.json')
    if complete['target_sha256'] != sha(parent / 'targets.jsonl'):
        raise AssertionError('Parent target changed')
    if complete['identity_sha256'] != sha(parent.parent / 'run_identity.json'):
        raise AssertionError('Parent identity mismatch')
    original = read_json(parent.parent / 'run_identity.json')
    for path, digest in original['sources'].items():
        if path.startswith('q3_sensor/') and sha(path) != digest:
            raise AssertionError(f'Frozen parent source changed {path}')
    targets = pd.read_json(parent / 'targets.jsonl', lines=True, convert_dates=False)
    if targets.duplicated(['cow_id', 'timestamp']).any():
        raise AssertionError('Duplicate parent targets')
    train = targets.split.eq('train').to_numpy()
    if protocol == 'TS':
        official = read_json('configs/official_ts.json')
        roles = official_roles(official, official['folds'][fold])
        normalizer = fit_normalizer(frames, roles)
        for row in targets.itertuples():
            if not any(r['split'] == row.split and r['start'] <= row.timestamp < r['end'] for r in roles):
                raise AssertionError('Illegal TS role')
    else:
        official = read_json('configs/official_os.json')
        _, fit_ids, test_ids = cow_roles(official['folds'][fold])
        if set(targets.loc[train, 'cow_id']) != set(fit_ids) or set(targets.loc[~train, 'cow_id']) != set(test_ids):
            raise AssertionError('OS cow split mismatch')
        normalizer = os_normalizer(frames, fit_ids, day_role(read_json('configs/sensor_os_20261007.json')))
    if normalizer != read_json(parent / 'normalizer.json'):
        raise AssertionError('Recomputed Fit normalizer differs from parent')
    labels_dir = Path(cfg['formal_source']) / 'behavior_labels/individual'
    for cow, group in targets.groupby('cow_id'):
        gt = pd.read_csv(labels_dir / f'C{cow:02d}_0725.csv').set_index('timestamp').behavior
        np.testing.assert_array_equal(gt.loc[group.timestamp].to_numpy(dtype=int), group.label.to_numpy())
    windows = build_windows(targets.to_dict('records'), frames, normalizer)
    assert_window_sources(targets, windows)
    tensors = to_tensors(windows)
    del windows
    expected = targets.loc[~train, ['cow_id', 'timestamp', 'label']].reset_index(drop=True)
    baseline = pd.read_csv(parent / 'predictions.csv')
    pd.testing.assert_frame_equal(baseline[['cow_id', 'timestamp', 'label']], expected)
    parent_files = {str(p): sha(p) for p in [parent / 'targets.jsonl', parent / 'normalizer.json',
                    parent / 'predictions.csv', parent / 'COMPLETE.json', parent.parent / 'run_identity.json']}
    replay = []
    counts = np.bincount(targets.label.to_numpy()[train], minlength=8)[1:8]
    for seed in cfg['seeds']:
        path = parent / f'NN_seed{seed}.pt'
        if sha(path) != complete['checkpoint_sha256'][path.name]:
            raise AssertionError('Parent checkpoint changed')
        parent_files[str(path)] = sha(path)
        state = torch.load(path, map_location='cpu', weights_only=False)
        model = EndpointFusion('gru', counts / counts.sum()).cuda().eval()
        model.load_state_dict(state['state_dict'], strict=True)
        prediction = predict(model, tensors, np.flatnonzero(~train), cfg['batch_size'])
        np.testing.assert_array_equal(prediction, baseline[f'NN_seed{seed}'].to_numpy())
        replay.append(dict(seed=seed, targets=len(prediction), hard_mismatches=0))
        del model, state
    gc.collect()
    torch.cuda.empty_cache()
    return dict(targets=targets, train=train, tensors=tensors, normalizer=normalizer,
                parent_files=parent_files, parent_replay=replay)


def input_identity(data):
    hashes = {}
    for m in MODALITIES:
        for k in FIELDS:
            hashes[f'{m}_{k}'] = hashlib.sha256(data['tensors'][m][k].numpy().tobytes()).hexdigest()
    return dict(parent_files=data['parent_files'], tensor_sha256=hashes,
                fit_targets=int(data['train'].sum()), evaluation_targets=int((~data['train']).sum()),
                fit_class_counts=np.bincount(data['targets'].label.to_numpy()[data['train']], minlength=8)[1:8].tolist())


def set_progress(root, progress):
    path = root / 'JOB_STATES.json'
    states = read_json(path) if path.exists() else {j: dict(state='pending') for j in read_json(root / 'MANIFEST.json')['jobs']}
    states[progress['job_id']] = progress
    write_json(path, states)
    counts = {status: sum(v['state'] == status for v in states.values())
              for status in ('pending', 'running', 'completed', 'failed')}
    write_json(root / 'QUEUE_STATE.json', dict(current_job=progress, counts=counts))


def verify_complete(directory, root, inputs):
    complete = read_json(directory / 'COMPLETE.json')
    if complete['identity_sha256'] != sha(root / 'run_identity.json') or complete['inputs'] != inputs:
        raise AssertionError('Completed job identity changed')
    for name, digest in complete['outputs'].items():
        if sha(directory / name) != digest:
            raise AssertionError('Completed artifact changed')


def run_job(cfg, root, scope, fold, system, seed, data, inputs, features=None):
    directory = root / scope / fold / f'{system}_seed{seed}'
    if (directory / 'COMPLETE.json').exists():
        verify_complete(directory, root, inputs)
        return
    directory.mkdir(parents=True, exist_ok=True)
    job_id = f'{scope}/{fold}/{system}_seed{seed}'
    started = time.monotonic()
    progress = dict(job_id=job_id, state='running', pid=os.getpid(), started_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    set_progress(root, progress)
    write_json(directory / 'INPUTS.json', inputs)
    labels = data['targets'].label.to_numpy(dtype=int)
    tr, ev = np.flatnonzero(data['train']), np.flatnonzero(~data['train'])
    counts = np.bincount(labels[tr], minlength=8)[1:8]
    if np.any(counts == 0):
        raise AssertionError('Missing Fit class')
    prior = counts / counts.sum()
    history = []
    if system == 'W-RF':
        model = RandomForestClassifier(**cfg['RF'], random_state=seed, n_jobs=cfg['threads'])
        model.fit(features[tr], labels[tr])
        prediction, fitted_prediction = model.predict(features[ev]), model.predict(features[tr])
        available = available_rows(data['tensors'])
        prediction[~available[ev]] = prior.argmax() + 1
        fitted_prediction[~available[tr]] = prior.argmax() + 1
        joblib.dump(dict(model=model, prior=prior, feature_order=cfg['rf_feature_order']), directory / 'checkpoint.joblib')
        detail = dict(features=features.shape[1], n_estimators=100,
                      unavailable_evaluation=int((~available[ev]).sum()))
    else:
        seed_all(seed)
        rng = np.random.default_rng(seed)
        weights = torch.tensor(np.sqrt(counts.sum() / (7 * counts)), dtype=torch.float32, device='cuda')
        y = torch.from_numpy(labels - 1).long().cuda()
        model = CurrentRepeatGRU(prior).cuda()
        if sum(p.numel() for p in model.parameters()) != cfg['expected_parameters']:
            raise AssertionError('Parameter count changed')
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(1, cfg['epochs'] + 1):
            model.train()
            numerator = denominator = 0.
            order = rng.permutation(tr)
            for at in range(0, len(order), cfg['batch_size']):
                idx = order[at:at + cfg['batch_size']]
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(tensor_batch(data['tensors'], idx, 'cuda')), y[idx])
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss')
                loss.backward()
                if epoch == 1 and at == 0 and not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise RuntimeError('Inactive/nonfinite trainable parameter')
                optimizer.step()
                weight_sum = float(weights[y[idx]].sum())
                numerator += float(loss.detach()) * weight_sum
                denominator += weight_sum
            entry = dict(epoch=epoch, train_loss=numerator / denominator)
            if scope == 'development':
                model.eval()
                entry['val_macro_f1'] = metrics(labels[ev], predict(model, data['tensors'], ev, cfg['batch_size']))['macro_f1']
            history.append(entry)
            write_json(directory / 'history.json', history)
            set_progress(root, dict(progress, epoch=epoch, last_train_loss=entry['train_loss']))
            print('HISTORY_CONTROL_EPOCH', job_id, epoch, entry['train_loss'], flush=True)
        model.eval()
        prediction = predict(model, data['tensors'], ev, cfg['batch_size'])
        fitted_prediction = predict(model, data['tensors'], tr, cfg['batch_size'])
        torch.save(dict(kind='CurrentRepeatGRU', state_dict=model.cpu().state_dict(), prior=prior,
                   CE_weights=weights.cpu(), normalizer=data['normalizer'], config=cfg,
                   input_identity=inputs, identity_sha256=sha(root / 'run_identity.json')), directory / 'checkpoint.pt')
        available = available_rows(data['tensors'], endpoint=True)
        detail = dict(parameters=cfg['expected_parameters'], expanded_steps=60,
                      unavailable_evaluation=int((~available[ev]).sum()),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated())
        del optimizer, weights, y
    table = data['targets'].loc[~data['train'], ['cow_id', 'timestamp', 'label']].reset_index(drop=True)
    table['prediction'] = np.asarray(prediction, dtype=int)
    table.to_csv(directory / 'predictions.csv', index=False)
    result = dict(system=system, seed=seed, scope=scope, fold=fold, selection=cfg['selection'],
                  metrics=metrics(labels[ev], prediction), Fit_metrics=metrics(labels[tr], fitted_prediction),
                  prior=prior.tolist(), fit_class_counts=counts.tolist(), seconds=time.monotonic() - started,
                  max_process_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, **detail)
    write_json(directory / 'result.json', result)
    write_json(directory / 'history.json', history)
    outputs = {p.name: sha(p) for p in directory.iterdir() if p.is_file() and p.name != 'COMPLETE.json'}
    write_json(directory / 'COMPLETE.json', dict(job_id=job_id, inputs=inputs, outputs=outputs,
                                               identity_sha256=sha(root / 'run_identity.json')))
    set_progress(root, dict(progress, state='completed', seconds=result['seconds']))
    print('HISTORY_CONTROL_JOB_COMPLETE', job_id, result['metrics']['macro_f1'], result['seconds'], flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def run_scope(cfg, root, scope):
    if scope != 'development':
        lock = read_json(root / 'METHOD_LOCK.json')
        if lock['identity_sha256'] != sha(root / 'run_identity.json'):
            raise AssertionError('Formal method lock differs')
    frames = raw_frames(cfg) if scope != 'development' else None
    scopes = [('development', 'fold_1')] if scope == 'development' else [(p, f) for p in cfg['protocols'] for f in cfg['folds']]
    for protocol, fold in scopes:
        data = developmental(cfg) if scope == 'development' else formal_data(cfg, protocol, fold, frames)
        inputs = input_identity(data)
        directory = root / protocol / fold
        directory.mkdir(parents=True, exist_ok=True)
        if 'parent_replay' in data:
            write_json(directory / 'BASE_REPLAY.json', data['parent_replay'])
        write_json(directory / 'DATA_RECEIPT.json', inputs)
        features = window_rf_features(data['tensors'])
        for system in cfg['systems']:
            for seed in cfg['seeds']:
                run_job(cfg, root, protocol, fold, system, seed, data, inputs, features)
        for path, digest in data['parent_files'].items():
            if sha(path) != digest:
                raise AssertionError('Parent changed during jobs')
        del data, features
        gc.collect()
    write_json(root / f'{scope.upper()}_COMPLETE.json', dict(status='COMPLETE',
               identity_sha256=sha(root / 'run_identity.json'),
               expected_jobs=6 if scope == 'development' else 60))


def lock_method(cfg, root):
    if read_json(root / 'SANITY.json')['status'] != 'PASS':
        raise AssertionError('Sanity not passed')
    if read_json(root / 'DEVELOPMENT_COMPLETE.json')['identity_sha256'] != sha(root / 'run_identity.json'):
        raise AssertionError('Development identity differs')
    jobs = []
    for system in cfg['systems']:
        for seed in cfg['seeds']:
            path = root / 'development/fold_1' / f'{system}_seed{seed}'
            complete = read_json(path / 'COMPLETE.json')
            verify_complete(path, root, complete['inputs'])
            result = read_json(path / 'result.json')
            if not np.isfinite(result['metrics']['macro_f1']) or result['selection'] != 'fixed_final_epoch':
                raise AssertionError('Invalid development result')
            history = read_json(path / 'history.json')
            if system == 'CurrentRepeat-GRU' and (len(history) != 20 or history[-1]['train_loss'] >= history[0]['train_loss']):
                raise AssertionError('Development training did not decrease loss')
            jobs.append(dict(job_id=complete['job_id'], complete_sha256=sha(path / 'COMPLETE.json')))
    lock = dict(status='LOCKED_AFTER_DEVELOPMENT', config=cfg, identity_sha256=sha(root / 'run_identity.json'),
                developmental_jobs=jobs, score_selection=False, historical_TEST_exposure=True,
                formal_jobs=60, created_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    path = root / 'METHOD_LOCK.json'
    if path.exists():
        previous = read_json(path)
        if {k: v for k, v in previous.items() if k != 'created_UTC'} != {k: v for k, v in lock.items() if k != 'created_UTC'}:
            raise AssertionError('Existing lock changed')
    else:
        write_json(path, lock)
    print('HISTORY_CONTROLS_METHOD_LOCKED', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['sanity', 'development', 'lock', 'formal'])
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
                run_scope(cfg, root, args.phase)
        except Exception:
            if (root / 'QUEUE_STATE.json').exists():
                current = read_json(root / 'QUEUE_STATE.json')['current_job']
                if current['state'] == 'running':
                    set_progress(root, dict(current, state='failed', traceback=traceback.format_exc()))
            write_json(root / f'FAILURE_{time.time_ns()}.json', dict(phase=args.phase, traceback=traceback.format_exc(),
                                                                  historical_results_preserved=True))
            raise


if __name__ == '__main__':
    main()
