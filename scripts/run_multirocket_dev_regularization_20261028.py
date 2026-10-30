"""Bounded FIT-only MultiRocket fitting and TRAIN-internal ridge selection.

All validation metrics describe existing development data, not an independent test.
Ridge scores remain uncalibrated scores: no softmax or cross-entropy is computed.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from scripts.resume_comparator_queue_20261028 import resource_conflict

CONFIG = Path('configs/sensor_multirocket_dev_regularization_20261028_v1.json')
MODULE = 'scripts.run_multirocket_dev_regularization_20261028'
ALPHAS = [0.01, 1, 100, 10000, 1000000]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def config():
    cfg = read(CONFIG)
    assert cfg['alphas'] == ALPHAS and cfg['features'] == 49890
    assert cfg['protocols'] == ['TS', 'OS'] and cfg['inners'] == [0, 1, 2] and cfg['seed'] == 17
    assert cfg['worker_seconds'] == 1800 and cfg['max_attempts'] == 1
    assert all(cfg[k] is False for k in ['Test_allowed', 'outer_VAL_allowed', 'formal_refit_allowed', 'new_blind_evaluation'])
    return cfg


def implementation(cfg):
    files = [CONFIG, Path(__file__), Path(cfg['model_config']),Path(cfg['ledger']),Path('scripts/resume_comparator_queue_20261028.py'),Path('tests/test_multirocket_dev_regularization_20261028.py'),
             Path('scripts/sensor_comparator_ridge_20261025.py'),
             Path('experiments/sensor_comparator_pilot_20261022_v2/cpu.py')]
    files += list(Path('experiments/sensor_comparator_pilot_20261022_v2/vendor/multirocket').rglob('*.py'))
    return {p.as_posix() if not p.is_absolute() else p.relative_to(Path.cwd()).as_posix(): sha(p) for p in files}


def cloud_lock():
    from scripts import run_convtran_context_20261022 as core
    # Lock before CUDA initialization, matching the shared single-GPU contract.
    assert Path.cwd() == core.ROOT
    lock = core.acquire_gpu()
    core.configure()
    return lock


def predict_scores(x, model, block=512):
    result = np.broadcast_to(model['intercept'], (len(x), 7)).copy()
    for start in range(0, x.shape[1], block):
        result += np.asarray(x[:, start:start + block], np.float64) @ model['coef'][start:start + block]
    assert np.isfinite(result).all()
    return result


def fit_ridge(x, y, weights, alpha, block=512, device='cuda', progress=None):
    """Same original weighted, centered objective; positive alpha is the only change."""
    import torch
    from scripts.sensor_comparator_ridge_20261025 import inplace_cholesky
    assert np.isfinite(alpha) and alpha > 0
    assert np.array_equal(np.unique(y), np.arange(7))
    started = time.monotonic(); progress = progress or (lambda **kw: None)
    n, d = x.shape; weights = np.asarray(weights, np.float64)
    assert weights.shape == (n,) and np.isfinite(weights).all() and (weights > 0).all()
    root = np.sqrt(weights)
    target = 2 * (y[:, None] == np.arange(7)[None]).astype(np.float64) - 1
    target_mean = np.average(target, axis=0, weights=weights)
    rhs = torch.as_tensor(root[:, None] * (target - target_mean), device=device, dtype=torch.float64)
    gram = torch.zeros((n, n), device=device, dtype=torch.float64).T
    means = np.empty(d, np.float64)
    if device == 'cuda': torch.cuda.reset_peak_memory_stats()
    for start in range(0, d, block):
        values = np.asarray(x[:, start:start + block], np.float64)
        assert np.isfinite(values).all()
        mean = np.average(values, axis=0, weights=weights); means[start:start + values.shape[1]] = mean
        z = torch.as_tensor(np.ascontiguousarray((values - mean) * root[:, None]), device=device, dtype=torch.float64)
        gram.addmm_(z, z.T); del z
        progress(stage='gram', columns=min(start + block, d), total_columns=d)
    gram.diagonal().add_(alpha)
    dual = inplace_cholesky(gram, rhs); del gram
    coefficients = np.empty((d, 7), np.float64); reconstructed = alpha * dual.clone()
    for start in range(0, d, block):
        values = np.asarray(x[:, start:start + block], np.float64)
        z = torch.as_tensor(np.ascontiguousarray((values - means[start:start + values.shape[1]]) * root[:, None]), device=device, dtype=torch.float64)
        coef = z.T @ dual; coefficients[start:start + values.shape[1]] = coef.cpu().numpy()
        reconstructed.addmm_(z, coef); del z, coef
    relative = float(torch.linalg.vector_norm(reconstructed - rhs) / torch.linalg.vector_norm(rhs))
    assert np.isfinite(relative) and relative <= 1e-6, relative
    intercept = target_mean - means @ coefficients
    assert np.isfinite(coefficients).all() and np.isfinite(intercept).all()
    return dict(coef=coefficients, intercept=intercept, classes=np.arange(7), alpha=float(alpha),
                relative_system_residual=relative, fit_seconds=time.monotonic() - started,
                peak_cuda_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else 0)


def choose_alpha(rows):
    assert sorted(r['alpha'] for r in rows) == ALPHAS, 'All fixed candidates are mandatory'
    assert all(np.isfinite(r['dev']['macro_f1_percent']) for r in rows)
    return max(rows, key=lambda r: (r['dev']['macro_f1_percent'], r['alpha']))['alpha']


def metrics(y, scores, class_weight, missing, prior):
    target = 2 * (y[:, None] == np.arange(7)[None]).astype(np.float64) - 1
    pred = scores.argmax(1); pred[missing] = int(np.argmax(prior))
    cm = np.zeros((7, 7), np.int64); np.add.at(cm, (y, pred), 1)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * cm.diagonal(), denom, out=np.zeros(7), where=denom > 0)
    sse = float(np.sum(class_weight[y, None] * (scores - target) ** 2))
    return dict(macro_f1_percent=float(100 * f1.mean()), class_f1_percent=(100 * f1).tolist(),
                confusion_matrix=cm.tolist(), weighted_square_error_sum=sse,
                weighted_square_error_per_row=sse / float(class_weight[y].sum()),
                squared_loss_scope='raw ridge scores before original all-missing class fallback'), pred


def case_dir(cfg, protocol, inner):
    assert protocol in cfg['protocols'] and inner in cfg['inners']
    return Path(cfg['output']) / protocol / f'inner{inner}'


def sources(cfg, protocol, inner):
    folder = Path(cfg['data_parent']) / 'data' / protocol / f'inner{inner}'
    receipt = read(folder / 'RECEIPT.json'); assert (receipt['protocol'], receipt['inner']) == (protocol, inner)
    files = {str(folder / 'RECEIPT.json'): sha(folder / 'RECEIPT.json')}
    for name in ['FIT.npz', 'DEV_INPUTS.npz', 'DEV_TARGETS.jsonl']:
        file = folder / name; digest = sha(file); assert digest == receipt['files'][name], file
        files[str(file)] = digest
    return folder, receipt, files


def load_data(cfg, protocol, inner):
    folder, receipt, files = sources(cfg, protocol, inner); groups = []
    for name in ['FIT.npz', 'DEV_INPUTS.npz']:
        with np.load(folder / name, allow_pickle=False) as z:
            assert ('labels' in z.files) == (name == 'FIT.npz')
            data = {k: z[k].copy() for k in ['dynamic', 'state', 'state_valid', 'context', 'current']}
            if name == 'FIT.npz': data['y'] = z['labels'].astype(np.int64)
        dyn = data.pop('dynamic'); valid = dyn[:, :, 6] > .5
        dyn[:, :, :6] = np.where(valid[:, :, None], dyn[:, :, :6], 0.)
        dyn[:, :, 7] = np.where(valid, np.clip(dyn[:, :, 7], 0, .2), .2)
        data['native'] = dyn; data['missing'] = ~(valid.any(1) | data['state_valid'].any(1)); groups.append(data)
    targets = [json.loads(line) for line in (folder / 'DEV_TARGETS.jsonl').read_text().splitlines()]
    assert len({r['original_id'] for r in targets}) == len(targets)
    groups[1]['y'] = np.asarray([r['label'] - 1 for r in targets], np.int64)
    for data, key in zip(groups, ['fit_rows', 'held_rows']):
        assert len(data['y']) == receipt[key] and np.isin(data['y'], np.arange(7)).all()
    counts = np.bincount(groups[0]['y'], minlength=7); assert (counts > 0).all()
    # Preserve the original GPU comparator's float32 -> float64 weights exactly.
    weights = np.sqrt(counts.sum() / (7 * counts)).astype(np.float32).astype(np.float64)
    return groups, weights, counts / counts.sum(), files


def rocket_functions():
    import numba
    from experiments.sensor_comparator_pilot_20261022_v2.cpu import mr_functions, mr_fit, mr_transform, numba_seed
    numba.set_num_threads(4)
    return mr_functions(), mr_fit, mr_transform, numba_seed


def witness(cfg):
    import torch
    torch.set_num_threads(2)
    rng = np.random.default_rng(17); rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for n, d in [(70, 17), (70, 160)]:
            x = rng.normal(size=(n, d)) + 2; y = np.arange(n) % 7
            weights = rng.uniform(.25, 4, n); query = rng.normal(size=(42, d))
            target = 2 * (y[:, None] == np.arange(7)[None]).astype(float) - 1
            xm = np.average(x, axis=0, weights=weights); ym = np.average(target, axis=0, weights=weights)
            xc = (x - xm) * np.sqrt(weights)[:, None]; yc = (target - ym) * np.sqrt(weights)[:, None]
            for alpha in ALPHAS:
                model = fit_ridge(x, y, weights, alpha, block=13, device='cpu')
                primal = np.linalg.solve(xc.T @ xc + alpha * np.eye(d), xc.T @ yc)
                expected = (query - xm) @ primal + ym; actual = predict_scores(query, model)
                error = float(np.max(np.abs(actual - expected))); assert error < 1e-6
                np.testing.assert_array_equal(actual.argmax(1), expected.argmax(1))
                path = Path(tmp) / 'model.npz'; np.savez(path, **model)
                with np.load(path, allow_pickle=False) as archive:
                    restored = dict(archive)
                np.testing.assert_array_equal(actual, predict_scores(query, restored))
                rows.append(dict(n=n, d=d, alpha=alpha, score_error=error, residual=model['relative_system_residual']))
    save(Path(cfg['output']) / 'CPU_WITNESS.json', dict(status='PASS', real_data_rows_read=0,
         independent_numpy_primal=True, serialized_scores_exact=True, cases=rows, code=implementation(cfg)))


def preflight(cfg):
    """Measure full-size Gram/solve and small FIT-only transforms before fitting."""
    import torch
    from scripts.sensor_comparator_ridge_20261025 import full_shape_witness
    lock = cloud_lock(); out = Path(cfg['output']); out.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(out).free >= cfg['minimum_free_disk_GiB'] * 2 ** 30
    records = []; sizes = []
    for protocol in cfg['protocols']:
        for inner in cfg['inners']:
            _, receipt, _ = sources(cfg, protocol, inner)
            sizes.append(dict(protocol=protocol, inner=inner, n=receipt['fit_rows'], dev=receipt['held_rows']))
    # Retain all six feature caches. Budget the larger float64 representation,
    # even if the original transformer emits float32 on this runtime.
    required_disk = sum(r['n'] + r['dev'] for r in sizes) * cfg['features'] * 8 + 2 * 2 ** 30
    free_disk = shutil.disk_usage(out).free
    assert free_disk >= required_disk, 'Insufficient disk to retain all fixed development feature caches'
    n = max(r['n'] for r in sizes); free, total = torch.cuda.mem_get_info()
    estimated_peak = 8 * n * n + 8 * n * 512 * 3 + 2 ** 30
    assert free > estimated_peak, 'Insufficient free GPU memory for streamed dual ridge'
    full = full_shape_witness(n, columns=512)
    module, fit, transform, setseed = rocket_functions(); options = read(cfg['model_config'])['multirocket']
    for protocol in cfg['protocols']:
        groups, _, _, _ = load_data(cfg, protocol, 0); data = groups[0]
        ids = np.concatenate([np.flatnonzero(data['y'] == c)[:16] for c in range(7)])
        seconds = 0.; np.random.seed(17); setseed(17)
        for key in ['native', 'state']:
            x = np.ascontiguousarray(data[key].transpose(0, 2, 1))
            pars = fit(module, x[ids], options['requested_features_per_branch'], options)
            tick = time.monotonic(); transform(module, x[:256], pars, options)
            seconds += time.monotonic() - tick
        for row in [r for r in sizes if r['protocol'] == protocol]:
            feature_seconds = seconds * (row['n'] + row['dev']) / 256 * 1.7 + 120
            ridge_seconds = (full['gram_seconds'] / 512 * 49890 * (row['n'] / n) ** 2 + full['solve_seconds'] * (row['n'] / n) ** 3) * 1.7 + 120
            records.append(dict(**row, estimated_prepare_seconds=feature_seconds, estimated_ridge_seconds=ridge_seconds))
        del groups, data; gc.collect()
    passed = all(max(r['estimated_prepare_seconds'], r['estimated_ridge_seconds']) <= cfg['worker_seconds'] for r in records)
    save(out / 'PREFLIGHT.json', dict(status='PASS' if passed else 'OVER_BUDGET', free_gpu_bytes=free,
         estimated_peak_gpu_bytes=estimated_peak, free_disk_bytes=free_disk, required_disk_bytes=required_disk,
         full_shape=full, cases=records, code=implementation(cfg),
         data_scope='existing outer TRAIN internal FIT/DEV only', Test_rows=0, outer_VAL_rows=0))
    lock.close(); assert passed, 'Fixed worker budget insufficient; do not launch'


def begin(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'STARTED.json').open('x', encoding='utf-8') as handle:
        json.dump(dict(pid=os.getpid(), unix=time.time(), max_attempts=1), handle)


def prepare(cfg, protocol, inner):
    import joblib
    from sklearn.preprocessing import StandardScaler
    lock = cloud_lock(); folder = case_dir(cfg, protocol, inner) / 'features'; begin(folder); start = time.monotonic()
    groups, weights, prior, files = load_data(cfg, protocol, inner)
    module, fit, transform, setseed = rocket_functions(); options = read(cfg['model_config'])['multirocket']
    np.random.seed(17); random.seed(17); setseed(17); parts = [[], []]; parameters = []
    for key in ['native', 'state']:
        par = fit(module, np.ascontiguousarray(groups[0][key].transpose(0, 2, 1)), options['requested_features_per_branch'], options)
        parameters.append(par)
        for i, data in enumerate(groups):
            save(folder / 'PROGRESS.json', dict(stage=key, role=['fit', 'dev'][i], elapsed_seconds=time.monotonic() - start))
            parts[i].append(transform(module, np.ascontiguousarray(data[key].transpose(0, 2, 1)), par, options))
    scaler = StandardScaler(copy=False)
    for i, (data, role) in enumerate(zip(groups, ['fit', 'dev'])):
        parts[i] += [data['context'], data['current']]; x = np.concatenate(parts[i], -1); parts[i].clear()
        assert x.shape[1] == cfg['features']
        x = scaler.fit_transform(x) if i == 0 else scaler.transform(x)
        np.save(folder / (role + '.npy'), x); del x
        np.savez(folder / (role + '_targets.npz'), y=data['y'], missing=data['missing'])
    joblib.dump(dict(parameters=parameters, scaler=scaler, options=options, weights=weights, prior=prior), folder / 'transform.joblib')
    # Replay serialized kernel, scaler and feature transform on raw DEV inputs.
    restored = joblib.load(folder / 'transform.joblib')
    replay = [transform(module, np.ascontiguousarray(groups[1][key][:128].transpose(0, 2, 1)), par, restored['options'])
              for key, par in zip(['native', 'state'], restored['parameters'])]
    replay += [groups[1]['context'][:128], groups[1]['current'][:128]]
    replay = restored['scaler'].transform(np.concatenate(replay, -1))
    np.testing.assert_array_equal(replay, np.load(folder / 'dev.npy', mmap_mode='r')[:128])
    names = ['fit.npy', 'dev.npy', 'fit_targets.npz', 'dev_targets.npz', 'transform.joblib']
    save(folder / 'COMPLETE.json', dict(status='COMPLETE', source_files=files, code=implementation(cfg),
         serialized_kernel_scaler_replay_rows=128, seconds=time.monotonic() - start,
         files={name: sha(folder / name) for name in names}))
    lock.close()


def worker(cfg, protocol, inner, alpha):
    import joblib
    if inner == 0:
        assert alpha in ALPHAS
    else:
        selected = read(Path(cfg['output']) / protocol / 'SELECTED.json')
        assert alpha == selected['alpha'] and selected['code'] == implementation(cfg)
    lock = cloud_lock(); case = case_dir(cfg, protocol, inner)
    folder = case / ('alpha_' + format(alpha, '.12g')); begin(folder); started = time.monotonic()
    feat = case / 'features'; receipt = read(feat / 'COMPLETE.json'); assert receipt['code'] == implementation(cfg)
    for name, digest in receipt['files'].items(): assert sha(feat / name) == digest
    transform = joblib.load(feat / 'transform.joblib'); weights, prior = transform['weights'], transform['prior']
    x = np.load(feat / 'fit.npy', mmap_mode='r'); targets = dict(np.load(feat / 'fit_targets.npz', allow_pickle=False))
    def progress(**kw): save(folder / 'PROGRESS.json', dict(**kw, elapsed_seconds=time.monotonic() - started))
    model = fit_ridge(x, targets['y'], weights[targets['y']], alpha, cfg['feature_block'], progress=progress)
    joblib.dump(model, folder / 'model.joblib'); restored = joblib.load(folder / 'model.joblib')
    result = dict(alpha=float(alpha), protocol=protocol, inner=inner, seed=17,
                  relative_system_residual=model['relative_system_residual'], optimization_stable=True,
                  no_overfit_claim=False, generalization_scope=cfg['confirmation'])
    for role in ['fit', 'dev']:
        values = np.load(feat / (role + '.npy'), mmap_mode='r'); target = dict(np.load(feat / (role + '_targets.npz'), allow_pickle=False))
        scores = predict_scores(values, model); np.testing.assert_array_equal(scores, predict_scores(values, restored))
        result[role], pred = metrics(target['y'], scores, weights, target['missing'], prior)
        np.save(folder / (role + '_predictions.npy'), pred); np.save(folder / (role + '_scores.npy'), scores)
    result['regularization_penalty'] = float(alpha * np.sum(model['coef'] ** 2))
    result['fit_objective'] = result['fit']['weighted_square_error_sum'] + result['regularization_penalty']
    result['fit_minus_DEV_F1_pp'] = result['fit']['macro_f1_percent'] - result['dev']['macro_f1_percent']
    result.update(status='COMPLETE', seconds=time.monotonic() - started, serialized_scores_exact=True,
                  Test_rows=0, outer_VAL_rows=0, code=implementation(cfg), features_receipt_sha256=sha(feat / 'COMPLETE.json'))
    result['artifacts_sha256']={n:sha(folder/n) for n in ['model.joblib','fit_predictions.npy','dev_predictions.npy','fit_scores.npy','dev_scores.npy']}
    save(folder / 'COMPLETE.json', result); lock.close()


def checked_run(cfg, args, folder):
    started = time.monotonic()
    try:
        logroot=Path(cfg['output'])/'logs';logroot.mkdir(exist_ok=True)
        name='_'.join(folder.relative_to(cfg['output']).parts)
        for admission in range(4):
            while int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()[0])>=500:
                assert time.monotonic()-started<172800,'Resource wait budget exhausted'
                save(Path(cfg['output'])/'PROGRESS.json',dict(status='WAITING_RESOURCE',case=str(folder)));time.sleep(15)
            path=logroot/f'{name}_admission{admission}.log'
            save(Path(cfg['output'])/'PROGRESS.json',dict(status='RUNNING',case=str(folder),log=str(path)))
            with path.open('x') as log:
                result=subprocess.run([sys.executable, '-s', '-u', '-m', MODULE] + args,stdout=log,stderr=subprocess.STDOUT,timeout=cfg['worker_seconds'])
            if result.returncode==0:return
            if admission<3 and resource_conflict(result.returncode,path,folder):continue
            raise subprocess.CalledProcessError(result.returncode,result.args)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
        save(folder / 'FAILED.json', dict(status='TIMEOUT' if isinstance(error, subprocess.TimeoutExpired) else 'FAILED',
             elapsed_seconds=time.monotonic() - started, error=str(error), attempts=1, retry_allowed=False))
        raise


def run(cfg):
    out = Path(cfg['output'])
    for name in ['CPU_WITNESS.json', 'PREFLIGHT.json']:
        record = read(out / name); assert record['status'] == 'PASS' and record['code'] == implementation(cfg)
    begin(out / 'run')
    for protocol in cfg['protocols']:
        case = case_dir(cfg, protocol, 0)
        checked_run(cfg, ['--mode', 'prepare', '--protocol', protocol, '--inner', '0'], case / 'features')
        rows = []
        for alpha in ALPHAS:
            folder = case / ('alpha_' + format(alpha, '.12g'))
            checked_run(cfg, ['--mode', 'worker', '--protocol', protocol, '--inner', '0', '--alpha', str(alpha)], folder)
            rows.append(read(folder / 'COMPLETE.json'))
        selected = choose_alpha(rows)
        save(out / protocol / 'SELECTED.json', dict(alpha=selected, rule=cfg['selection'], all_candidates=rows,
             selected_before_confirmation=True, code=implementation(cfg)))
        for inner in [1, 2]:
            case = case_dir(cfg, protocol, inner)
            checked_run(cfg, ['--mode', 'prepare', '--protocol', protocol, '--inner', str(inner)], case / 'features')
            checked_run(cfg, ['--mode', 'worker', '--protocol', protocol, '--inner', str(inner), '--alpha', str(selected)],
                        case / ('alpha_' + format(selected, '.12g')))
    save(out / 'COMPLETE.json', dict(status='COMPLETE_STOP_FOR_REVIEW', ridge_fits=14, feature_fits=6,
         Test_rows=0, outer_VAL_rows=0, selection=cfg['selection'], scope=cfg['confirmation'],
         no_overfit_claim=False, code=implementation(cfg)))
    save(out/'PROGRESS.json',dict(status='COMPLETE_STOP_FOR_REVIEW',ridge_fits=14,feature_fits=6))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', required=True, choices=['witness', 'preflight', 'prepare', 'worker', 'run'])
    parser.add_argument('--protocol', choices=['TS', 'OS']); parser.add_argument('--inner', type=int, choices=[0, 1, 2]); parser.add_argument('--alpha', type=float)
    args = parser.parse_args(); cfg = config()
    if args.mode == 'witness': witness(cfg)
    elif args.mode == 'preflight': preflight(cfg)
    elif args.mode == 'run': run(cfg)
    elif args.mode == 'prepare': prepare(cfg, args.protocol, args.inner)
    else: worker(cfg, args.protocol, args.inner, args.alpha)


if __name__ == '__main__':
    main()
