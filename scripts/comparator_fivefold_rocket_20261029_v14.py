"""V14: fresh full-FIT author MultiRocket, then separately unlocked predictions.

Thirty final fits use frozen TS/OS alphas. No selection or Test scoring lives
here. Historical Test exposure means this is not a new blind evaluation.
Large feature matrices exist only in host RAM, never in reusable disk caches.
"""
import gc
import json
import os
from pathlib import Path
import random
import time

import numpy as np

from scripts.run_multirocket_dev_regularization_20261028 import (
    fit_ridge, predict_scores, rocket_functions,
)

FEATURES = 49890
MODEL = 'MultiRocket_Fusion'
ALPHA = {'TS': 20000, 'OS': 100000}
BATCH = 256
MAX_FIT_ROWS = 38960
INPUTS = ('dynamic', 'state', 'state_valid', 'context', 'current')


def _common():
    from scripts import comparator_fivefold_common_20261029_v14
    return comparator_fivefold_common_20261029_v14


def _options(cfg):
    """Validate the fixed recipe before sealing; do not require training SEAL."""
    common = _common()
    assert cfg['rocket_model'] == MODEL
    assert cfg['alpha_by_protocol'] == ALPHA, 'Frozen alphas cannot be selected again'
    assert cfg['ridge']['feature_block'] == 512
    assert cfg['ridge']['max_relative_residual'] == 1e-6
    options = dict(common.read(cfg['model_config'])['multirocket'])
    assert options['requested_features_per_branch'] == 25000
    assert options['features_per_kernel'] == 4
    assert options['max_dilations_per_kernel'] == 32
    assert options['transform_batch_size'] == BATCH
    assert options['internal_cross_validation'] is False
    return options


def _configure_cuda():
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(4)
    threadpool_limits(limits=4)
    assert torch.cuda.is_available(), 'Scientific worker requires CUDA'
    torch.cuda.set_per_process_memory_fraction(.65)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _begin(folder, protocol, fold, seed):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'STARTED.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(protocol=protocol, fold=fold, model=MODEL, seed=seed,
                       pid=os.getpid(), started_unix=time.time(), max_attempts=1,
                       scientific_retry_allowed=False), stream)


def _failed(folder, started, error):
    _common().save(folder / 'FAILED.json', dict(status='FAILED', error=repr(error),
        elapsed_seconds=time.monotonic() - started, retry_allowed=False))


def _deadline(cfg, started):
    assert time.monotonic() - started < cfg['budget']['rocket_worker_seconds'], 'Worker time budget exhausted'


def _data(raw, training):
    """Apply precisely the original dynamic masking and all-missing rule."""
    assert ('labels' in raw) == training
    data = {key: np.asarray(raw[key]) for key in INPUTS}
    n = len(data['dynamic'])
    assert n > 0 and all(len(data[k]) == n for k in INPUTS)
    dynamic = data.pop('dynamic').copy()
    valid = dynamic[:, :, 6] > .5
    dynamic[:, :, :6] = np.where(valid[:, :, None], dynamic[:, :, :6], 0.)
    dynamic[:, :, 7] = np.where(valid, np.clip(dynamic[:, :, 7], 0, .2), .2)
    data['native'] = dynamic
    data['missing'] = ~(valid.any(1) | data['state_valid'].any(1))
    if training:
        data['labels'] = np.asarray(raw['labels'], dtype=np.int64)
        assert np.array_equal(np.unique(data['labels']), np.arange(7))
    return data


def _take_raw(raw, rows):
    return {key: np.asarray(raw[key])[rows].copy() for key in INPUTS}


def _features(module, transform, data, parameters, options):
    parts = [transform(module, np.ascontiguousarray(data[key].transpose(0, 2, 1)), par, options)
             for key, par in zip(('native', 'state'), parameters)]
    parts += [data['context'], data['current']]
    values = np.concatenate(parts, axis=1)
    assert values.shape[1] == FEATURES and np.isfinite(values).all()
    return values


def _scores(values, ridge):
    # Identical batch boundaries make serialization/replay bitwise comparable.
    return np.concatenate([predict_scores(values[i:i + BATCH], ridge, block=512)
                           for i in range(0, len(values), BATCH)])


def _predictions(scores, missing, prior):
    pred = scores.argmax(1).astype(np.int64) + 1
    pred[missing] = int(np.argmax(prior)) + 1
    assert np.isin(pred, np.arange(1, 8)).all()
    return pred


def _verify_model(folder, protocol, fold, seed):
    common = _common()
    receipt = common.read(folder / 'COMPLETE.json')
    assert receipt['status'] == 'COMPLETE_REFIT'
    assert (receipt['protocol'], receipt['fold'], receipt['model'], receipt['seed']) == (protocol, fold, MODEL, seed)
    assert receipt['alpha'] == ALPHA[protocol] and receipt['features'] == FEATURES
    for name, digest in receipt['files'].items():
        assert common.sha(folder / name) == digest, 'Refit artifact hash mismatch: ' + name
    return receipt


def refit(cfg, p, f, seed, slot):
    """Fresh kernel/scaler/ridge training on complete FIT; never open Test."""
    import joblib
    from sklearn.preprocessing import StandardScaler
    common = _common(); common.check(cfg); options = _options(cfg)
    assert slot == 2 and p in ALPHA and f in cfg['folds'] and seed in cfg['seeds']
    folder = common.case_dir(cfg, 'refit', p, f, MODEL, seed)
    started = None; lock = None
    try:
        lock = common.gpu_lock(cfg, slot, kind='rocket'); _configure_cuda()
        _begin(folder, p, f, seed); started = time.monotonic()
        raw = common.raw_refit(cfg, p, f); data = _data(raw, training=True)
        # Preserve the original GPU/V7 recipe's float32 -> float64 weights.
        labels = data['labels']; weights = np.asarray(raw['class_weights'], np.float32).astype(np.float64)
        prior = np.asarray(raw['prior'], np.float64)
        assert weights.shape == prior.shape == (7,)
        assert np.isfinite(weights).all() and (weights > 0).all()
        counts = np.bincount(labels, minlength=7)
        np.testing.assert_allclose(prior, counts / counts.sum(), rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(weights, np.sqrt(len(labels) / (7 * counts)), rtol=1e-6)
        module, fit, transform, setseed = rocket_functions()
        np.random.seed(seed); random.seed(seed); setseed(seed)
        parameters = []; parts = []
        for key in ('native', 'state'):
            _deadline(cfg, started)
            common.save(folder / 'PROGRESS.json', dict(stage='fresh_author_fit', branch=key,
                fit_rows=len(labels), elapsed_seconds=time.monotonic() - started))
            branch = np.ascontiguousarray(data[key].transpose(0, 2, 1))
            par = fit(module, branch, options['requested_features_per_branch'], options)
            parameters.append(par)
            parts.append(transform(module, branch, par, options)); del branch
        parts += [data['context'], data['current']]
        values = np.concatenate(parts, axis=1); parts.clear()
        assert values.shape == (len(labels), FEATURES) and np.isfinite(values).all()
        scaler = StandardScaler(copy=False); values = scaler.fit_transform(values)
        def progress(**info):
            _deadline(cfg, started)
            common.save(folder / 'PROGRESS.json', dict(**info, elapsed_seconds=time.monotonic() - started))
        ridge = fit_ridge(values, labels, weights[labels], ALPHA[p], block=512,
                          device='cuda', progress=progress)
        assert ridge['relative_system_residual'] <= cfg['ridge']['max_relative_residual']
        checkpoint = dict(parameters=parameters, scaler=scaler, ridge=ridge,
            weights=weights, prior=prior, options=options, features=FEATURES,
            protocol=p, fold=f, seed=seed, alpha=ALPHA[p], model=MODEL,
            fit_rows=len(labels), training_scope='complete official FIT only',
            kernel_fit_fresh=True, Test_inputs_opened=False, Test_labels_parsed=False,
            new_blind_evaluation=False)
        joblib.dump(checkpoint, folder / 'model.joblib')
        restored = joblib.load(folder / 'model.joblib')
        n = min(BATCH, len(labels)); sample = _take_raw(raw, slice(0, n))
        replay = _features(module, transform, _data(sample, False), restored['parameters'], restored['options'])
        replay = restored['scaler'].transform(replay)
        np.testing.assert_array_equal(replay, values[:n])
        expected = _scores(values[:n], ridge); actual = _scores(replay, restored['ridge'])
        np.testing.assert_array_equal(actual, expected)
        np.savez_compressed(folder / 'fit_replay_inputs.npz', **sample)
        np.save(folder / 'fit_replay_scores.npy', actual)
        _deadline(cfg, started)
        receipt = dict(status='COMPLETE_REFIT', protocol=p, fold=f, seed=seed, model=MODEL,
            alpha=ALPHA[p], features=FEATURES, fit_rows=len(labels), kernel_fit_fresh=True,
            fitted_components=['author_original_and_difference_kernels_both_branches', 'standard_scaler', 'weighted_ridge'],
            relative_system_residual=ridge['relative_system_residual'],
            peak_cuda_bytes=ridge['peak_cuda_bytes'], seconds=time.monotonic() - started,
            raw_input_replay_rows=n, serialized_features_exact=True, serialized_scores_exact=True,
            feature_cache_reused=False, feature_cache_written=False, Test_inputs_opened=False,
            Test_labels_parsed=False, new_blind_evaluation=False,
            files={name: common.sha(folder / name) for name in ('model.joblib', 'fit_replay_inputs.npz', 'fit_replay_scores.npy')})
        common.save(folder / 'COMPLETE.json', receipt)
        return receipt
    except Exception as error:
        if started is not None: _failed(folder, started, error)
        raise
    finally:
        if lock is not None: lock.close()


def predict(cfg, p, f, seed, slot):
    """Unlocked, label-free Test inference from the exact persisted final fit."""
    import joblib
    common = _common(); common.check(cfg); _options(cfg)
    assert slot == 2 and p in ALPHA and f in cfg['folds'] and seed in cfg['seeds']
    common.check_training_lock(cfg)  # Must precede opening any Test input.
    source = common.case_dir(cfg, 'refit', p, f, MODEL, seed)
    _verify_model(source, p, f, seed)
    folder = common.case_dir(cfg, 'prediction', p, f, MODEL, seed)
    started = None; lock = None
    try:
        lock = common.gpu_lock(cfg, slot, kind='rocket'); _configure_cuda()
        _begin(folder, p, f, seed); started = time.monotonic()
        model = joblib.load(source / 'model.joblib')
        assert (model['protocol'], model['fold'], model['seed'], model['alpha'], model['features']) == (p, f, seed, ALPHA[p], FEATURES)
        raw = common.raw_test(cfg, p, f)
        assert 'labels' not in raw and 'targets' not in raw
        data = _data(raw, training=False)
        module, _, transform, _ = rocket_functions()
        values = model['scaler'].transform(_features(module, transform, data, model['parameters'], model['options']))
        scores = _scores(values, model['ridge'])
        pred = _predictions(scores, data['missing'], model['prior'])
        restored = joblib.load(source / 'model.joblib')
        np.testing.assert_array_equal(scores, _scores(values, restored['ridge']))
        n = min(BATCH, len(pred)); sample = _take_raw(raw, slice(0, n))
        replay = restored['scaler'].transform(_features(module, transform, _data(sample, False),
                                                       restored['parameters'], restored['options']))
        np.testing.assert_array_equal(replay, values[:n])
        np.testing.assert_array_equal(_scores(replay, restored['ridge']), scores[:n])
        np.testing.assert_array_equal(_predictions(_scores(replay, restored['ridge']), data['missing'][:n], restored['prior']), pred[:n])
        np.save(folder / 'test_predictions.npy', pred); np.save(folder / 'test_scores.npy', scores)
        _deadline(cfg, started)
        receipt = dict(status='COMPLETE_PREDICTION', protocol=p, fold=f, seed=seed,
            model=MODEL, alpha=ALPHA[p], features=FEATURES, test_rows=len(pred),
            raw_input_replay_rows=n, serialized_features_exact=True, serialized_scores_exact=True,
            prediction_labels='1..7', all_missing_fallback='saved FIT class prior argmax',
            all_missing_rows=int(data['missing'].sum()), Test_labels_parsed=False,
            new_blind_evaluation=False, historical_Test_exposure=True,
            refit_receipt_sha256=common.sha(source / 'COMPLETE.json'),
            seconds=time.monotonic() - started,
            files={name: common.sha(folder / name) for name in ('test_predictions.npy', 'test_scores.npy')})
        common.save(folder / 'COMPLETE.json', receipt)
        return receipt
    except Exception as error:
        if started is not None: _failed(folder, started, error)
        raise
    finally:
        if lock is not None: lock.close()


def witness(cfg, device, work_dir):
    """Independent primal solve, serialization and author seed/batch witnesses."""
    import joblib
    import torch
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    assert device in ('cpu', 'cuda')
    options = _options(cfg); torch.set_num_threads(4); threadpool_limits(limits=4)
    folder = Path(work_dir); folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(17); rows = []
    for n, d in ((70, 17), (70, 160)):
        values = rng.normal(size=(n, d)) + 2; y = np.arange(n) % 7
        weights = rng.uniform(.25, 4, n); query = rng.normal(size=(42, d))
        target = 2 * (y[:, None] == np.arange(7)).astype(np.float64) - 1
        xm = np.average(values, axis=0, weights=weights); ym = np.average(target, axis=0, weights=weights)
        xc = (values - xm) * np.sqrt(weights)[:, None]
        yc = (target - ym) * np.sqrt(weights)[:, None]
        for alpha in (.01, ALPHA['TS'], ALPHA['OS']):
            ridge = fit_ridge(values, y, weights, alpha, block=13, device=device)
            primal = np.linalg.solve(xc.T @ xc + alpha * np.eye(d), xc.T @ yc)
            expected = (query - xm) @ primal + ym; actual = predict_scores(query, ridge)
            error = float(np.max(np.abs(actual - expected)))
            assert error < 1e-6
            np.testing.assert_array_equal(actual.argmax(1), expected.argmax(1))
            path = folder / f'ridge_{device}_{n}_{d}_{alpha}.joblib'
            joblib.dump(ridge, path)
            np.testing.assert_array_equal(actual, predict_scores(query, joblib.load(path)))
            rows.append(dict(n=n, d=d, alpha=alpha, score_error=error,
                residual=ridge['relative_system_residual'], peak_cuda_bytes=ridge['peak_cuda_bytes']))
    module, fit, transform, setseed = rocket_functions()
    small = dict(options, transform_batch_size=5)
    author_rows = []
    for channels, length in ((8, 100), (26, 60)):
        x = np.ascontiguousarray(rng.normal(size=(21, channels, length)))
        np.random.seed(17); setseed(17); first = fit(module, x, 1344, small)
        np.random.seed(17); setseed(17); second = fit(module, x, 1344, small)
        for a, b in zip(first, second):
            for left, right in zip(a, b): np.testing.assert_array_equal(left, right)
        np.random.seed(23); setseed(23); different = fit(module, x, 1344, small)
        assert any(not np.array_equal(left, right) for a, b in zip(first, different) for left, right in zip(a, b))
        z = transform(module, x, first, small)
        whole = module.transform(x, np.ascontiguousarray(np.diff(x, axis=2)), *first, 4)
        np.testing.assert_array_equal(z, np.nan_to_num(whole, nan=0.))
        scaler = StandardScaler(copy=False); scaled = scaler.fit_transform(z.copy())
        path = folder / f'author_{channels}_{length}.joblib'
        joblib.dump(dict(parameters=first, scaler=scaler), path); restored = joblib.load(path)
        replay = restored['scaler'].transform(transform(module, x, restored['parameters'], small))
        np.testing.assert_array_equal(scaled, replay)
        author_rows.append(dict(channels=channels, length=length, features=z.shape[1],
                                seed_exact=True, other_seed_changes_kernels=True, batching_exact=True, serialized_exact=True))
    result = dict(status='PASS', device=device, cases=rows, author=author_rows,
        independent_numpy_primal=True, real_data_rows_read=0, scientific_fit=False,
        Test_labels_parsed=False, serialized_scores_exact=True)
    _common().save(folder / f'{device.upper()}_WITNESS.json', result)
    return result


def preflight(cfg):
    """Exclusive root phase: real FIT shapes and full-size synthetic GPU solve."""
    import torch
    from scripts.sensor_comparator_ridge_20261025 import full_shape_witness
    common = _common(); options = _options(cfg)
    lock = common.gpu_lock(cfg, 2, kind='rocket')
    try:
        _configure_cuda(); sizes = []; samples = {}
        for p in cfg['protocols']:
            for f in cfg['folds']:
                raw = common.raw_refit(cfg, p, f)
                y = np.asarray(raw['labels']); assert np.array_equal(np.unique(y), np.arange(7))
                sizes.append(dict(protocol=p, fold=f, fit_rows=len(y)))
                if p not in samples:
                    ids = np.concatenate([np.flatnonzero(y == c)[:32] for c in range(7)])
                    samples[p] = _data(_take_raw(raw, ids), False)
                del raw; gc.collect()
        nmax = max(row['fit_rows'] for row in sizes)
        assert nmax <= MAX_FIT_ROWS, 'Unexpected FIT size exceeds audited capacity'
        free, total = torch.cuda.mem_get_info()
        estimated_peak = 8 * nmax * nmax + 8 * nmax * 512 * 3 + 512 * 2 ** 20
        assert estimated_peak < min(free, int(.65 * total)), 'Insufficient reserved GPU memory'
        full = full_shape_witness(nmax, columns=512)
        module, fit, transform, setseed = rocket_functions(); timings = {}
        for p, data in samples.items():
            np.random.seed(17); random.seed(17); setseed(17)
            features = []; elapsed = 0.; kernel_seconds = 0.
            for key in ('native', 'state'):
                branch = np.ascontiguousarray(data[key].transpose(0, 2, 1))
                tick = time.monotonic(); par = fit(module, branch, options['requested_features_per_branch'], options)
                kernel_seconds += time.monotonic() - tick
                transform(module, branch, par, options)  # Warm compiler separately.
                tick = time.monotonic(); features.append(transform(module, branch, par, options))
                elapsed += time.monotonic() - tick
            features += [data['context'], data['current']]
            x = np.concatenate(features, axis=1); assert x.shape[1] == FEATURES
            dummy = dict(coef=np.ones((FEATURES, 7)), intercept=np.zeros(7))
            tick = time.monotonic(); _scores(x, dummy); prediction_seconds = time.monotonic() - tick
            timings[p] = dict(sample_rows=len(x), transform_seconds=elapsed,
                              kernel_seconds=kernel_seconds, score_seconds=prediction_seconds)
        records = []
        for row in sizes:
            timing = timings[row['protocol']]; n = row['fit_rows']
            feature_seconds = timing['transform_seconds'] * n / timing['sample_rows'] * 1.7 + 120
            ridge_seconds = (full['gram_seconds'] / 512 * FEATURES * (n / nmax) ** 2
                             + full['solve_seconds'] * (n / nmax) ** 3) * 1.7 + 120
            # Bound prediction by the largest FIT; never inspect Test for admission.
            prediction_seconds = ((timing['transform_seconds'] + timing['score_seconds'])
                                  * nmax / timing['sample_rows'] * 1.7 + 120)
            records.append(dict(**row, estimated_refit_seconds=feature_seconds + ridge_seconds + timing['kernel_seconds'],
                estimated_feature_seconds=feature_seconds, estimated_ridge_seconds=ridge_seconds,
                estimated_prediction_seconds=prediction_seconds, prediction_row_bound=nmax,
                estimated_host_feature_bytes=n * FEATURES * 8))
        budget = cfg['budget']['rocket_worker_seconds']
        passed = all(max(r['estimated_refit_seconds'], r['estimated_prediction_seconds']) <= budget for r in records)
        result = dict(status='PASS' if passed else 'OVER_BUDGET', records=records,
            full_shape=full, author_capacity=timings, nmax=nmax, features=FEATURES,
            peak_cuda_bytes=full['peak_cuda_bytes'], estimated_peak_gpu_bytes=estimated_peak,
            free_gpu_bytes=free, total_gpu_bytes=total, torch_memory_fraction=.65,
            worker_seconds=budget, real_FIT_shapes_verified=True, scientific_fit=False,
            Test_inputs_opened=False, Test_labels_parsed=False, feature_cache_written=False,
            prediction_rows_are_conservative_FIT_bound=True)
        common.save(Path(cfg['output']) / 'ROCKET_PREFLIGHT.json', result)
        assert passed, 'Frozen worker budget insufficient; do not launch'
        return result
    finally:
        lock.close()
