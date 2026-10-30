"""Frozen descriptors and official MiniRocket wrapper; no labels in features."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from numba import njit, set_num_threads
from q3_sensor.endpoint_models import tokens
from q3_sensor.protocol import MODALITIES, sha
from scripts.sensor_history_control_models_20261011 import repeat_current
from scripts.run_sensor_history_controls_20261011 import write_json

VENDOR = Path('third_party/minirocket_20261012')
sys.path.insert(0, str(VENDOR.resolve()))
import minirocket_multivariate as rocket


@njit(cache=True)
def seed_numba(seed):
    np.random.seed(seed)


def rocket_fit(values, seed, features=10000):
    np.random.seed(seed)
    seed_numba(seed)
    return rocket.fit(np.ascontiguousarray(values, dtype=np.float32), num_features=features)


def numpy_batch(arrays, chosen=None):
    return {m: {k: torch.from_numpy(np.array(v if chosen is None else v[chosen], copy=True))
                for k, v in fields.items()} for m, fields in arrays.items()}


def sequence_features(arrays):
    batch = numpy_batch(arrays)
    result = torch.cat([tokens(batch[m]) for m in MODALITIES], -1).permute(0, 2, 1).numpy()
    result = np.ascontiguousarray(result, dtype=np.float32)
    assert result.shape[1:] == (26, 60) and np.isfinite(result).all()
    return result


def descriptors(arrays, normalizer, current=False):
    """Held-grid summaries; native UWB deduplicates identical observed sources.

    Cached UWB age is oldest-component age; a source group contributes motion
    only when all 3 fields are valid and the position agrees within that group.
    Therefore partial asynchronous records never invent a full-position pair.
    No cow ID, labels, target absolute timestamp, future or raw extra points.
    """
    batch = numpy_batch(arrays)
    if current:
        batch = repeat_current(batch)
    parts, names = [], []
    for m in MODALITIES:
        data = {k: v.numpy() for k, v in batch[m].items()}
        x, valid = data['x'], data['valid'].astype(bool)
        count = valid.sum(1)
        safe = np.maximum(count, 1)
        mean = (x * valid).sum(1) / safe
        std = np.sqrt(((x - mean[:, None]) ** 2 * valid).sum(1) / safe)
        low = np.where(valid, x, np.inf).min(1)
        high = np.where(valid, x, -np.inf).max(1)
        low[count == 0] = high[count == 0] = 0
        values = [('endpoint', x[:, -1]), ('endpoint_valid', valid[:, -1]),
                  ('held_mean', mean), ('held_std', std), ('held_min', low),
                  ('held_max', high), ('held_range', high - low),
                  ('held_count', count), ('no_support', count == 0)]
        for label, value in values:
            parts.append(value)
            names.extend(f'{m}.{label}.{d}' for d in range(x.shape[-1]))
        meta = np.column_stack([data['fresh'][:, :, 0].sum(1),
            valid.any(2).mean(1), data['fresh'][:, -1, 0],
            np.nan_to_num(data['age'][:, -1, 0], nan=60., posinf=60., neginf=0.).clip(0, 60)])
        parts.append(meta)
        names.extend(f'{m}.{k}' for k in ('held_fresh_count', 'coverage', 'endpoint_fresh', 'endpoint_age'))
    uwb = {k: v.numpy() for k, v in batch['uwb'].items()}
    scale = np.asarray(normalizer['uwb']['std'])
    center = np.asarray(normalizer['uwb']['mean'])
    motion = np.zeros((len(uwb['x']), 8), dtype=np.float32)
    for i in range(len(motion)):
        usable = uwb['valid'][i].all(1) & np.isfinite(uwb['age'][i, :, 0])
        if current:
            if usable[-1]:
                motion[i, 0] = 1
            motion[i, 7] = 1
            continue
        grid = np.arange(-59, 1, dtype=np.float64)
        relative_source = grid[usable] - uwb['age'][i, usable, 0].astype(np.float64)
        position = uwb['x'][i, usable].astype(np.float64) * scale + center
        source_times, points = [], []
        for source in np.unique(relative_source):
            group = position[np.isclose(relative_source, source, rtol=0, atol=1e-7)]
            if len(group) and np.max(np.abs(group - group[0])) <= 1e-6:
                source_times.append(source)
                points.append(group[0])
        motion[i, 0] = len(points)
        motion[i, 7] = len(points) < 2
        if len(points) > 1:
            times, pos = np.asarray(source_times), np.asarray(points)
            dt = np.diff(times)
            assert (dt > 0).all()
            distance = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1)
            speed = distance / dt
            motion[i, 1:7] = [times[-1] - times[0], np.linalg.norm(pos[-1, :2] - pos[0, :2]),
                             distance.sum(), speed.mean(), speed.max(), np.std(pos[:, 2])]
    parts.append(motion)
    names.extend('uwb.native_' + key for key in ('count', 'span_seconds', 'net_xy_m', 'path_xy_m',
                 'mean_speed_m_s', 'max_speed_m_s', 'z_std_m', 'insufficient_pair'))
    output = np.concatenate(parts, 1).astype(np.float32)
    assert output.shape[1] == len(names) == 110 and np.isfinite(output).all()
    return output, names


def witness(path):
    import numba, llvmlite, sklearn
    set_num_threads(4)
    rng = np.random.default_rng(17)
    values = rng.standard_normal((21, 26, 60)).astype(np.float32)
    first, second = rocket_fit(values, 17), rocket_fit(values, 17)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    transformed = rocket.transform(values, first)
    assert transformed.shape == (21, 9996) and np.isfinite(transformed).all()
    batched = np.concatenate([rocket.transform(values[:8], first), rocket.transform(values[8:], first)])
    np.testing.assert_array_equal(transformed, batched)
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(transformed)
    classifier = RidgeClassifier(alpha=1., solver='lsqr', tol=1e-6, max_iter=10000).fit(scaler.transform(transformed), np.arange(21) % 7 + 1)
    assert classifier.coef_.shape == (7, 9996)
    matrix = torch.randn(8, 8, device='cuda', generator=torch.Generator(device='cuda').manual_seed(17))
    assert torch.isfinite(matrix @ matrix).all()
    info = dict(status='PASS', synthetic_only=True, scientific_F1=False,
        versions=dict(torch=torch.__version__, numpy=np.__version__, numba=numba.__version__,
                      llvmlite=llvmlite.__version__, sklearn=sklearn.__version__),
        actual_features=9996, native_seed_reproducible=True, batch_transform_exact=True,
        ridge_fit_finite=True, CUDA_kernel_finite=True,
        environment_canonical_sha256=hashlib.sha256(json.dumps(json.loads(Path('configs/sensor_classic_env_20261012.json').read_text()),sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        vendor_sha256=sha(VENDOR/'minirocket_multivariate.py'))
    write_json(Path(path), info)
    print('CLASSIC_ENV_KERNEL_WITNESS_PASS', json.dumps(info), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['witness'])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    witness(args.output)
