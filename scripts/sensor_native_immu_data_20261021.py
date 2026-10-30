"""Pure array operations for role-bounded published 10 Hz IMMU inputs."""
import numpy as np

CHANNELS = ('accel_x_mps2', 'accel_y_mps2', 'accel_z_mps2',
            'mag_x_uT', 'mag_y_uT', 'mag_z_uT')


def canonical_source(frame):
    """Use exact decisecond indices, after checking the published grid."""
    times = frame.timestamp.to_numpy(dtype=np.float64)
    if not np.isfinite(times).all():
        raise ValueError('Nonfinite source clock')
    ticks = np.rint(times * 10).astype(np.int64)
    if not np.array_equal(times, ticks / 10):
        raise ValueError('Source is not on the verified 0.1-second grid')
    values = frame.loc[:, CHANNELS].to_numpy(dtype=np.float64)
    if (np.diff(ticks) < 0).any():
        raise ValueError('Unsorted source')
    same = np.flatnonzero(np.diff(ticks) == 0)
    for i in same:
        if not np.array_equal(values[i], values[i + 1], equal_nan=True):
            raise ValueError('Conflicting duplicate timestamp')
    unique = np.r_[True, np.diff(ticks) != 0] if len(ticks) else np.empty(0, bool)
    return ticks[unique], values[unique]


def fit_scaler(sources, train_roles):
    """Each complete (cow, source tick) row contributes once, outside windows."""
    total = np.zeros(6, np.float64)
    squares = np.zeros(6, np.float64)
    count = 0
    per_cow = {}
    for cow, segments in sorted(train_roles.items()):
        ticks, values = sources[cow]
        keep = np.zeros(len(ticks), bool)
        for role in segments:
            if role['split'] != 'train':
                raise ValueError('Scaler received non-TRAIN role')
            keep |= (ticks >= role['start'] * 10) & (ticks < role['end'] * 10)
        keep &= np.isfinite(values).all(1)
        selected = values[keep]
        total += selected.sum(0)
        squares += np.square(selected).sum(0)
        count += len(selected)
        per_cow[str(cow)] = len(selected)
    if count == 0:
        raise ValueError('No complete TRAIN source rows')
    mean = total / count
    variance = np.maximum(squares / count - mean ** 2, 0.)
    return dict(mean=mean.tolist(), std=np.maximum(np.sqrt(variance), 1e-6).tolist(),
                unique_complete_source_rows=count, unique_rows_per_cow=per_cow,
                scope='TRAIN source rows once; no target/window multiplicity')


def windows(rows, sources, scaler):
    """Return N x 100 x 8 values and exact provenance in decisecond ticks."""
    output = np.zeros((len(rows), 100, 8), np.float32)
    output[:, :, 7] = .2
    provenance = np.full((len(rows), 100), -1, np.int64)
    mean, std = np.asarray(scaler['mean']), np.asarray(scaler['std'])
    groups = {}
    for index, row in enumerate(rows):
        if row['split'] not in ('train', 'val'):
            raise ValueError('Only TRAIN/VAL targets accepted')
        key = (row['cow_id'], row['start'], row['end'], row['split'])
        groups.setdefault(key, []).append(index)
    for (cow, start, end, split), ids in groups.items():
        ticks, values = sources[cow]
        keep = (ticks >= start * 10) & (ticks < end * 10) & np.isfinite(values).all(1)
        ticks, values = ticks[keep], values[keep]
        target = np.asarray([rows[i]['timestamp'] for i in ids], np.int64) * 10
        if not ((target >= start * 10) & (target < end * 10)).all():
            raise ValueError('Target outside role')
        query = target[:, None] + np.arange(-99, 1)[None, :]
        if not len(ticks):
            continue
        pos = np.searchsorted(ticks, query, side='right') - 1
        picked = ticks[np.maximum(pos, 0)]
        age = query - picked
        # The first grid point must not borrow an older-than-window source.
        valid = (pos >= 0) & (age >= 0) & (age <= 2) & (picked > target[:, None] - 100)
        physical = (values[np.maximum(pos, 0)] - mean) / std
        output[ids, :, :6] = np.where(valid[:, :, None], physical, 0)
        output[ids, :, 6] = valid
        output[ids, :, 7] = np.where(valid, age / 10, .2)
        provenance[ids] = np.where(valid, picked, -1)
        assert np.all(picked[valid] <= query[valid])
        assert np.all(picked[valid] >= start * 10)
    if not np.isfinite(output).all():
        raise ValueError('Nonfinite model input')
    return output, provenance


def low_frequency_windows(rows, sources, scaler):
    """Causal means of completed (k-1,k] bins, held until the next bin end.

    A bin's entire interval must be inside the target window and role. An empty
    bin is invalid. Holding stops before the next bin boundary, so gaps are not
    bridged indefinitely. age is time since the latest contributing source,
    clipped at 1 second. This intervention changes availability/age too; it is
    a fixed compression control, not a claim of identical missingness.
    """
    output = np.zeros((len(rows), 100, 8), np.float32)
    output[:, :, 7] = 1.
    first_source = np.full((len(rows), 100), -1, np.int64)
    last_source = np.full_like(first_source, -1)
    mean, std = np.asarray(scaler['mean']), np.asarray(scaler['std'])
    groups = {}
    for i, row in enumerate(rows):
        if row['split'] not in ('train', 'val'):
            raise ValueError('Only TRAIN/VAL targets accepted')
        groups.setdefault((row['cow_id'], row['start'], row['end']), []).append(i)
    for (cow, start, end), ids in groups.items():
        ticks, values = sources[cow]
        good = (ticks >= start * 10) & (ticks < end * 10) & np.isfinite(values).all(1)
        ticks, values = ticks[good], values[good]
        if not len(ticks):
            continue
        bin_end = ((ticks + 9) // 10) * 10
        bins, begin, counts = np.unique(bin_end, return_index=True, return_counts=True)
        averaged = np.add.reduceat(values, begin, axis=0) / counts[:, None]
        lo, hi = ticks[begin], ticks[begin + counts - 1]
        target = np.asarray([rows[i]['timestamp'] for i in ids], np.int64) * 10
        query = target[:, None] + np.arange(-99, 1)[None, :]
        pos = np.searchsorted(bins, query, side='right') - 1
        safe = np.maximum(pos, 0)
        selected_bin = bins[safe]
        valid = ((pos >= 0) & (query - selected_bin < 10) &
                 (selected_bin - 10 >= start * 10) &
                 (selected_bin - 10 >= target[:, None] - 100))
        physical = (averaged[safe] - mean) / std
        output[ids, :, :6] = np.where(valid[:, :, None], physical, 0)
        output[ids, :, 6] = valid
        output[ids, :, 7] = np.where(valid, np.minimum((query - hi[safe]) / 10, 1.), 1.)
        first_source[ids] = np.where(valid, lo[safe], -1)
        last_source[ids] = np.where(valid, hi[safe], -1)
        assert (last_source[ids][valid] <= query[valid]).all()
        assert (first_source[ids][valid] > np.broadcast_to(target[:, None] - 100, valid.shape)[valid]).all()
    assert np.isfinite(output).all()
    return output, first_source, last_source


def masked_readout(features, valid):
    """NumPy reference for the two model readouts, including all-missing cases."""
    outputs = []
    for start in (0, 90):
        mask = valid[:, start:, None].astype(bool)
        total = np.where(mask, features[:, start:], 0).sum(1)
        count = mask.sum(1)
        outputs.append(np.divide(total, count, out=np.zeros_like(total), where=count > 0))
    return np.concatenate(outputs, axis=1)
