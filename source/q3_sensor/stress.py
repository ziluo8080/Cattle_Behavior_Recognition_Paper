"""Past-only perturbations with source time distinct from arrival time.

This module leaves the locked clean protocol and networks unchanged.
"""
import numpy as np

from .protocol import FEATURES, MODALITIES, observations


def empty_modality(n, dimensions):
    return dict(x=np.zeros((n, 60, dimensions), np.float32),
                valid=np.zeros((n, 60, dimensions), np.uint8), fresh=np.zeros((n, 60, 1), np.uint8),
                source=np.full((n, 60, dimensions), np.nan), age=np.full((n, 60, 1), np.nan, np.float32),
                native_count=np.zeros(n, np.int32), native_valid_count=np.zeros(n, np.int32))


def stressed_modality(rows, frames, normalizer, modality, condition):
    dims = 6 if modality == 'hd' else len(FEATURES[modality])
    output = empty_modality(len(rows), dims)
    if condition['kind'] == 'drop':
        return output
    delay = condition.get('seconds', 0) if condition['kind'] == 'delay' else 0
    mean, std = [np.asarray(normalizer[modality][key]) for key in ('mean', 'std')]
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault((row['cow_id'], row['segment']), []).append(i)
    for (cow, _), indices in groups.items():
        first = rows[indices[0]]
        frame = frames[cow][modality]
        frame = frame[(frame.timestamp >= first['start']) & (frame.timestamp < first['end'])]
        if condition['kind'] == 'rate':
            frame = frame.iloc[::condition['factor']]
        elif condition['kind'] == 'burst':
            frame = frame[((frame.timestamp - first['start']) % 300) >= 30]
        endpoints, original, values = observations(frame, modality)
        arrival = endpoints + delay
        times = np.asarray([rows[i]['timestamp'] for i in indices])
        grid = times[:, None] + np.arange(-59, 1)
        raw = frame.timestamp.to_numpy()
        finite_rows = np.isfinite(frame.loc[:, list(FEATURES[modality])].to_numpy()).all(1)
        usable = raw[finite_rows]
        output['native_count'][indices] = np.maximum(0, np.searchsorted(raw + delay, times, side='right') -
                                                       np.searchsorted(raw, times - 60, side='right'))
        output['native_valid_count'][indices] = np.maximum(0, np.searchsorted(usable + delay, times, side='right') -
                                                             np.searchsorted(usable, times - 60, side='right'))
        for d in range(dims):
            finite = np.isfinite(values[:, d])
            et, ot, vals = arrival[finite], original[finite], values[finite, d]
            if not len(et):
                continue
            position = np.searchsorted(et, grid, side='right') - 1
            safe = np.maximum(position, 0)
            valid = (position >= 0) & (ot[safe] > times[:, None] - 60)
            output['x'][indices, :, d] = np.where(valid, (vals[safe] - mean[d]) / std[d], 0)
            output['valid'][indices, :, d] = valid
            output['source'][indices, :, d] = np.where(valid, ot[safe], np.nan)
            output['fresh'][indices, :, 0] |= (valid & (et[safe] > grid - 1) & (et[safe] <= grid)).astype(np.uint8)
        source = output['source'][indices]
        oldest = np.min(np.where(np.isfinite(source), source, np.inf), axis=2)
        output['age'][indices, :, 0] = np.where(np.isfinite(oldest), grid - oldest, np.nan)
    return output


def rf_features(windows, endpoint):
    parts = []
    for modality in MODALITIES:
        data = windows[modality]
        age = np.log1p(np.nan_to_num(data['age'], nan=60.0).clip(0, 60)) / np.float32(np.log(61))
        parts.append(np.concatenate((data['x'], data['valid'], data['fresh'], age), axis=-1))
    tokens = np.concatenate(parts, axis=-1).astype(np.float32)
    return tokens[:, -1] if endpoint else tokens.reshape(len(tokens), -1)
