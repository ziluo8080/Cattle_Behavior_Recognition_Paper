"""Fixed targets and past-only windows on the official temporal split."""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

MODALITIES = ('uwb', 'hd', 'akl')
FEATURES = {'uwb': ('coord_x_cm', 'coord_y_cm', 'coord_z_cm'),
            'hd': ('roll', 'pitch', 'yaw'), 'akl': ('lying',)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def development_segments(config, fold='fold_1'):
    zone = pytz.timezone('America/Chicago')
    def unix(value):
        return int(zone.localize(datetime.strptime(value, '%Y-%m-%d %H:%M:%S')).timestamp())
    segments = []
    for split in ('train', 'val'):
        for group in ('group_1', 'group_2'):
            for key in config['folds'][fold][split]:
                a, b = map(unix, config[group][key])
                segments.append(dict(split=split, segment=f'{group}_{key}', start=a, end=b))
    return merge_segments(segments)


def merge_segments(segments):
    """Adjacent official blocks in the same split form one legal continuous run."""
    merged = []
    for s in sorted(segments, key=lambda x: x['start']):
        if merged and merged[-1]['split'] == s['split'] and merged[-1]['end'] == s['start']:
            merged[-1]['end'] = s['end']
            merged[-1]['segment'] += '+' + s['segment']
        else:
            merged.append(dict(s))
    return merged


def targets(labels, segments, cow_id):
    """Labels determine targets before any observation availability is examined."""
    rows = []
    for s in segments:
        selected = labels[(labels.timestamp >= s['start'] + 60) &
                          (labels.timestamp < s['end']) &
                          (labels.timestamp % 15 == 0) & labels.behavior.isin(range(1, 8))]
        for row in selected.itertuples():
            rows.append(dict(cow_id=cow_id, timestamp=int(row.timestamp), label=int(row.behavior), **s))
    return rows


def observations(frame, modality):
    """Convert HD to trailing circular one-second means, retaining source times."""
    fields = FEATURES[modality]
    values = frame.loc[:, list(fields)].to_numpy(dtype=np.float64)
    times = frame.timestamp.to_numpy(dtype=np.float64)
    if np.any(np.diff(times) < 0):
        raise ValueError('Non-monotonic source timestamps')
    if modality != 'hd':
        if modality == 'uwb':
            values = values / 100.0
        return times, times, values
    angle = np.deg2rad(values)
    encoded = np.stack((np.sin(angle), np.cos(angle)), axis=-1).reshape(len(values), 6)
    valid = np.isfinite(values).all(axis=1)
    hd = pd.DataFrame(encoded[valid])
    hd['source'] = times[valid]
    hd['endpoint'] = np.ceil(times[valid]).astype(np.int64)
    grouped = hd.groupby('endpoint', sort=True)
    average = grouped[list(range(6))].mean()
    magnitude = np.sqrt(np.square(average.to_numpy().reshape(-1, 3, 2)).sum(-1))
    usable = (magnitude > 1e-8).all(axis=1)
    # Normalize circular means to unit sin/cos; never average wrapped degrees.
    normalized = average.to_numpy().reshape(-1, 3, 2) / np.maximum(magnitude[..., None], 1e-8)
    return average.index.to_numpy()[usable], grouped.source.max().to_numpy()[usable], normalized.reshape(-1, 6)[usable]


def fit_normalizer(frames, segments):
    result = {}
    for modality in MODALITIES:
        parts = []
        for cow in sorted(frames):
            df = frames[cow][modality]
            keep = np.zeros(len(df), dtype=bool)
            for s in segments:
                if s['split'] == 'train':
                    keep |= (df.timestamp >= s['start']) & (df.timestamp < s['end'])
            raw = df.loc[keep, list(FEATURES[modality])].to_numpy(dtype=np.float64)
            raw[~np.isfinite(raw)] = np.nan
            if modality == 'uwb':
                raw = raw / 100.0
            if modality == 'hd':
                rad = np.deg2rad(raw)
                raw = np.stack((np.sin(rad), np.cos(rad)), -1).reshape(-1, 6)
            parts.append(raw)
        joined = np.concatenate(parts)
        counts = np.isfinite(joined).sum(axis=0)
        if np.any(counts == 0):
            raise ValueError(f'No real TRAIN observations for {modality}')
        mean, std = np.nanmean(joined, axis=0), np.nanstd(joined, axis=0)
        if modality == 'akl':
            mean, std = np.zeros(1), np.ones(1)
        result[modality] = dict(mean=mean.tolist(), std=np.maximum(std, 1e-6).tolist(),
                                real_train_valid_counts=counts.tolist())
    return result


def build_windows(rows, frames, normalizer):
    """Per-field past hold; every source must be strictly inside (t-60,t]."""
    result = {}
    for modality in MODALITIES:
        dims = 6 if modality == 'hd' else len(FEATURES[modality])
        n = len(rows)
        x = np.zeros((n, 60, dims), np.float32)
        v = np.zeros_like(x, dtype=np.uint8)
        fresh = np.zeros((n, 60, 1), np.uint8)
        source = np.full((n, 60, dims), np.nan, np.float64)
        age = np.full((n, 60, 1), np.nan, np.float32)
        native_count = np.zeros(n, np.int32)
        native_valid_count = np.zeros(n, np.int32)
        mean = np.asarray(normalizer[modality]['mean'])
        std = np.asarray(normalizer[modality]['std'])
        for cow in sorted(frames):
            for segment in sorted({r['segment'] for r in rows if r['cow_id'] == cow}):
                indices = [i for i, r in enumerate(rows) if r['cow_id'] == cow and r['segment'] == segment]
                if not indices:
                    continue
                r0 = rows[indices[0]]
                df = frames[cow][modality]
                df = df[(df.timestamp >= r0['start']) & (df.timestamp < r0['end'])]
                endpoints, original, values = observations(df, modality)
                times = np.asarray([rows[i]['timestamp'] for i in indices])
                grid = times[:, None] + np.arange(-59, 1)[None, :]
                raw_time = df.timestamp.to_numpy()
                raw_valid = np.isfinite(df.loc[:, list(FEATURES[modality])].to_numpy()).all(axis=1)
                usable_time = raw_time[raw_valid]
                native_count[indices] = np.searchsorted(raw_time, times, side='right') - np.searchsorted(raw_time, times - 60, side='right')
                native_valid_count[indices] = np.searchsorted(usable_time, times, side='right') - np.searchsorted(usable_time, times - 60, side='right')
                for d in range(dims):
                    good = np.isfinite(values[:, d])
                    et, ot, vals = endpoints[good], original[good], values[good, d]
                    if not len(et):
                        continue
                    pos = np.searchsorted(et, grid, side='right') - 1
                    safe = np.maximum(pos, 0)
                    valid = (pos >= 0) & (ot[safe] > times[:, None] - 60)
                    selected = ((vals[safe] - mean[d]) / std[d]).astype(np.float32)
                    x[indices, :, d] = np.where(valid, selected, 0)
                    v[indices, :, d] = valid
                    source[indices, :, d] = np.where(valid, ot[safe], np.nan)
                    fresh[indices, :, 0] |= (valid & (et[safe] > grid - 1) & (et[safe] <= grid)).astype(np.uint8)
                src = source[indices]
                # Oldest available component age avoids calling partial old values fresh.
                oldest = np.min(np.where(np.isfinite(src), src, np.inf), axis=2)
                age[indices, :, 0] = np.where(np.isfinite(oldest), grid - oldest, np.nan)
        result[modality] = dict(x=x, valid=v, fresh=fresh, source=source, age=age,
                               native_count=native_count, native_valid_count=native_valid_count)
    return result
