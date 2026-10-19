"""Apparent motion between distinct valid native UWB source observations."""
import numpy as np


def native_motion(x, valid, age, timestamps, starts, std):
    n, length, _ = x.shape
    grid = np.asarray(timestamps, dtype=float)[:, None] - (length - 1) + np.arange(length)
    source = grid - np.asarray(age[:, :, 0], dtype=float)
    full = np.asarray(valid).all(-1) & np.isfinite(source)
    if np.any(full & ((source < np.asarray(starts)[:, None]) | (source > grid + 1e-6))):
        raise AssertionError('Motion sources must remain inside past legal role')
    full &= source > np.asarray(timestamps)[:, None] - length
    index = np.broadcast_to(np.arange(length), (n, length))
    last = np.where(full, index, -1).max(1)
    r = np.arange(n)
    latest_source = source[r, np.maximum(last, 0)]
    previous = np.where(full & (source < latest_source[:, None] - 1e-6), index, -1).max(1)
    dt = latest_source - source[r, np.maximum(previous, 0)]
    pair = (last >= 0) & (previous >= 0) & (dt > 0)
    velocity = np.zeros((n, 3), dtype=float)
    delta = (np.asarray(x[r, np.maximum(last, 0)], dtype=float) -
             np.asarray(x[r, np.maximum(previous, 0)], dtype=float)) * np.asarray(std)[None]
    velocity[pair] = delta[pair] / dt[pair, None]
    native_count = np.asarray([len(np.unique(source[i, full[i]])) for i in range(n)], dtype=float)
    endpoint_age = np.where(last >= 0, np.asarray(timestamps) - latest_source, 60.)
    features = np.column_stack((velocity, np.linalg.norm(velocity, axis=1), np.linalg.norm(velocity[:, :2], axis=1),
        np.where(pair, dt, 0.) / 60., np.minimum(endpoint_age, 60.) / 60., np.minimum(native_count, 5.) / 5., pair.astype(float)))
    if not np.isfinite(features).all():
        raise AssertionError('Motion features must be finite')
    return features, pair, dt


def fit_motion_scaler(features, pair):
    if not pair.any():
        raise AssertionError('TRAIN requires actual native pairs')
    mean, std = features[pair, :5].mean(0), features[pair, :5].std(0)
    return dict(mean=mean.tolist(), std=np.maximum(std, 1e-6).tolist(), scope='TRAIN_valid_native_pairs_only_no_labels')


def scale_motion(features, pair, scaler):
    out = features.copy()
    out[:, :5] = (out[:, :5] - np.asarray(scaler['mean'])) / np.asarray(scaler['std'])
    out[~pair, :5] = 0.
    return out.astype(np.float32)
