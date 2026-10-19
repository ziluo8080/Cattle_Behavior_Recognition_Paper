"""Author feature pipeline on scoped development data, excluding unobservable joins.

For a full-day nearest join, a closest ankle record can lie outside the allowed
development scope. Timestamp-only boundary candidates detect these rows; they are
counted and excluded rather than silently substituting a more distant source.
"""
import numpy as np
import pandas as pd


def author_features(uwb, hd, ankle, labels, outside_ankle_timestamps=()):
    head = hd.copy()
    head['timestamp'] = np.floor(head.timestamp).astype(np.int64)
    head = head.groupby('timestamp', as_index=False).mean()
    head = head[head.timestamp.isin(uwb.timestamp)].astype(np.float64)
    akl = ankle.drop(columns=['datetime']).astype(np.float64)
    akl['ankle_source_timestamp'] = akl.timestamp
    head = pd.merge_asof(head.sort_values('timestamp'), akl.sort_values('timestamp'), on='timestamp', direction='nearest')
    matched_distance = np.abs(head.timestamp - head.ankle_source_timestamp).to_numpy()
    ambiguous = np.zeros(len(head), dtype=bool)
    for t in outside_ankle_timestamps:
        if t is not None:
            ambiguous |= np.abs(head.timestamp.to_numpy() - t) <= matched_distance
    head['outside_scope_nearest'] = ambiguous
    head = head.drop(columns=['ankle_source_timestamp'])
    gt = labels[labels.timestamp.isin(uwb.timestamp) & (labels.behavior != 0)]
    merged = uwb.merge(head, on='timestamp').merge(gt, on='timestamp')
    merged = merged.drop(columns=['datetime']).dropna().drop(columns=['accel_norm', 'relative_angle'])
    excluded = int(merged.outside_scope_nearest.sum())
    merged = merged[~merged.outside_scope_nearest].drop(columns=['outside_scope_nearest']).reset_index(drop=True)
    return merged, excluded
