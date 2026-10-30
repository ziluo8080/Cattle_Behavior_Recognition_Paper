"""OS identity helpers; reuse the locked sensor representation without mutation."""
import importlib.util
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

from q3_sensor.protocol import fit_normalizer


RF_FEATURES = ['coord_x_cm', 'coord_y_cm', 'coord_z_cm', 'roll', 'pitch', 'yaw', 'lying']


def cow_roles(fold):
    groups = {key: [int(x) for x in fold[key]] for key in ('train', 'val', 'test')}
    if [len(groups[k]) for k in groups] != [6, 2, 2]:
        raise ValueError('Official OS requires 6/2/2 cows')
    flattened = sum(groups.values(), [])
    if len(set(flattened)) != 10 or set(flattened) != set(range(1, 11)):
        raise ValueError('OS cow overlap or incomplete universe')
    return groups, groups['train'] + groups['val'], groups['test']


def day_role(config):
    zone = pytz.timezone(config['timezone'])
    day = datetime.strptime(config['day'], '%Y-%m-%d')
    start = int(zone.localize(day).timestamp())
    end = int(zone.localize(day + timedelta(days=1)).timestamp())
    return dict(start=start, end=end, segment='OS_original_0725_day', split='train')


def os_normalizer(frames, fit_ids, role):
    if len(set(fit_ids)) != 8 or not set(fit_ids).issubset(frames):
        raise ValueError('Exactly eight distinct fit cows required')
    return fit_normalizer({cow: frames[cow] for cow in fit_ids}, [dict(role, split='train')])


def load_author_daily(source):
    spec = importlib.util.spec_from_file_location('sensor_author_os_loader',
                                                 'reports/sensor_baselines_20261007/upstream/data_loader.py')
    author = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = author
    spec.loader.exec_module(author)
    cached, frames = {}, {}
    for cow in range(1, 11):
        cached[cow] = author.uwb_hd_aa_pre_loader(str(source), [cow])
        cached[cow]['cow_id'] = cow  # Identity only; excluded from every model input.
        tag, name = f'T{cow:02d}', f'C{cow:02d}'
        frames[cow] = {
            'uwb': pd.read_csv(source / f'main_data/uwb/{tag}/{tag}_0725.csv'),
            'hd': pd.read_csv(source / f'sub_data/head_direction/{tag}/{tag}_0725.csv'),
            'akl': pd.read_csv(source / f'main_data/ankle/{name}/{name}_0725.csv')}

    def cached_preload(sensor_data_dir, id_list, date='0725', label=True):
        if Path(sensor_data_dir) != source or date != '0725' or not label or len(set(id_list)) != len(id_list):
            raise ValueError('Unexpected OS author loader arguments')
        return pd.concat([cached[int(cow)] for cow in id_list], ignore_index=True)

    author.uwb_hd_aa_pre_loader = cached_preload
    return author, frames


def fold_data(author, source, fold, role):
    groups, fit_ids, test_ids = cow_roles(fold)
    train_frame, val_frame, test_frame = author.data_loader_s1(
        'uwb_hd_aa_pre_loader', str(source), fold, val=False, timestamp=True)
    if len(val_frame) or set(train_frame.cow_id) != set(fit_ids) or set(test_frame.cow_id) != set(test_ids):
        raise AssertionError('Author OS membership mismatch')
    rows = []
    for split, frame in (('train', train_frame), ('test', test_frame)):
        for r in frame.itertuples(index=False):
            if not role['start'] <= r.timestamp < role['end'] or r.timestamp != int(r.timestamp):
                raise AssertionError('Author target outside original day or nonintegral')
            if int(r.behavior) not in range(1, 8):
                raise AssertionError('Unknown target in seven-class comparison')
            original_role = next(k for k, ids in groups.items() if int(r.cow_id) in ids)
            rows.append(dict(role, split=split, cow_id=int(r.cow_id), timestamp=int(r.timestamp),
                             label=int(r.behavior), official_cow_role=original_role))
    targets = pd.DataFrame(rows)
    if targets.duplicated(['cow_id', 'timestamp']).any():
        raise AssertionError('Duplicate OS targets')
    return train_frame, test_frame, targets, fit_ids, test_ids


def assert_window_sources(targets, windows):
    for data in windows.values():
        source = data['source']
        valid = np.isfinite(source)
        times = targets.timestamp.to_numpy()
        grid = times[:, None] + np.arange(-59, 1)
        if not np.array_equal(valid, data['valid'].astype(bool)) or not np.isfinite(data['x']).all():
            raise AssertionError('Window validity or values corrupted')
        if not np.all(np.where(valid, (source <= grid[..., None]) &
                               (source > times[:, None, None] - 60) &
                               (source >= targets.start.to_numpy()[:, None, None]), True)):
            raise AssertionError('Illegal OS source timestamp')
