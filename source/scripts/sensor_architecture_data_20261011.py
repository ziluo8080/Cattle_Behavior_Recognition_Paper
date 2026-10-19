"""Own-fold official TRAIN/VAL selection and immutable parent Test targets."""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

from q3_sensor.protocol import build_windows, fit_normalizer, merge_segments, sha
from scripts.run_sensor_formal_ts import official_roles
from scripts.sensor_os_protocol import assert_window_sources, cow_roles, day_role
from scripts.run_sensor_history_controls_20261011 import read_json, write_json, raw_frames

FIELDS = ('x', 'valid', 'fresh', 'age')


def inner_ts_roles(official, fold):
    zone = pytz.timezone('America/Chicago')
    def unix(value):
        return int(zone.localize(datetime.strptime(value, '%Y-%m-%d %H:%M:%S')).timestamp())
    roles = []
    for group in ('group_1', 'group_2'):
        for key, (a, b) in official[group].items():
            split = next(role for role in ('train', 'val', 'test') if key in fold[role])
            roles.append(dict(start=unix(a), end=unix(b), split=split, segment=f'{group}_{key}'))
    return merge_segments(roles)


def parent_hashes(cfg):
    paths = [Path(cfg['source']) / 'source_receipt.json', Path('configs/official_ts.json'),
             Path('configs/official_os.json'), Path('configs/sensor_os_20261007.json')]
    for protocol, directory in cfg['parent_roots'].items():
        root = Path(directory)
        identity = read_json(root / 'run_identity.json')
        if identity['source_receipt_sha256'] != sha(Path(cfg['source']) / 'source_receipt.json'):
            raise AssertionError('Parent raw receipt changed')
        for name, digest in identity['sources'].items():
            if name.startswith('q3_sensor/') and sha(name) != digest:
                raise AssertionError(f'Frozen original sensor code changed {name}')
        paths.append(root / 'run_identity.json')
        for fold in cfg['folds']:
            parent = root / fold
            complete = read_json(parent / 'COMPLETE.json')
            if complete['target_sha256'] != sha(parent / 'targets.jsonl'):
                raise AssertionError('Original target identity mismatch')
            if protocol == 'TS':
                if complete['method_lock_sha256'] != identity['method_lock_sha256'] or not complete['source_targets_same']:
                    raise AssertionError('Legacy TS receipt mismatch')
            elif complete['identity_sha256'] != sha(root / 'run_identity.json'):
                raise AssertionError('OS receipt mismatch')
            paths.extend(parent / name for name in ('targets.jsonl', 'normalizer.json', 'predictions.csv', 'COMPLETE.json'))
            for seed in cfg['formal_seeds']:
                path = parent / f'NN_seed{seed}.pt'
                if protocol == 'OS' and sha(path) != complete['checkpoint_sha256'][path.name]:
                    raise AssertionError('Original OS weight changed')
                paths.append(path)
    return {str(p): sha(p) for p in paths}


def frames_from_source(cfg):
    return raw_frames(dict(formal_source=cfg['source']))


def make_cache(cfg, root, phase, protocol, fold, frames):
    directory = root / 'data' / phase / protocol / fold
    receipt_path = directory / 'DATA_RECEIPT.json'
    parent = Path(cfg['parent_roots'][protocol]) / fold
    expected_parents = {str(p): sha(p) for p in (parent / 'targets.jsonl', parent / 'normalizer.json')}
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt['parents'] != expected_parents:
            raise AssertionError('Cached parent inputs changed')
        for name, digest in receipt['outputs'].items():
            if sha(directory / name) != digest:
                raise AssertionError('Cached data changed')
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    original = pd.read_json(parent / 'targets.jsonl', lines=True, convert_dates=False)
    if original.duplicated(['cow_id', 'timestamp']).any():
        raise AssertionError('Repeated original targets')
    targets = original[original.split.eq('train')].copy() if phase == 'selection' else original.copy()
    official = read_json(f'configs/official_{protocol.lower()}.json')
    fold_config = official['folds'][fold]
    if protocol == 'TS':
        roles = inner_ts_roles(official, fold_config) if phase == 'selection' else official_roles(official, fold_config)
        records = []
        for row in targets.to_dict('records'):
            role = next(r for r in roles if r['start'] <= row['timestamp'] < r['end'])
            if phase == 'selection' and role['split'] == 'test':
                raise AssertionError('Test target entered selection')
            records.append(dict(row, **role))
        targets = pd.DataFrame(records)
        normalizer = fit_normalizer(frames, roles)
    else:
        groups, fit_ids, test_ids = cow_roles(fold_config)
        role = day_role(read_json('configs/sensor_os_20261007.json'))
        if phase == 'selection':
            targets['split'] = ['train' if int(cow) in groups['train'] else 'val' for cow in targets.cow_id]
            if set(targets.cow_id) != set(fit_ids):
                raise AssertionError('Selection includes wrong OS cows')
            normalizer = fit_normalizer({c: frames[c] for c in groups['train']}, [dict(role, split='train')])
        else:
            if set(targets.loc[targets.split.eq('train'), 'cow_id']) != set(fit_ids) or set(targets.loc[targets.split.eq('test'), 'cow_id']) != set(test_ids):
                raise AssertionError('Wrong final OS cow membership')
            normalizer = fit_normalizer({c: frames[c] for c in fit_ids}, [dict(role, split='train')])
    if phase == 'formal':
        if normalizer != read_json(parent / 'normalizer.json'):
            raise AssertionError('Formal normalization no longer matches parent')
        pd.testing.assert_frame_equal(targets[['cow_id', 'timestamp', 'label']], original[['cow_id', 'timestamp', 'label']])
    else:
        if set(targets.split) != {'train', 'val'}:
            raise AssertionError('Selection needs TRAIN and VAL only')
        pd.testing.assert_frame_equal(targets[['cow_id', 'timestamp', 'label']].reset_index(drop=True),
                                      original.loc[original.split.eq('train'), ['cow_id', 'timestamp', 'label']].reset_index(drop=True))
    for cow, group in targets.groupby('cow_id'):
        gt = pd.read_csv(Path(cfg['source']) / f'behavior_labels/individual/C{cow:02d}_0725.csv').set_index('timestamp').behavior
        np.testing.assert_array_equal(gt.loc[group.timestamp].to_numpy(dtype=int), group.label.to_numpy())
    windows = build_windows(targets.to_dict('records'), frames, normalizer)
    assert_window_sources(targets, windows)
    arrays = {f'{m}_{k}': data[k] for m, data in windows.items() for k in FIELDS}
    train = targets.split.eq('train').to_numpy()
    counts = np.bincount(targets.label.to_numpy()[train], minlength=8)[1:8]
    if np.any(counts == 0):
        raise AssertionError('All seven Fit classes required')
    targets.to_json(directory / 'targets.jsonl', orient='records', lines=True)
    write_json(directory / 'normalizer.json', normalizer)
    np.savez(directory / 'windows.npz', **arrays)
    available = np.stack([data['valid'][:, -1].any(1) for data in windows.values()], 1)
    receipt = dict(phase=phase, protocol=protocol, fold=fold, parents=expected_parents,
        train=int(train.sum()), evaluation=int((~train).sum()), class_counts=counts.tolist(),
        evaluation_class_counts=np.bincount(targets.label.to_numpy()[~train], minlength=8)[1:8].tolist(),
        no_source_after_grid=True, source_within_own_role=True, normalizer_scope='TRAIN_only' if phase == 'selection' else 'TRAIN_plus_VAL',
        all_endpoint_modalities_unavailable=int((~available.any(1)).sum()),
        outputs={name: sha(directory / name) for name in ('targets.jsonl', 'normalizer.json', 'windows.npz')})
    write_json(receipt_path, receipt)
    return directory
