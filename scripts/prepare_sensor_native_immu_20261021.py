"""Build native/causally compressed IMMU and context for frozen selection rosters."""
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch

from q3_sensor.endpoint_models import tokens
from q3_sensor.protocol import merge_segments
from scripts.sensor_classic_features_20261012 import descriptors
from scripts.sensor_native_immu_data_20261021 import canonical_source, fit_scaler, windows, low_frequency_windows

ROOT = Path('/q3/mmcows-sensor-behavior-20261007')
OUT = ROOT / 'reports/sensor_native_immu_preflight_20261021_v1'
BASE = ROOT / 'reports/sensor_architecture_comparison_20261011/data/selection'
INVENTORY = ROOT / 'reports/sensor_native_immu_inventory_20261021_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temporary.replace(path)


def sha(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        while chunk := file.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def unix(value):
    return int(datetime.strptime(value, '%Y-%m-%d %H:%M:%S').replace(tzinfo=ZoneInfo('America/Chicago')).timestamp())


def fold_roles(protocol, fold):
    cfg = read(ROOT / f'configs/official_{protocol.lower()}.json')
    chosen = cfg['folds'][fold]
    if protocol == 'TS':
        roles = []
        for group in ('group_1', 'group_2'):
            for key, (a, b) in cfg[group].items():
                split = next(s for s in ('train', 'val', 'test') if key in chosen[s])
                roles.append(dict(start=unix(a), end=unix(b), split=split, segment=f'{group}_{key}'))
        return {cow: merge_segments(roles) for cow in range(1, 11)}
    a, b = unix('2023-07-25 00:00:00'), unix('2023-07-26 00:00:00')
    return {int(cow): [dict(start=a, end=b, split=split, segment='OS_original_0725_day')]
            for split in ('train', 'val', 'test') for cow in chosen[split]}


def coverage(x, rows):
    valid = x[:, :, 6].astype(bool)
    output = []
    for cow in sorted({r['cow_id'] for r in rows}):
        for split in ('train', 'val'):
            ids = np.asarray([i for i, r in enumerate(rows) if r['cow_id'] == cow and r['split'] == split])
            if not len(ids):
                continue
            v = valid[ids]
            output.append(dict(cow_id=cow, split=split, targets=len(ids),
                               valid_point_fraction=float(v.mean()),
                               fully_missing_windows=int((~v.any(1)).sum()),
                               complete_window_fraction=float(v.all(1).mean()),
                               empty_tail_readouts=int((~v[:, -10:].any(1)).sum())))
    return output


def prepare(protocol, fold, sources, freeze):
    parent, output = BASE / protocol / fold, OUT / 'data' / protocol / fold
    receipt = output / 'DATA_RECEIPT.json'
    if receipt.exists():
        for name, digest in read(receipt)['outputs'].items():
            assert sha(output / name) == digest
        return read(receipt)
    output.mkdir(parents=True, exist_ok=True)
    parent_hashes = {}
    for name in ('targets.jsonl', 'normalizer.json', 'windows.npz', 'DATA_RECEIPT.json'):
        path = parent / name
        relative = path.relative_to(ROOT).as_posix()
        assert '/data/selection/' in relative
        parent_hashes[relative] = sha(path)
        assert parent_hashes[relative] == freeze[relative], 'Frozen selection input changed'
    assert read(parent / 'DATA_RECEIPT.json')['phase'] == 'selection'
    # These files contain TRAIN/VAL only. No formal/Test target file is opened.
    metadata = [json.loads(line) for line in (parent / 'targets.jsonl').read_text().splitlines()]
    if {r['split'] for r in metadata} != {'train', 'val'}:
        raise ValueError('Selection roster must contain only TRAIN and VAL')
    rows = [{k: r[k] for k in ('cow_id', 'timestamp', 'start', 'end', 'split', 'segment')} for r in metadata]
    assert len({(r['cow_id'], r['timestamp']) for r in rows}) == len(rows)
    roles = fold_roles(protocol, fold)
    for row in rows:
        role = next(s for s in roles[row['cow_id']] if s['start'] <= row['timestamp'] < s['end'])
        assert role['split'] == row['split'] and role['start'] == row['start'] and role['end'] == row['end']
    train_roles = {c: [s for s in rr if s['split'] == 'train'] for c, rr in roles.items()}
    train_roles = {c: rr for c, rr in train_roles.items() if rr}
    normal = fit_scaler(sources, train_roles)
    native, native_source = windows(rows, sources, normal)
    low, low_first, low_last = low_frequency_windows(rows, sources, normal)
    with np.load(parent / 'windows.npz', allow_pickle=False) as cached:
        arrays = {m: {k: cached[f'{m}_{k}'] for k in ('x', 'valid', 'fresh', 'age')} for m in ('uwb', 'hd', 'akl')}
    context, names = descriptors(arrays, read(parent / 'normalizer.json'))
    current = np.concatenate([tokens({k: torch.from_numpy(np.array(v[:, -1:], copy=True)) for k, v in arrays[m].items()}).numpy()[:, 0]
                              for m in ('uwb', 'hd', 'akl')], axis=1)
    assert current.shape == (len(rows), 26)
    context = np.concatenate([context, current], axis=1)
    assert context.shape == (len(rows), 136) and np.isfinite(context).all()
    train = np.asarray([r['split'] == 'train' for r in rows])
    context_mean = context[train].astype(np.float64).mean(0)
    context_std = np.maximum(context[train].astype(np.float64).std(0), 1e-6)
    normalized_context = ((context - context_mean) / context_std).astype(np.float32)
    assert np.isfinite(normalized_context).all()
    assert np.array_equal(native_source >= 0, native[:, :, 6].astype(bool))
    assert np.array_equal(low_first >= 0, low[:, :, 6].astype(bool))
    assert np.array_equal(low_last >= 0, low[:, :, 6].astype(bool))
    np.savez(output / 'inputs.npz', native=native, low_frequency=low, context=normalized_context,
             native_source_tick=native_source, low_first_tick=low_first, low_last_tick=low_last,
             train_mask=train)
    shutil.copyfile(parent / 'targets.jsonl', output / 'targets.jsonl')
    save(output / 'normalizer.json', dict(physical=normal,
         context=dict(mean=context_mean.tolist(), std=context_std.tolist(), scope='TRAIN target descriptors only'),
         context_feature_names=names + [f'current_token_{i}' for i in range(26)]))
    assert sha(output / 'targets.jsonl') == sha(parent / 'targets.jsonl')
    result = dict(protocol=protocol, fold=fold, phase='selection', targets=len(rows),
                  TRAIN_targets=int(train.sum()), VAL_targets=int((~train).sum()),
                  unchanged_target_order_and_labels=True, availability_target_filter=False,
                  parents=parent_hashes, physical_scaler=normal,
                  native_coverage=coverage(native, rows), low_frequency_coverage=coverage(low, rows),
                  no_future_source=True, no_source_before_window_or_role=True,
                  Test_files_opened=0, VAL_metrics_computed=0,
                  outputs={name: sha(output / name) for name in ('inputs.npz', 'targets.jsonl', 'normalizer.json')})
    save(receipt, result)
    return result


def main():
    if Path.cwd() != ROOT:
        raise RuntimeError('Run scientific preparation only in cloud workspace')
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    inventory = read(INVENTORY / 'FULL_DAY_TRAIN_INVENTORY.json')
    assert inventory['status'] == 'FULL_DAY_TRAIN_INVENTORY_COMPLETE'
    sources, source_hashes = {}, {}
    for cow in inventory['cows']:
        path = INVENTORY / 'raw_os_train_union' / f'T{cow["cow_id"]:02d}_0725.csv'
        source_hashes[path.relative_to(ROOT).as_posix()] = sha(path)
        assert sha(path) == cow['cached_csv_sha256']
        sources[cow['cow_id']] = canonical_source(pd.read_csv(path))
    freeze = read(ROOT / 'refine-logs/sensor_motion_backbone_selection_20261020/REFERENCE_FREEZE.json')['selection_cache_SHA256']
    all_results = []
    for protocol in ('TS', 'OS'):
        for fold in (f'fold_{i}' for i in range(1, 6)):
            result = prepare(protocol, fold, sources, freeze)
            all_results.append(result)
            save(OUT / 'PROGRESS.json', dict(completed=len(all_results), total=10, protocol=protocol, fold=fold))
            print('PREPARED', protocol, fold, 'targets', result['targets'], flush=True)
    save(OUT / 'DATA_PREFLIGHT.json', dict(status='PASS', caches=all_results, raw_source_SHA256=source_hashes,
         input_shapes=dict(native=[100, 8], low_frequency=[100, 8], context=[136]),
         training_started=False, Test_files_opened=0, VAL_metrics_computed=0))
    print('ALL_SELECTION_CACHES_PREPARED', flush=True)


if __name__ == '__main__':
    main()
