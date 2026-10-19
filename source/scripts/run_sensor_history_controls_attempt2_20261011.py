"""Explicit parent-schema adapter; frozen scientific recipe and first attempt retained."""
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from q3_sensor.endpoint_models import EndpointFusion
from q3_sensor.protocol import build_windows, fit_normalizer, sha
from scripts import run_sensor_history_controls_20261011 as runner
from scripts.run_sensor_formal_ts import official_roles
from scripts.sensor_os_protocol import assert_window_sources, cow_roles, day_role, os_normalizer

REPAIR = Path('reports/sensor_history_controls_20261011/REPAIR_FREEZE_attempt2.json')


def freeze_repair(cfg, root):
    paths = [Path(__file__), Path('scripts/launch_sensor_history_controls_attempt2_20261011.py'),
             root / 'run_identity.json', root / 'METHOD_LOCK.json',
             Path(cfg['formal_source']) / 'source_receipt.json']
    for protocol in cfg['protocols']:
        parent = Path(cfg['parent_roots'][protocol])
        paths.append(parent / 'run_identity.json')
        for fold in cfg['folds']:
            paths.extend(parent / fold / name for name in
                         ['targets.jsonl', 'normalizer.json', 'predictions.csv', 'COMPLETE.json'] +
                         [f'NN_seed{s}.pt' for s in cfg['seeds']])
    hashes = {str(p): sha(p) for p in paths}
    if REPAIR.exists():
        if runner.read_json(REPAIR)['source_sha256'] != hashes:
            raise AssertionError('Repair freeze or parents changed')
    else:
        states = runner.read_json(root / 'JOB_STATES.json')
        if any(v['state'] == 'completed' for k, v in states.items() if not k.startswith('development/')):
            raise AssertionError('Cannot introduce schema repair after formal fits')
        runner.write_json(REPAIR, dict(reason='Legacy TS COMPLETE lacks OS identity and checkpoint digest fields',
            implementation_change_only=True, scientific_configuration_changed=False,
            original_attempt_source_retained=True, source_sha256=hashes,
            created_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    for protocol in cfg['protocols']:
        parent = Path(cfg['parent_roots'][protocol])
        identity = runner.read_json(parent / 'run_identity.json')
        if identity['source_receipt_sha256'] != sha(Path(cfg['formal_source']) / 'source_receipt.json'):
            raise AssertionError('Parent raw receipt differs')
        for fold in cfg['folds']:
            complete = runner.read_json(parent / fold / 'COMPLETE.json')
            if complete['target_sha256'] != sha(parent / fold / 'targets.jsonl'):
                raise AssertionError('Target digest mismatch')
            if protocol == 'TS':
                if complete['method_lock_sha256'] != identity['method_lock_sha256'] or not complete['source_targets_same']:
                    raise AssertionError('Legacy TS method identity mismatch')
            elif complete['identity_sha256'] != sha(parent / 'run_identity.json'):
                raise AssertionError('OS identity mismatch')
    print('HISTORY_CONTROLS_PARENT_SCHEMA_PREFLIGHT_PASS', flush=True)


def formal_data(cfg, protocol, fold, frames):
    frozen = runner.read_json(REPAIR)['source_sha256']
    parent = Path(cfg['parent_roots'][protocol]) / fold
    complete = runner.read_json(parent / 'COMPLETE.json')
    original = runner.read_json(parent.parent / 'run_identity.json')
    if complete['target_sha256'] != sha(parent / 'targets.jsonl'):
        raise AssertionError('Parent target changed')
    if protocol == 'TS':
        if complete['method_lock_sha256'] != original['method_lock_sha256'] or not complete['source_targets_same']:
            raise AssertionError('Legacy TS method receipt mismatch')
    elif complete['identity_sha256'] != sha(parent.parent / 'run_identity.json'):
        raise AssertionError('OS parent identity mismatch')
    for path, digest in original['sources'].items():
        if path.startswith('q3_sensor/') and sha(path) != digest:
            raise AssertionError(f'Frozen sensor source changed {path}')
    targets = pd.read_json(parent / 'targets.jsonl', lines=True, convert_dates=False)
    if targets.duplicated(['cow_id', 'timestamp']).any():
        raise AssertionError('Duplicate parent targets')
    train = targets.split.eq('train').to_numpy()
    if protocol == 'TS':
        official = runner.read_json('configs/official_ts.json')
        roles = official_roles(official, official['folds'][fold])
        normalizer = fit_normalizer(frames, roles)
        for row in targets.itertuples():
            if not any(r['split'] == row.split and r['start'] <= row.timestamp < r['end'] for r in roles):
                raise AssertionError('Illegal TS role')
    else:
        official = runner.read_json('configs/official_os.json')
        _, fit_ids, test_ids = cow_roles(official['folds'][fold])
        if set(targets.loc[train, 'cow_id']) != set(fit_ids) or set(targets.loc[~train, 'cow_id']) != set(test_ids):
            raise AssertionError('OS cow split mismatch')
        normalizer = os_normalizer(frames, fit_ids, day_role(runner.read_json('configs/sensor_os_20261007.json')))
    if normalizer != runner.read_json(parent / 'normalizer.json'):
        raise AssertionError('Recomputed Fit normalizer differs')
    for cow, group in targets.groupby('cow_id'):
        gt = pd.read_csv(Path(cfg['formal_source']) / f'behavior_labels/individual/C{cow:02d}_0725.csv').set_index('timestamp').behavior
        np.testing.assert_array_equal(gt.loc[group.timestamp].to_numpy(dtype=int), group.label.to_numpy())
    windows = build_windows(targets.to_dict('records'), frames, normalizer)
    assert_window_sources(targets, windows)
    tensors = runner.to_tensors(windows)
    del windows
    baseline = pd.read_csv(parent / 'predictions.csv')
    expected = targets.loc[~train, ['cow_id', 'timestamp', 'label']].reset_index(drop=True)
    pd.testing.assert_frame_equal(baseline[['cow_id', 'timestamp', 'label']], expected)
    parent_files = {str(p): frozen[str(p)] for p in [parent / 'targets.jsonl', parent / 'normalizer.json',
                   parent / 'predictions.csv', parent / 'COMPLETE.json', parent.parent / 'run_identity.json']}
    parent_files[str(Path(__file__))] = sha(__file__)
    parent_files[str(REPAIR)] = sha(REPAIR)
    replay = []
    counts = np.bincount(targets.label.to_numpy()[train], minlength=8)[1:8]
    for seed in cfg['seeds']:
        path = parent / f'NN_seed{seed}.pt'
        digest = sha(path)
        if digest != frozen[str(path)]:
            raise AssertionError('Parent checkpoint differs from preflight freeze')
        if protocol == 'OS' and digest != complete['checkpoint_sha256'][path.name]:
            raise AssertionError('OS historical checkpoint digest differs')
        parent_files[str(path)] = digest
        state = torch.load(path, map_location='cpu', weights_only=False)
        model = EndpointFusion('gru', counts / counts.sum()).cuda().eval()
        model.load_state_dict(state['state_dict'], strict=True)
        prediction = runner.predict(model, tensors, np.flatnonzero(~train), cfg['batch_size'])
        np.testing.assert_array_equal(prediction, baseline[f'NN_seed{seed}'].to_numpy())
        replay.append(dict(seed=seed, targets=len(prediction), hard_mismatches=0,
                           checkpoint_identity='repair_prefit_snapshot' if protocol == 'TS' else 'historical_OS_receipt_and_repair_snapshot'))
        del model, state
    gc.collect()
    torch.cuda.empty_cache()
    return dict(targets=targets, train=train, tensors=tensors, normalizer=normalizer,
                parent_files=parent_files, parent_replay=replay)


def main():
    cfg, root = runner.setup()
    freeze_repair(cfg, root)
    if '--preflight-only' in sys.argv:
        return
    runner.formal_data = formal_data
    runner.main()


if __name__ == '__main__':
    main()
