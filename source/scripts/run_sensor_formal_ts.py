"""Locked fivefold original-target comparison; TEST never selects parameters."""
import copy
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier

from q3_sensor.endpoint_models import EndpointFusion
from q3_sensor.evaluation import metrics
from q3_sensor.protocol import MODALITIES, FEATURES, build_windows, fit_normalizer, merge_segments, sha
from scripts.train_sensor_baselines import tensor_batch


def official_roles(config, fold):
    import pytz
    from datetime import datetime
    zone = pytz.timezone('America/Chicago')
    result = []
    for group in ('group_1', 'group_2'):
        for key, (a, b) in config[group].items():
            start, end = [int(zone.localize(datetime.strptime(s, '%Y-%m-%d %H:%M:%S')).timestamp()) for s in (a, b)]
            result.append(dict(start=start, end=end, split='test' if key in fold['test'] else 'train', segment=f'{group}_{key}'))
    return merge_segments(result)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('Cloud CUDA required')
    lock_path = Path('reports/sensor_endpoint_screen_20261007/METHOD_LOCK.json')
    lock = json.loads(lock_path.read_text())
    source = Path('data/sensor_formal_0725_20261007')
    access = json.loads((source / 'ACCESS_LOCK.json').read_text())
    if access['method_lock_sha256'] != sha(lock_path) or not lock['screen']['promotion_passed']:
        raise RuntimeError('Frozen passing method/data identity mismatch')
    cfg, spec = lock['training'], lock['model']
    root = Path('reports/sensor_formal_ts_20261007')
    root.mkdir(parents=True, exist_ok=True)
    identity = dict(method_lock_sha256=sha(lock_path), source_receipt_sha256=sha(source / 'source_receipt.json'),
                    config=cfg, model=spec, precision='FP32_TF32_disabled', device=torch.cuda.get_device_name(0),
                    sources={str(p): sha(p) for p in list(Path('q3_sensor').glob('*.py')) + [Path(__file__)]})
    if (root / 'run_identity.json').exists() and json.loads((root / 'run_identity.json').read_text()) != identity:
        raise RuntimeError('Formal source/method changed after TEST access')
    (root / 'run_identity.json').write_text(json.dumps(identity, indent=2))
    torch.set_num_threads(cfg['threads'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    module_spec = importlib.util.spec_from_file_location('sensor_author_formal_loader', 'reports/sensor_baselines_20261007/upstream/data_loader.py')
    author = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = author
    module_spec.loader.exec_module(author)
    original_preload = author.uwb_hd_aa_pre_loader
    parts, frames = [], {}
    for cow in range(1, 11):
        official = original_preload(str(source), [cow])
        official['cow_id'] = cow  # Metadata only, excluded from RF and network inputs.
        parts.append(official)
        tag, name = f'T{cow:02d}', f'C{cow:02d}'
        frames[cow] = {
            'uwb': pd.read_csv(source / f'main_data/uwb/{tag}/{tag}_0725.csv'),
            'hd': pd.read_csv(source / f'sub_data/head_direction/{tag}/{tag}_0725.csv'),
            'akl': pd.read_csv(source / f'main_data/ankle/{name}/{name}_0725.csv')}
    annotated_day = pd.concat(parts, ignore_index=True)
    # Preserve the author's cow order, block order and train-then-val order.
    # Only avoid repeated CSV parsing; feature/GT arrays remain unchanged.
    def annotated_preload(sensor_data_dir, id_list, date='0725', label=True):
        if id_list != list(range(1, 11)) or date != '0725' or not label:
            raise ValueError('Unexpected author preload call')
        return annotated_day.copy()
    author.uwb_hd_aa_pre_loader = annotated_preload
    config = json.loads(Path('configs/official_ts.json').read_text())
    features = ['coord_x_cm', 'coord_y_cm', 'coord_z_cm', 'roll', 'pitch', 'yaw', 'lying']
    for fold_name, fold in config['folds'].items():
        directory = root / fold_name
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / 'COMPLETE.json').exists():
            continue
        train_frame, _, test_frame = author.data_loader_s2('uwb_hd_aa_pre_loader', str(source),
                                                          copy.deepcopy(config['group_1']), copy.deepcopy(config['group_2']),
                                                          fold, val=False, timestamp=True)
        roles = official_roles(config, fold)
        rows = []
        for split, frame in [('train', train_frame), ('test', test_frame)]:
            for r in frame.itertuples(index=False):
                s = next(s for s in roles if s['split'] == split and s['start'] <= r.timestamp < s['end'])
                rows.append(dict(cow_id=int(r.cow_id), timestamp=int(r.timestamp), label=int(r.behavior), **s))
        targets = pd.DataFrame(rows)
        if targets.duplicated(['cow_id', 'timestamp']).any():
            raise RuntimeError('Duplicate author targets require explicit review')
        targets.to_json(directory / 'targets.jsonl', orient='records', lines=True)
        train = targets.split.to_numpy() == 'train'
        label = targets.label.to_numpy(dtype=int)
        normalizer = fit_normalizer(frames, roles)
        (directory / 'normalizer.json').write_text(json.dumps(normalizer, indent=2))
        windows = build_windows(rows, frames, normalizer)
        for m, data in windows.items():
            safe = np.isfinite(data['source'])
            grid = targets.timestamp.to_numpy()[:, None] + np.arange(-59, 1)
            assert np.all(np.where(safe, data['source'] <= grid[..., None], True))
            assert np.all(np.where(safe, data['source'] >= targets.start.to_numpy()[:, None, None], True))
        tensors = {m: {k: torch.from_numpy(data[k]).bool() if k == 'valid' else torch.from_numpy(data[k])
                       for k in ('x', 'valid', 'fresh', 'age')} for m, data in windows.items()}
        y = torch.from_numpy(label - 1).long().cuda()
        counts = np.bincount(label[train], minlength=8)[1:8]
        if np.any(counts == 0):
            raise RuntimeError('Missing TRAIN class in formal fold')
        prior = counts / counts.sum()
        train_indices, test_indices = np.flatnonzero(train), np.flatnonzero(~train)
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=cfg['threads'])
        rf.fit(train_frame[features].to_numpy(), train_frame.behavior.to_numpy(dtype=int))
        rf_pred = rf.predict(test_frame[features].to_numpy()).astype(int)
        assert np.array_equal(test_frame.behavior.to_numpy(dtype=int), label[~train])
        prediction_table = targets.loc[~train, ['cow_id', 'timestamp', 'label']].reset_index(drop=True)
        prediction_table['RF'] = rf_pred
        records = dict(RF=metrics(label[~train], rf_pred))
        torch.save(dict(kind='author_RF', features=features, model=rf), directory / 'RF.pt')
        for seed in cfg['seeds']:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            rng = np.random.default_rng(seed)
            weights = counts.sum() / (7 * counts)
            if spec['loss_weight'] == 'sqrt_balanced':
                weights = np.sqrt(weights)
            weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device='cuda')
            model = EndpointFusion(spec['encoder'], prior).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
            criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
            start = time.monotonic()
            history = []
            for epoch in range(1, cfg['epochs'] + 1):
                model.train()
                order = rng.permutation(train_indices)
                numerator, denominator = 0., 0.
                for at in range(0, len(order), cfg['batch_size']):
                    idx = order[at:at + cfg['batch_size']]
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(tensor_batch(tensors, idx, 'cuda')), y[idx])
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite formal training loss')
                    loss.backward()
                    optimizer.step()
                    batch_weight = float(weight_tensor[y[idx]].sum())
                    numerator += float(loss.detach()) * batch_weight
                    denominator += batch_weight
                history.append(dict(epoch=epoch, train_loss=numerator / denominator))
                print('FORMAL_TRAIN_EPOCH', fold_name, seed, epoch, history[-1]['train_loss'], flush=True)
            # Only the predetermined final checkpoint sees TEST, exactly once.
            model.eval()
            prediction = []
            with torch.no_grad():
                for at in range(0, len(test_indices), cfg['batch_size']):
                    idx = test_indices[at:at + cfg['batch_size']]
                    prediction.append(model(tensor_batch(tensors, idx, 'cuda')).argmax(1).cpu().numpy() + 1)
            prediction = np.concatenate(prediction)
            prediction_table[f'NN_seed{seed}'] = prediction
            record = metrics(label[~train], prediction)
            record.update(seed=seed, epochs=cfg['epochs'], seconds=time.monotonic() - start,
                          train_samples=len(train_indices), selection='fixed_final_epoch_no_TEST_selection')
            records[f'NN_seed{seed}'] = record
            torch.save(dict(kind='EndpointFusion', encoder=spec['encoder'], state_dict=model.cpu().state_dict(),
                            prior=prior, normalizer=normalizer, config=cfg, fold=fold_name), directory / f'NN_seed{seed}.pt')
            (directory / f'history_seed{seed}.json').write_text(json.dumps(history, indent=2))
            print('FORMAL_TEST_COMPLETE', fold_name, seed, record['macro_f1'], flush=True)
        prediction_table.to_csv(directory / 'predictions.csv', index=False)
        (directory / 'COMPLETE.json').write_text(json.dumps(dict(records=records, fit='author_TRAIN_plus_VAL',
                                                                source_targets_same=True, target_sha256=sha(directory / 'targets.jsonl'),
                                                                method_lock_sha256=sha(lock_path)), indent=2))
    print('SENSOR_FORMAL_TS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
