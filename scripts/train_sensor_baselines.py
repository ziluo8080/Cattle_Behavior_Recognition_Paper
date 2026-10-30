"""Fold1 ordinary baselines on frozen common targets. No TEST support."""
import argparse
import json
import random
import time
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.ensemble import RandomForestClassifier

from q3_sensor.protocol import MODALITIES, sha
from q3_sensor.model import SensorConcat
from q3_sensor.evaluation import evaluate, metrics


def tensor_batch(tensors, indices, device='cpu'):
    return {m: {k: v[indices].to(device) for k, v in data.items()} for m, data in tensors.items()}


def save_result(output, system, seed, labels, predictions, ids, result, history, state=None):
    directory = output / f'{system}_seed{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    result.update(system=system, seed=seed, scope='fold1_TRAIN_fit_VAL_evaluation_exploratory', test_evaluated=False)
    pd.DataFrame(dict(cow_id=ids['cow_id'], timestamp=ids['timestamp'], label=labels,
                      prediction=predictions)).to_csv(directory / 'predictions.csv', index=False)
    (directory / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    if state is not None:
        torch.save(state, directory / 'checkpoint.pt')
    (directory / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('COMPLETE', system, seed, json.dumps(result), flush=True)


def run(source, output, frozen, systems=('O-RF', 'W-RF', 'C0', 'C1')):
    output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(frozen).read_text())
    if cfg['scope'] != 'fold1_TRAIN_VAL_no_TEST':
        raise ValueError('Only scoped development supported')
    provenance = dict(config=cfg, config_sha256=sha(frozen), source_receipt_sha256=sha(source / 'build_receipt.json'),
                      windows_sha256=sha(source / 'windows.npz'), official_csv_sha256=sha(source / 'official_development.csv'),
                      versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                                    torch=torch.__version__, sklearn=sklearn.__version__),
                      sources={str(p): sha(p) for p in list(Path('q3_sensor').glob('*.py')) +
                               [Path('scripts/train_sensor_baselines.py'), Path('scripts/build_sensor_development.py')]})
    receipt = output / 'run_identity.json'
    if receipt.exists() and json.loads(receipt.read_text()) != provenance:
        raise ValueError('Existing run identity differs; use a new versioned output directory')
    receipt.write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    torch.set_num_threads(cfg['threads'])
    device = torch.device(cfg['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    data = dict(np.load(source / 'windows.npz', allow_pickle=False))
    train = data['train']
    labels = data['label']
    counts = np.bincount(labels[train], minlength=8)[1:8]
    if (counts == 0).any():
        raise ValueError('All seven classes must have real TRAIN support')
    prior = counts / counts.sum()
    windows = {m: {k: data[f'{m}_{k}'][~train] for k in ('x', 'valid', 'fresh', 'age')} for m in MODALITIES}
    ids = {k: data[k][~train] for k in ('cow_id', 'timestamp')}
    validation_labels = labels[~train]
    if 'O-RF' in systems and not (output / 'O-RF_seed42/result.json').exists():
        official = pd.read_csv(source / 'official_development.csv')
        features = [c for c in official.columns if c not in ('timestamp', 'behavior', 'cow_id', 'split')]
        tr, va = official.split == 'train', official.split == 'val'
        start = time.monotonic()
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=cfg['threads'])
        clf.fit(official.loc[tr, features].values, official.loc[tr, 'behavior'].astype(int))
        prediction = clf.predict(official.loc[va, features].values).astype(int)
        gt = official.loc[va, 'behavior'].astype(int).to_numpy()
        result = metrics(gt, prediction)
        result.update(train_samples=int(tr.sum()), features=features, seconds=time.monotonic() - start,
                      author_pipeline='scoped feature reproduction; out-of-scope nearest matches excluded; TRAIN only')
        save_result(output, 'O-RF', 42, gt, prediction,
                    {k: official.loc[va, k].to_numpy() for k in ('cow_id', 'timestamp')}, result, [],
                    dict(kind='sklearn_RF', model=clf))
    if 'W-RF' in systems:
        flattened = []
        for m in MODALITIES:
            age = data[f'{m}_age']
            normalized = np.log1p(np.nan_to_num(age, nan=60.0).clip(0, 60)) / np.log(61)
            flattened.extend([data[f'{m}_{k}'].reshape(len(labels), -1) for k in ('x', 'valid', 'fresh')])
            flattened.append(normalized.reshape(len(labels), -1))
        x = np.concatenate(flattened, axis=1).astype(np.float32)
        missing = ~np.stack([windows[m]['valid'].any((1, 2)) for m in MODALITIES], axis=1).any(axis=1)
        for seed in cfg['seeds']:
            if (output / f'W-RF_seed{seed}/result.json').exists():
                continue
            start = time.monotonic()
            clf = RandomForestClassifier(n_estimators=100, random_state=seed, class_weight='balanced', n_jobs=cfg['threads'])
            clf.fit(x[train], labels[train])
            prediction = clf.predict(x[~train]).astype(int)
            prediction[missing] = int(prior.argmax() + 1)
            result = evaluate(validation_labels, prediction, windows)
            result.update(seconds=time.monotonic() - start, train_samples=int(train.sum()), features=x.shape[1])
            save_result(output, 'W-RF', seed, validation_labels, prediction, ids, result, [],
                        dict(kind='sklearn_RF', model=clf, prior=prior))
        del x
    tensors = {m: {k: torch.from_numpy(data[f'{m}_{k}']).bool() if k == 'valid'
                   else torch.from_numpy(data[f'{m}_{k}']) for k in ('x', 'valid', 'fresh', 'age')} for m in MODALITIES}
    train_indices = np.flatnonzero(train)
    val_indices = np.flatnonzero(~train)
    y = torch.from_numpy(labels - 1).long().to(device)
    weights = torch.tensor(counts.sum() / (7 * counts), dtype=torch.float32, device=device)
    for system in ('C0', 'C1'):
        if system not in systems:
            continue
        for seed in cfg['seeds']:
            if (output / f'{system}_seed{seed}/result.json').exists():
                continue
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(seed)
                torch.cuda.reset_peak_memory_stats()
            torch.use_deterministic_algorithms(True)
            rng = np.random.default_rng(seed)
            model = SensorConcat(system == 'C1', prior).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
            loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
            history, best, best_state, best_prediction = [], -1.0, None, None
            start = time.monotonic()
            for epoch in range(1, cfg['epochs'] + 1):
                model.train()
                order = rng.permutation(train_indices)
                loss_sum, weight_sum = 0.0, 0.0
                for at in range(0, len(order), cfg['batch_size']):
                    idx = order[at:at + cfg['batch_size']]
                    opt.zero_grad(set_to_none=True)
                    loss = loss_fn(model(tensor_batch(tensors, idx, device)), y[idx])
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite training loss')
                    loss.backward()
                    opt.step()
                    denominator = float(weights[y[idx]].sum())
                    loss_sum += float(loss.detach()) * denominator
                    weight_sum += denominator
                model.eval()
                predictions = []
                with torch.no_grad():
                    for at in range(0, len(val_indices), cfg['batch_size']):
                        idx = val_indices[at:at + cfg['batch_size']]
                        predictions.append(model(tensor_batch(tensors, idx, device)).argmax(1).cpu().numpy() + 1)
                prediction = np.concatenate(predictions)
                score = metrics(validation_labels, prediction)['macro_f1']
                record = dict(epoch=epoch, train_loss=loss_sum / weight_sum, val_macro_f1=score,
                              elapsed_seconds=time.monotonic() - start)
                history.append(record)
                print('EPOCH', system, seed, json.dumps(record), flush=True)
                if score > best:
                    best, best_prediction = score, prediction.copy()
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_epoch = epoch
            result = evaluate(validation_labels, best_prediction, windows)
            result.update(seconds=time.monotonic() - start, train_samples=len(train_indices),
                          parameters=sum(p.numel() for p in model.parameters()), best_epoch=best_epoch,
                          fixed_epochs=cfg['epochs'], class_weights=weights.cpu().tolist(),
                          device=str(device), gpu_name=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else None)
            save_result(output, system, seed, validation_labels, best_prediction, ids, result, history,
                        dict(kind='SensorConcat', include_age=system == 'C1', prior=prior, state_dict=best_state,
                             config=cfg, normalizer=json.loads((source / 'normalizer.json').read_text())))
    summaries = []
    for system in systems:
        records = [json.loads(p.read_text()) for p in sorted(output.glob(f'{system}_seed*/result.json'))]
        if records:
            summaries.append(dict(system=system, completed_seeds=[r['seed'] for r in records],
                                  macro_f1_mean=float(np.mean([r['macro_f1'] for r in records])),
                                  macro_f1_std=float(np.std([r['macro_f1'] for r in records])),
                                  per_class_f1_mean=np.mean([r['f1'] for r in records], axis=0).tolist()))
    (output / 'summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='data/sensor_windows_fold1_20261007')
    p.add_argument('--output', default='reports/sensor_baselines_20261007/runs')
    p.add_argument('--config', default='configs/sensor_baseline_cloud_run_20261007.json')
    p.add_argument('--systems', nargs='+', choices=('O-RF', 'W-RF', 'C0', 'C1'), default=['O-RF', 'W-RF', 'C0', 'C1'])
    a = p.parse_args()
    run(Path(a.source), Path(a.output), a.config, tuple(a.systems))


if __name__ == '__main__':
    main()
