"""Bounded CPU XGBoost convergence on existing outer-TRAIN Fit/DEV only.

The fixed 866-feature / sqrt-balanced / hist depth6 eta0.05 recipe is unchanged.
Convergence means observed training and DEV stationarity, not global optimality.
Best DEV checkpoint and terminal convergence are separate outputs. No Test loader.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

CONFIG = Path('configs/sensor_tree_convergence_20261027_v1.json')
ROOT = Path('/q3/mmcows-sensor-behavior-20261007')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def class_weights(labels):
    labels = np.asarray(labels, dtype=np.int64)
    if not len(labels) or not np.isin(labels, np.arange(7)).all():
        raise ValueError('Fit must contain labels 0..6')
    counts = np.bincount(labels, minlength=7)
    if (counts == 0).any():
        raise ValueError('Every class must exist in Fit')
    return np.sqrt(counts.sum() / (7 * counts)), counts / counts.sum()


def metrics(labels, probabilities, weights):
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != (len(y), 7) or not len(y) or not np.isin(y, np.arange(7)).all():
        raise ValueError('Seven-class prediction/target shape mismatch')
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError('Invalid probabilities')
    loss = -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1))
    cm = np.zeros((7, 7), dtype=np.int64)
    np.add.at(cm, (y, p.argmax(1)), 1)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * np.diag(cm), denom, out=np.zeros(7), where=denom > 0)
    return dict(weighted_CE=float(np.average(loss, weights=np.asarray(weights)[y])),
                CE=float(loss.mean()), macro_f1_percent=float(100 * f1.mean()),
                class_f1_percent=(100 * f1).tolist(), confusion_matrix=cm.tolist())


def assess(history, cfg):
    """Require at least four actual improving-or-stationary block transitions.

    Absolute drift rejects both ongoing improvement and deteriorating overfit.
    Fit objective and DEV must plateau together, while DEV remains near run best.
    """
    count = cfg['consecutive_blocks']
    if len(history) < count + 1 or history[-1]['rounds'] < cfg['minimum_rounds']:
        return dict(passed=False, reason='insufficient_rounds_or_blocks')
    rows = history[-count - 1:]
    best = min(r['dev']['weighted_CE'] for r in history)
    transitions = []
    for before, after in zip(rows, rows[1:]):
        item = {}
        for key, limit_key in [('train', 'max_abs_train_CE_relative_change'),
                               ('dev', 'max_abs_DEV_CE_relative_change')]:
            old, new = before[key]['weighted_CE'], after[key]['weighted_CE']
            drift = (new - old) / max(old, 1e-15)
            item[key + '_CE_relative_change'] = drift
            item[key + '_CE_passed'] = math.isfinite(drift) and abs(drift) <= cfg[limit_key]
        item['DEV_F1_change_pp'] = after['dev']['macro_f1_percent'] - before['dev']['macro_f1_percent']
        item['passed'] = bool(item['train_CE_passed'] and item['dev_CE_passed'] and
                              abs(item['DEV_F1_change_pp']) <= cfg['max_abs_DEV_F1_change_pp'])
        transitions.append(item)
    f1 = [r['dev']['macro_f1_percent'] for r in rows]
    near_best = all(r['dev']['weighted_CE'] <= best * cfg['DEV_CE_within_run_minimum_factor'] for r in rows)
    f1_range = max(f1) - min(f1)
    return dict(passed=bool(all(r['passed'] for r in transitions) and near_best and
                            f1_range <= cfg['max_DEV_F1_window_range_pp']),
                transitions=transitions, DEV_CE_near_run_best=bool(near_best),
                DEV_F1_window_range_pp=f1_range, best_DEV_CE=best,
                scope='empirical Fit objective and DEV stationarity; not global optimality')


def params(cfg, seed):
    old = read(cfg['model_config'])
    options = old['xgboost']
    return dict(objective='multi:softprob', num_class=7, tree_method='hist', device='cpu',
                nthread=cfg['threads'], eta=options['eta'], max_depth=options['max_depth'],
                seed=seed, subsample=1., colsample_bytree=1., disable_default_eval_metric=1)


def load_case(cfg, protocol, inner):
    """Only these three hashed, existing Fit/DEV files can supply scientific data."""
    from experiments.sensor_comparator_pilot_20261022_v2.cpu import summary_features
    folder = Path(cfg['data_parent']) / 'data' / protocol / f'inner{inner}'
    receipt_path = folder / 'RECEIPT.json'
    receipt = read(receipt_path)
    if (receipt['protocol'], receipt['inner']) != (protocol, inner):
        raise ValueError('Source identity mismatch')
    names = ('FIT.npz', 'DEV_INPUTS.npz', 'DEV_TARGETS.jsonl')
    files = {str(receipt_path): sha(receipt_path)}
    for name in names:
        path = folder / name
        digest = sha(path)
        if digest != receipt['files'][name]:
            raise ValueError('Source hash mismatch: ' + str(path))
        files[str(path)] = digest
    groups = []
    for name in names[:2]:
        with np.load(folder / name, allow_pickle=False) as archive:
            data = {k: archive[k] for k in archive.files}
        if name == 'DEV_INPUTS.npz' and 'labels' in data:
            raise ValueError('DEV inputs unexpectedly contain labels')
        features = summary_features(dict(context=data['context'], current=data['current'],
                                        native=data['dynamic'], state=data['state']))
        if features.shape != (len(data['dynamic']), 866):
            raise ValueError('Original 866-feature interface changed')
        missing = ~((data['dynamic'][:, :, 6] > .5).any(1) | data['state_valid'].any(1))
        groups.append(dict(features=features, missing=missing,
                           labels=data.get('labels')))
    targets = [json.loads(line) for line in (folder / names[2]).read_text().splitlines()]
    dev_ids = np.asarray([r['original_id'] for r in targets])
    if len(dev_ids) != len(set(dev_ids.tolist())):
        raise ValueError('Duplicate DEV identity')
    groups[1]['labels'] = np.asarray([r['label'] - 1 for r in targets], np.int64)
    for group, key in zip(groups, ('fit_rows', 'held_rows')):
        if len(group['labels']) != receipt[key] or len(group['features']) != receipt[key]:
            raise ValueError('Source row counts changed')
    return groups[0], groups[1], files


def train_case(cfg, fit, dev, seed, out, synthetic=False):
    import xgboost as xgb
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    marker = out / 'STARTED.json'
    # Exclusive creation prevents duplicate workers and preserves failure history.
    with marker.open('x', encoding='utf-8') as handle:
        json.dump(dict(pid=os.getpid(), unix=time.time(), synthetic=synthetic), handle)
    weight, prior = class_weights(fit['labels'])
    dm = xgb.DMatrix(fit['features'], label=fit['labels'], weight=weight[fit['labels']], nthread=cfg['threads'])
    vm = xgb.DMatrix(dev['features'], nthread=cfg['threads'])
    model, history, best_ce, best_rounds = None, [], float('inf'), None
    started = time.monotonic()
    verdict = dict(passed=False, reason='not_started')
    for rounds in range(cfg['block_rounds'], cfg['max_rounds'] + 1, cfg['block_rounds']):
        model = xgb.train(params(cfg, seed), dm, num_boost_round=cfg['block_rounds'], xgb_model=model)
        if model.num_boosted_rounds() != rounds:
            raise RuntimeError('Block continuation lost history')
        pfit, pdev = model.predict(dm).astype(np.float64), model.predict(vm).astype(np.float64)
        fit_metrics = metrics(fit['labels'], pfit, weight)
        # Same entirely-missing fallback as the existing tree expert.
        pfit[fit['missing']], pdev[dev['missing']] = prior, prior
        row = dict(rounds=rounds, train=fit_metrics, fit_eval=metrics(fit['labels'], pfit, weight),
                   dev=metrics(dev['labels'], pdev, weight), elapsed_seconds=time.monotonic() - started)
        history.append(row)
        if row['dev']['weighted_CE'] < best_ce:
            best_ce, best_rounds = row['dev']['weighted_CE'], rounds
            model.save_model(out / 'best_DEV.json')
            np.save(out / 'best_DEV_probabilities.npy', pdev)
        model.save_model(out / 'latest.json')
        verdict = assess(history, cfg['convergence'])
        save(out / 'HISTORY.json', history)
        save(out / 'PROGRESS.json', dict(rounds=rounds, best_DEV_rounds=best_rounds,
                                       best_DEV_CE=best_ce, convergence=verdict))
        print(json.dumps(dict(rounds=rounds, train_CE=row['train']['weighted_CE'],
                              DEV_CE=row['dev']['weighted_CE'], DEV_F1=row['dev']['macro_f1_percent'],
                              converged=verdict['passed'])), flush=True)
        if verdict['passed'] or time.monotonic() - started >= cfg['worker_seconds']:
            break
    restored = xgb.Booster()
    restored.load_model(out / 'latest.json')
    restored.set_param({'nthread': cfg['threads']})
    np.testing.assert_array_equal(model.predict(vm), restored.predict(vm))
    restored.load_model(out / 'best_DEV.json')
    restored.set_param({'nthread': cfg['threads']})
    replay = restored.predict(vm).astype(np.float64)
    replay[dev['missing']] = prior
    np.testing.assert_array_equal(replay, np.load(out / 'best_DEV_probabilities.npy'))
    save(out / 'COMPLETE.json', dict(status='PASS' if verdict['passed'] else 'FAIL',
         termination='stationarity' if verdict['passed'] else 'budget_without_convergence',
         terminal_rounds=history[-1]['rounds'], best_DEV_rounds=best_rounds, best_DEV_CE=best_ce,
         convergence=verdict, best_DEV_and_latest_replay_exact=True, Test_rows_read=0,
         artifacts_sha256={name: sha(out / name) for name in
                           ('latest.json', 'best_DEV.json', 'best_DEV_probabilities.npy', 'HISTORY.json')},
         elapsed_seconds=time.monotonic() - started, synthetic=synthetic))
    return read(out / 'COMPLETE.json')


def validate_config(cfg):
    if cfg['block_rounds'] <= 0 or cfg['max_rounds'] % cfg['block_rounds']:
        raise ValueError('Round budget must contain complete positive blocks')
    if cfg['convergence']['consecutive_blocks'] < 3:
        raise ValueError('At least three consecutive transitions required')
    if any(cfg[k] for k in ('Test_allowed', 'outer_VAL_allowed', 'formal_refit_allowed', 'hyperparameter_search')):
        raise ValueError('Only existing TRAIN-internal development scope is authorized')
    if cfg['seeds'] != [17]:
        raise ValueError('Formal full-sampling tree uses fixed seed17 only')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(CONFIG))
    parser.add_argument('--protocol', choices=['TS', 'OS'])
    parser.add_argument('--inner', type=int, choices=[0, 1, 2])
    parser.add_argument('--seed', type=int, choices=[17], default=17)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--output', help='Synthetic outputs only; real paths are configuration-bound')
    args = parser.parse_args()
    cfg = read(args.config)
    validate_config(cfg)
    if args.synthetic:
        generator = np.random.default_rng(17)
        x = generator.normal(size=(210, 866)).astype(np.float32)
        y = np.arange(len(x)) % 7
        fit = dict(features=x, labels=y, missing=np.zeros(len(x), bool))
        cfg = dict(cfg, block_rounds=2, max_rounds=10, worker_seconds=120)
        cfg['convergence'] = dict(cfg['convergence'], minimum_rounds=500)
        result = train_case(cfg, fit, fit, 17, args.output or Path(cfg['output']) / 'synthetic', synthetic=True)
        if result['status'] != 'FAIL' or not result['best_DEV_and_latest_replay_exact']:
            raise RuntimeError('Synthetic budget exhaustion or serialization contract failed')
        print('SYNTHETIC_TREE_WITNESS_PASS: deliberately insufficient rounds correctly FAIL', flush=True)
        return
    if Path.cwd().resolve() != ROOT or args.protocol is None or args.inner is None:
        raise ValueError('Scientific fitting is cloud-only and requires protocol/inner')
    import xgboost as xgb
    if (xgb.__version__, np.__version__) != ('2.1.4', '1.24.4'):
        raise RuntimeError('Reuse frozen cloud dependencies; no installation')
    out = Path(cfg['output']) / args.protocol / f'inner{args.inner}' / f'seed{args.seed}'
    if out.exists():
        raise FileExistsError('Preserve original run; no implicit retry: ' + str(out))
    fit, dev, files = load_case(cfg, args.protocol, args.inner)
    dependencies = [Path(__file__), Path(args.config), Path(cfg['model_config']),
                    Path('experiments/sensor_comparator_pilot_20261022_v2/cpu.py')]
    save(out / 'SOURCES.json', dict(files=files, config=cfg, config_sha256=sha(args.config),
         implementation_sha256=sha(__file__), scope='outer TRAIN Fit/DEV only', Test_rows_read=0))
    save(out / 'METHOD.json', dict(implementation_files={str(p): sha(p) for p in dependencies},
         parameters=params(cfg, args.seed), xgboost_version=xgb.__version__, numpy_version=np.__version__,
         protocol=args.protocol, inner=args.inner, seed=args.seed,
         features=866, class_weight='sqrt(Nfit/(7*Fit_class_counts))', selection='DEV weighted CE only'))
    train_case(cfg, fit, dev, args.seed, out)


if __name__ == '__main__':
    main()
