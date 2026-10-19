"""Rescore fixed seven-class predictions and paired conditional intervals."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from q3_sensor.protocol import sha
from scripts.run_sensor_history_controls_20261011 import CONFIG, read_json, setup, write_json


def f1_from_cm(cm):
    true_positive = np.diagonal(cm, axis1=-2, axis2=-1)
    denominator = cm.sum(-1) + cm.sum(-2)
    return np.divide(2 * true_positive, denominator, out=np.zeros_like(true_positive, dtype=float), where=denominator != 0)


def group_cms(gt, predictions, group_ids, groups):
    return np.stack([np.bincount(group_ids * 49 + (gt - 1) * 7 + pred - 1,
                                 minlength=groups * 49).reshape(groups, 7, 7) for pred in predictions])


def main():
    cfg, root = setup()
    if read_json(root / 'FORMAL_COMPLETE.json')['identity_sha256'] != sha(root / 'run_identity.json'):
        raise AssertionError('Formal batch incomplete or changed')
    sources = {str(CONFIG): sha(CONFIG), str(Path(__file__)): sha(__file__),
               str(root / 'run_identity.json'): sha(root / 'run_identity.json'),
               str(root / 'METHOD_LOCK.json'): sha(root / 'METHOD_LOCK.json')}
    raw_gt = {}
    for cow in range(1, 11):
        path = Path(cfg['formal_source']) / f'behavior_labels/individual/C{cow:02d}_0725.csv'
        table = pd.read_csv(path)
        if table.timestamp.duplicated().any():
            raise AssertionError('Duplicate raw GT')
        raw_gt[cow] = table.set_index('timestamp').behavior
        sources[str(path)] = sha(path)
    systems = ['Base', *cfg['systems']]
    folds_table, seed_table, class_table = [], [], []
    results = {}
    for protocol in cfg['protocols']:
        generator_time = np.random.default_rng(cfg['bootstrap']['seed'])
        generator_cow = np.random.default_rng(cfg['bootstrap']['seed'])
        replicates = cfg['bootstrap']['replicates']
        bootstrap_time = np.zeros((replicates, 3))
        bootstrap_cow = np.zeros((replicates, 3)) if protocol == 'OS' else None
        scores, class_scores, folds, cows = [], [], [], []
        target_pairs = []
        for fold in cfg['folds']:
            parent = Path(cfg['parent_roots'][protocol]) / fold
            targets = pd.read_json(parent / 'targets.jsonl', lines=True, convert_dates=False)
            expected = targets.loc[targets.split.eq('test'), ['cow_id', 'timestamp', 'label']].reset_index(drop=True)
            baseline = pd.read_csv(parent / 'predictions.csv')
            pd.testing.assert_frame_equal(expected, baseline[['cow_id', 'timestamp', 'label']])
            for cow, rows in expected.groupby('cow_id'):
                np.testing.assert_array_equal(raw_gt[int(cow)].loc[rows.timestamp].to_numpy(dtype=int), rows.label.to_numpy())
            gt = expected.label.to_numpy(dtype=int)
            predictions = [baseline[f'NN_seed{s}'].to_numpy(dtype=int) for s in cfg['seeds']]
            target_pairs.extend(zip(expected.cow_id, expected.timestamp))
            for system in cfg['systems']:
                for seed in cfg['seeds']:
                    directory = root / protocol / fold / f'{system}_seed{seed}'
                    complete = read_json(directory / 'COMPLETE.json')
                    if complete['identity_sha256'] != sha(root / 'run_identity.json'):
                        raise AssertionError('Job identity differs')
                    for name, digest in complete['outputs'].items():
                        if sha(directory / name) != digest:
                            raise AssertionError('Job artifact changed')
                    for path, digest in complete['inputs']['parent_files'].items():
                        if sha(path) != digest:
                            raise AssertionError('Parent artifact changed')
                    table = pd.read_csv(directory / 'predictions.csv')
                    pd.testing.assert_frame_equal(table[['cow_id', 'timestamp', 'label']], expected)
                    pred = table.prediction.to_numpy(dtype=int)
                    if not np.isin(pred, range(1, 8)).all():
                        raise AssertionError('Invalid seven-class prediction')
                    cm = np.bincount((gt - 1) * 7 + pred - 1, minlength=49).reshape(7, 7)
                    record = read_json(directory / 'result.json')
                    np.testing.assert_array_equal(cm, record['metrics']['confusion_matrix'])
                    np.testing.assert_allclose(f1_from_cm(cm), record['metrics']['f1'], atol=1e-12, rtol=0)
                    np.testing.assert_allclose(f1_from_cm(cm).mean(), record['metrics']['macro_f1'], atol=1e-12, rtol=0)
                    predictions.append(pred)
                    sources[str(directory / 'COMPLETE.json')] = sha(directory / 'COMPLETE.json')
                    if system == 'CurrentRepeat-GRU':
                        history = read_json(directory / 'history.json')
                        if len(history) != 20 or any(set(r) != {'epoch', 'train_loss'} for r in history):
                            raise AssertionError('Formal training used evaluation for selection')
            sources[str(parent / 'targets.jsonl')] = sha(parent / 'targets.jsonl')
            sources[str(parent / 'predictions.csv')] = sha(parent / 'predictions.csv')
            predictions = np.stack(predictions)
            _, blocks = np.unique(expected.timestamp.to_numpy() // cfg['bootstrap']['block_seconds'], return_inverse=True)
            nblocks = blocks.max() + 1
            cm_time = group_cms(gt, predictions, blocks, nblocks)
            total = cm_time.sum(1)
            f1 = f1_from_cm(total).reshape(3, 3, 7)
            macro = f1.mean(2)
            scores.append(macro)
            class_scores.append(f1)
            draws = generator_time.multinomial(nblocks, np.full(nblocks, 1 / nblocks), size=replicates)
            values = f1_from_cm(np.einsum('rb,mbij->rmij', draws, cm_time)).mean(2).reshape(replicates, 3, 3).mean(2)
            bootstrap_time += values / 5
            if protocol == 'OS':
                unique_cows, cow_ids = np.unique(expected.cow_id.to_numpy(), return_inverse=True)
                if len(unique_cows) != 2:
                    raise AssertionError('Expected two held-out cows per OS fold')
                cm_cow = group_cms(gt, predictions, cow_ids, 2)
                draws_cow = generator_cow.multinomial(2, np.full(2, .5), size=replicates)
                values_cow = f1_from_cm(np.einsum('rb,mbij->rmij', draws_cow, cm_cow)).mean(2).reshape(replicates, 3, 3).mean(2)
                bootstrap_cow += values_cow / 5
                for index, cow in enumerate(unique_cows):
                    by_cow = f1_from_cm(cm_cow[:, index]).reshape(3, 3, 7)
                    cows.append(dict(cow_id=int(cow), fold=fold, targets=int((cow_ids == index).sum()),
                                     class_support=cm_cow[0, index].sum(1).tolist(),
                                     macro_by_system={s: float(by_cow[i].mean()) for i, s in enumerate(systems)}))
            fold_record = dict(fold=fold, targets=len(expected),
                               macro_by_system={s: float(macro[i].mean()) for i, s in enumerate(systems)})
            folds.append(fold_record)
            folds_table.append(dict(protocol=protocol, fold=fold, targets=len(expected), **fold_record['macro_by_system']))
            for i, system in enumerate(systems):
                for j, seed in enumerate(cfg['seeds']):
                    seed_table.append(dict(protocol=protocol, fold=fold, system=system, seed=seed, macro_f1=float(macro[i, j])))
        if len(target_pairs) != len(set(target_pairs)):
            raise AssertionError('Repeated Test targets across folds')
        scores = np.stack(scores)
        classes = np.stack(class_scores)
        means = scores.mean((0, 2))
        seed_means = scores.mean(0)
        comparisons = {}
        for index, system in enumerate(cfg['systems'], 1):
            comparisons[f'Base_minus_{system}'] = dict(delta=float(means[0] - means[index]),
                conditional_time_CI95=np.quantile(bootstrap_time[:, 0] - bootstrap_time[:, index], [.025, .975]).tolist(),
                conditional_cow_CI95=None if bootstrap_cow is None else
                    np.quantile(bootstrap_cow[:, 0] - bootstrap_cow[:, index], [.025, .975]).tolist(),
                per_seed_fivefold_delta=(seed_means[0] - seed_means[index]).tolist(),
                per_fold_three_seed_delta=(scores[:, 0].mean(1) - scores[:, index].mean(1)).tolist(),
                per_class_delta=(classes[:, 0].mean((0, 1)) - classes[:, index].mean((0, 1))).tolist())
        for i, system in enumerate(systems):
            for category in range(7):
                class_table.append(dict(protocol=protocol, system=system, class_id=category + 1,
                                        f1=float(classes[:, i, :, category].mean())))
        results[protocol] = dict(folds=folds, per_test_cow=cows, unique_TEST_targets=len(target_pairs),
            scores={s: dict(mean=float(means[i]), per_seed_fivefold_mean=seed_means[i].tolist(),
                           seed_population_std=float(seed_means[i].std()),
                           per_class_mean=classes[:, i].mean((0, 1)).tolist()) for i, s in enumerate(systems)},
            comparisons=comparisons)
    for path, digest in sources.items():
        if sha(path) != digest:
            raise AssertionError('Input changed during independent rescoring')
    output = dict(protocols=results, verified_prediction_scores=True, original_GT_verified=True,
                  historical_TEST_exposure=True, new_blind_evaluation=False, inference_replayed_by_author_runner=True,
                  interval_scope=cfg['interval_scope'], new_statistical_significance_claim=False,
                  mainline_replacement=False, independent_semantic_audit='pending', sources=sources)
    write_json(root / 'verified_summary.json', output)
    pd.DataFrame(folds_table).to_csv(root / 'fold_results.csv', index=False)
    pd.DataFrame(seed_table).to_csv(root / 'seed_results.csv', index=False)
    pd.DataFrame(class_table).to_csv(root / 'class_results.csv', index=False)
    lines = ['# History information controls', '',
             'Retrospective existing-benchmark supplement. Five folds and three seeds per system. Fixed seven classes.', '',
             '| Protocol | Base F1% | CurrentRepeat F1% | W-RF F1% |', '| --- | ---: | ---: | ---: |']
    for protocol, record in results.items():
        values = [record['scores'][s]['mean'] * 100 for s in systems]
        lines.append(f'| {protocol} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} |')
    lines += ['', '| Protocol | Comparison | Delta pp | Conditional time CI pp | Conditional cow CI pp |',
              '| --- | --- | ---: | --- | --- |']
    for protocol, record in results.items():
        for name, comparison in record['comparisons'].items():
            def interval(values):
                return 'N/A' if values is None else '[' + ', '.join(f'{v * 100:+.6f}' for v in values) + ']'
            lines.append(f"| {protocol} | {name} | {comparison['delta'] * 100:+.6f} | "
                         f"{interval(comparison['conditional_time_CI95'])} | {interval(comparison['conditional_cow_CI95'])} |")
    lines += ['', 'All saved prediction metrics were independently reconstructed from confusion counts and raw labels.',
              'These intervals condition on trained models and existing folds; they do not include retraining or model-selection uncertainty.',
              'CurrentRepeat matches parameters and unrolled steps but uses artificial repeated endpoint tokens. It does not identify biological causality.',
              'No new blind Test, all-simple-method superiority, equivalence, or new-farm claim. Fresh semantic audit pending.', '']
    (root / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print('HISTORY_CONTROLS_RESULTS_RESCORED', flush=True)


if __name__ == '__main__':
    main()
