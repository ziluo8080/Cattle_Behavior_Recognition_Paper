"""Raw-GT prediction reconstruction and synchronized conditional comparisons."""
from pathlib import Path

import numpy as np
import pandas as pd

from q3_sensor.protocol import sha
from scripts.run_sensor_architecture_comparison_20261011 import setup
from scripts.run_sensor_history_controls_20261011 import read_json, write_json


def f1(cm):
    positive = np.diagonal(cm, axis1=-2, axis2=-1)
    denominator = cm.sum(-1) + cm.sum(-2)
    return np.divide(2 * positive, denominator, out=np.zeros_like(positive, dtype=float), where=denominator != 0)


def group_confusions(labels, predictions, groups, count):
    return np.stack([np.bincount(groups * 49 + (labels - 1) * 7 + pred - 1,
                                minlength=count * 49).reshape(count, 7, 7) for pred in predictions])


def main():
    cfg, root = setup()
    if read_json(root / 'FORMAL_COMPLETE.json')['identity_sha256'] != sha(root / 'run_identity.json'):
        raise AssertionError('Formal batch incomplete')
    lock = read_json(root / 'METHOD_LOCK.json')
    raw_gt = {}
    sources = {str(Path(__file__)): sha(__file__), str(root / 'METHOD_LOCK.json'): sha(root / 'METHOD_LOCK.json'),
               str(root / 'run_identity.json'): sha(root / 'run_identity.json')}
    for cow in range(1, 11):
        path = Path(cfg['source']) / f'behavior_labels/individual/C{cow:02d}_0725.csv'
        values = pd.read_csv(path)
        if values.timestamp.duplicated().any():
            raise AssertionError('Repeated raw label time')
        raw_gt[cow] = values.set_index('timestamp').behavior
        sources[str(path)] = sha(path)
    fold_rows, seed_rows, class_rows, output = [], [], [], {}
    for protocol in cfg['protocols']:
        loaded = []
        all_times = []
        unique_targets = []
        for fold in cfg['folds']:
            data_directory = root / 'data/formal' / protocol / fold
            targets = pd.read_json(data_directory / 'targets.jsonl', lines=True, convert_dates=False)
            expected = targets.loc[targets.split.eq('test'), ['cow_id', 'timestamp', 'label']].reset_index(drop=True)
            for cow, records in expected.groupby('cow_id'):
                np.testing.assert_array_equal(raw_gt[int(cow)].loc[records.timestamp].to_numpy(dtype=int), records.label.to_numpy())
            unique_targets.extend(zip(expected.cow_id, expected.timestamp))
            labels = expected.label.to_numpy(dtype=int)
            predictions = []
            for kind in cfg['models']:
                recipe = lock['selected'][f'{protocol}/{fold}/{kind}']
                for seed in cfg['formal_seeds']:
                    directory = root / 'formal' / protocol / fold / kind / f'seed{seed}'
                    complete = read_json(directory / 'COMPLETE.json')
                    if complete['identity_sha256'] != sha(root / 'run_identity.json'):
                        raise AssertionError('Job identity changed')
                    for name, digest in complete['outputs'].items():
                        if sha(directory / name) != digest:
                            raise AssertionError('Saved job changed')
                    for name, digest in complete['data_identity']['outputs'].items():
                        if sha(data_directory / name) != digest:
                            raise AssertionError('Fit data changed')
                    table = pd.read_csv(directory / 'predictions.csv')
                    pd.testing.assert_frame_equal(table[['cow_id', 'timestamp', 'label']], expected)
                    pred = table.prediction.to_numpy(dtype=int)
                    if not np.isin(pred, range(1, 8)).all():
                        raise AssertionError('Invalid class')
                    matrix = np.bincount((labels - 1) * 7 + pred - 1, minlength=49).reshape(7, 7)
                    result = read_json(directory / 'result.json')
                    np.testing.assert_array_equal(matrix, result['metrics']['confusion_matrix'])
                    np.testing.assert_allclose(f1(matrix), result['metrics']['f1'], atol=1e-12, rtol=0)
                    history = read_json(directory / 'history.json')
                    if len(history) != recipe['epochs'] or result['learning_rate'] != recipe['learning_rate']:
                        raise AssertionError('Final budget differs from lock')
                    if any('val_macro_f1' in row or 'test_macro_f1' in row for row in history):
                        raise AssertionError('Formal refit used evaluation for selection')
                    predictions.append(pred)
                    sources[str(directory / 'COMPLETE.json')] = sha(directory / 'COMPLETE.json')
            parent = Path(cfg['parent_roots'][protocol]) / fold
            legacy = pd.read_csv(parent / 'predictions.csv')
            pd.testing.assert_frame_equal(legacy[['cow_id', 'timestamp', 'label']], expected)
            legacy_cm = np.stack([np.bincount((labels - 1) * 7 + legacy[f'NN_seed{s}'].to_numpy(dtype=int) - 1,
                                            minlength=49).reshape(7, 7) for s in cfg['formal_seeds']])
            times = expected.timestamp.to_numpy() // cfg['bootstrap']['block_seconds']
            all_times.append(times)
            loaded.append(dict(fold=fold, table=expected, labels=labels, predictions=np.stack(predictions),
                               block_times=times, legacy_f1=f1(legacy_cm)))
            sources[str(parent / 'predictions.csv')] = sha(parent / 'predictions.csv')
            sources[str(data_directory / 'targets.jsonl')] = sha(data_directory / 'targets.jsonl')
        if len(unique_targets) != len(set(unique_targets)):
            raise AssertionError('Duplicate Test targets across folds')
        blocks = np.unique(np.concatenate(all_times))
        replicates = cfg['bootstrap']['replicates']
        generator = np.random.default_rng(cfg['bootstrap']['seed'])
        shared_draws = generator.multinomial(len(blocks), np.full(len(blocks), 1 / len(blocks)), size=replicates)
        cow_generator = np.random.default_rng(cfg['bootstrap']['seed'])
        bootstrap_time = np.zeros((replicates, 4))
        bootstrap_cow = np.zeros((replicates, 4)) if protocol == 'OS' else None
        scores, classes, legacy_scores, legacy_classes = [], [], [], []
        per_cow = []
        for data in loaded:
            ids = np.searchsorted(blocks, data['block_times'])
            matrices = group_confusions(data['labels'], data['predictions'], ids, len(blocks))
            total = matrices.sum(1)
            values = f1(total).reshape(4, 3, 7)
            macro = values.mean(2)
            scores.append(macro)
            classes.append(values)
            legacy_scores.append(data['legacy_f1'].mean(1))
            legacy_classes.append(data['legacy_f1'])
            sampled = np.einsum('rb,mbij->rmij', shared_draws, matrices)
            if (sampled.sum((-1, -2)) == 0).any():
                raise AssertionError('A synchronized draw produced an empty fold; inference needs explicit revision')
            bootstrap_time += f1(sampled).mean(2).reshape(replicates, 4, 3).mean(2) / 5
            if protocol == 'OS':
                cows, cow_ids = np.unique(data['table'].cow_id.to_numpy(), return_inverse=True)
                if len(cows) != 2:
                    raise AssertionError('Expected two Test cows per OS fold')
                cow_cm = group_confusions(data['labels'], data['predictions'], cow_ids, 2)
                draws = cow_generator.multinomial(2, np.full(2, .5), size=replicates)
                bootstrap_cow += f1(np.einsum('rb,mbij->rmij', draws, cow_cm)).mean(2).reshape(replicates, 4, 3).mean(2) / 5
                for index, cow in enumerate(cows):
                    values_cow = f1(cow_cm[:, index]).reshape(4, 3, 7)
                    per_cow.append(dict(cow_id=int(cow), fold=data['fold'], support=cow_cm[0, index].sum(1).tolist(),
                        macro_by_model={kind: float(values_cow[i].mean()) for i, kind in enumerate(cfg['models'])}))
            for i, kind in enumerate(cfg['models']):
                fold_rows.append(dict(protocol=protocol, fold=data['fold'], model=kind, targets=len(data['labels']),
                    macro_f1=float(macro[i].mean()), selected_recipe=lock['selected'][f"{protocol}/{data['fold']}/{kind}"]))
                for j, seed in enumerate(cfg['formal_seeds']):
                    seed_rows.append(dict(protocol=protocol, fold=data['fold'], model=kind, seed=seed, macro_f1=float(macro[i, j])))
        scores, classes = np.stack(scores), np.stack(classes)
        mean = scores.mean((0, 2))
        seed_mean = scores.mean(0)
        comparisons = {}
        for left, right in cfg['primary_comparisons']:
            li, ri = cfg['models'].index(left), cfg['models'].index(right)
            comparisons[f'{left}_minus_{right}'] = dict(delta=float(mean[li] - mean[ri]),
                conditional_time_CI95=np.quantile(bootstrap_time[:, li] - bootstrap_time[:, ri], [.025, .975]).tolist(),
                conditional_cow_CI95=None if bootstrap_cow is None else np.quantile(bootstrap_cow[:, li] - bootstrap_cow[:, ri], [.025, .975]).tolist(),
                per_seed_fivefold_delta=(seed_mean[li] - seed_mean[ri]).tolist(),
                per_fold_three_seed_delta=(scores[:, li].mean(1) - scores[:, ri].mean(1)).tolist(),
                per_class_delta=(classes[:, li].mean((0, 1)) - classes[:, ri].mean((0, 1))).tolist())
        for i, kind in enumerate(cfg['models']):
            for category in range(7):
                class_rows.append(dict(protocol=protocol, model=kind, class_id=category + 1,
                                       f1=float(classes[:, i, :, category].mean())))
        output[protocol] = dict(scores={kind: dict(mean=float(mean[i]), per_seed_fivefold_mean=seed_mean[i].tolist(),
            seed_population_std=float(seed_mean[i].std()), per_class_mean=classes[:, i].mean((0, 1)).tolist())
            for i, kind in enumerate(cfg['models'])}, comparisons=comparisons, per_test_cow=per_cow,
            unique_Test_targets=len(unique_targets), shared_time_blocks=len(blocks),
            legacy_frozen_20_epoch_GRU_mean=float(np.mean(legacy_scores)),
            legacy_frozen_GRU_class_mean=np.mean(legacy_classes, axis=(0, 1)).tolist())
    for path, digest in sources.items():
        if sha(path) != digest:
            raise AssertionError('Source changed while rescoring')
    summary = dict(protocols=output, original_GT_verified=True, saved_metrics_reconstructed=True,
        primary_comparisons=cfg['primary_comparisons'], historical_Test_exposure=True, new_blind_evaluation=False,
        interval_scope=cfg['interval_scope'], synchronized_time_blocks_across_folds=True,
        statistical_significance_claim=False, multiple_comparison_correction=False,
        semantic_audit='pending', mainline_automatically_replaced=False, sources=sources)
    write_json(root / 'verified_summary.json', summary)
    pd.DataFrame(fold_rows).to_csv(root / 'fold_results.csv', index=False)
    pd.DataFrame(seed_rows).to_csv(root / 'seed_results.csv', index=False)
    pd.DataFrame(class_rows).to_csv(root / 'class_results.csv', index=False)
    lines = ['# Matched sequence architecture comparison', '',
        'Official TS/OS five folds, three refit seeds, validation-selected recipes. Retrospective benchmark.', '',
        '| Model | TS macro F1 % | OS macro F1 % | Parameters |', '| --- | ---: | ---: | ---: |']
    for kind in cfg['models']:
        lines.append(f"| {kind} | {output['TS']['scores'][kind]['mean'] * 100:.6f} | {output['OS']['scores'][kind]['mean'] * 100:.6f} | {cfg['expected_parameters'][kind]} |")
    lines += ['', 'Legacy frozen 20-epoch GRU remains a separate reference, not a same-budget selected model.', '',
        '| Protocol | Comparison | Delta pp | Conditional time CI pp | Conditional cow CI pp |',
        '| --- | --- | ---: | --- | --- |']
    for protocol, result in output.items():
        for name, comparison in result['comparisons'].items():
            def interval(values):
                return 'N/A' if values is None else '[' + ', '.join(f'{v * 100:+.6f}' for v in values) + ']'
            lines.append(f"| {protocol} | {name} | {comparison['delta'] * 100:+.6f} | {interval(comparison['conditional_time_CI95'])} | {interval(comparison['conditional_cow_CI95'])} |")
    lines += ['', 'All five contrasts were specified before training. Intervals are descriptive and not corrected for multiplicity.',
        'Time-block weights are synchronized across folds; OS cow resampling stays within the original two-cow fold strata.',
        'Intervals condition on fitted models, selected recipes and existing data. They do not cover selection, retraining or new environments.',
        'No new blind evaluation or automatic mainline replacement. Fresh integrity audit pending.', '']
    (root / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print('ARCHITECTURE_RESULTS_RECONSTRUCTED', flush=True)


if __name__ == '__main__':
    main()
