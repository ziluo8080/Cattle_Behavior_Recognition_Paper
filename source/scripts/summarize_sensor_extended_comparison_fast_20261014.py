"""One ordered-target overview of nine reused and two new models."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from q3_sensor.evaluation import metrics
from q3_sensor.protocol import sha
from scripts.run_sensor_history_controls_20261011 import read_json, write_json
from scripts.summarize_sensor_architecture_comparison_20261011 import f1, group_confusions

CFG = Path('configs/sensor_xgboost_tcn_fast_20261014.json')
SIX = Path('reports/sensor_classic_with_original_rf_20261014_attempt2')
CLASSIC = Path('reports/sensor_classic_comparison_20261012')
ARCH = Path('reports/sensor_architecture_comparison_20261011')
SIX_AUDIT = Path('reports/sensor_classic_with_original_rf_audit_20261014/EXPERIMENT_AUDIT.json')
ARCH_AUDIT = Path('reports/sensor_architecture_integrity_20261011/FINAL_AUDIT_REPORT.json')
EXISTING = ['Original-Paper-RF', 'E-GRU-SQRT-VAL', 'CurrentRepeat-GRU-VAL', 'Stats-RF-Window',
    'Stats-RF-CurrentOnly', 'MiniRocket-Window', 'LSTM', 'Transformer', 'Transformer-LSTM']


def directory_for(model, protocol, fold, seed, new_root):
    if model == 'Original-Paper-RF':
        return SIX / 'reused_original_RF' / protocol / fold / 'seed42'
    if model in ('E-GRU-SQRT-VAL', 'LSTM', 'Transformer', 'Transformer-LSTM'):
        return ARCH / 'formal' / protocol / fold / ('GRU' if model == 'E-GRU-SQRT-VAL' else model) / f'seed{seed}'
    return (new_root if model in ('XGBoost-Window', 'TCN-Window-VAL') else CLASSIC) / 'formal' / protocol / fold / model / f'seed{seed}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--existing-only', action='store_true')
    args = parser.parse_args()
    if Path.cwd() != Path('/q3/mmcows-sensor-behavior-20261007'):
        raise RuntimeError('All scientific score aggregation is cloud-only')
    cfg = read_json(CFG)
    new_root = Path(cfg['output'])
    models = EXISTING if args.existing_only else EXISTING + cfg['models']
    output = Path('reports/sensor_extended_model_overview_20261014') / ('existing9' if args.existing_only else 'all11_fast')
    output.mkdir(parents=True, exist_ok=True)
    if not args.existing_only:
        assert read_json(new_root / 'FORMAL_COMPLETE.json')['expected_jobs'] == 60
    assert read_json(SIX_AUDIT)['deterministic_status'] == 'PASS'
    assert read_json(ARCH_AUDIT)['verdict'] in ('PASS', 'WARN')
    numerical = Path('reports/sensor_architecture_integrity_20261011/FINAL_NUMERICAL_AUDIT.json')
    frozen = read_json('reports/sensor_architecture_integrity_20261011/FINAL_AUDIT_FREEZE.json')
    assert sha(numerical) == frozen['files'][str(numerical)]
    assert not read_json(numerical)['failed_checks']
    proofs = {}
    for audit in (SIX_AUDIT, ARCH_AUDIT):
        for path, digest in read_json(audit)['audited_input_hashes'].items():
            key = str(Path(path).resolve())
            value = digest.removeprefix('sha256:')
            if key in proofs:
                assert proofs[key] == value
            proofs[key] = value
    sources = {str(p): sha(p) for p in (CFG, Path(__file__), SIX_AUDIT, ARCH_AUDIT, SIX / 'COMPLETE.json')}
    six_complete = read_json(SIX / 'COMPLETE.json')
    for name, digest in six_complete['outputs'].items():
        assert sha(SIX / name) == digest
    prior_six = read_json(SIX / 'verified_summary.json')
    prior_arch = read_json(ARCH / 'verified_summary.json')
    seeds = {m: ([42] if m == 'Original-Paper-RF' else cfg['formal_seeds']) for m in models}
    model_slices, offset = {}, 0
    for model in models:
        model_slices[model] = slice(offset, offset + len(seeds[model]))
        offset += len(seeds[model])
    results, fold_rows, seed_rows, class_rows, cm_rows, pair_rows = {}, [], [], [], [], []
    for protocol in cfg['protocols']:
        rows, unique = [], []
        for fold in cfg['folds']:
            cache = ARCH / 'data/formal' / protocol / fold
            receipt = read_json(cache / 'DATA_RECEIPT.json')
            for p in (cache / 'DATA_RECEIPT.json', cache / 'targets.jsonl'):
                assert str(p.resolve()) in proofs and sha(p) == proofs[str(p.resolve())]
                sources[str(p)] = sha(p)
            assert sha(cache / 'targets.jsonl') == receipt['outputs']['targets.jsonl']
            targets = pd.read_json(cache / 'targets.jsonl', lines=True, convert_dates=False)
            expected = targets.loc[targets.split.eq('test'), ['cow_id', 'timestamp', 'label']].reset_index(drop=True)
            for cow, group in expected.groupby('cow_id'):
                path = Path(cfg['source']) / f'behavior_labels/individual/C{cow:02d}_0725.csv'
                raw = pd.read_csv(path).set_index('timestamp').behavior
                assert raw.index.is_unique
                np.testing.assert_array_equal(group.label.to_numpy(), raw.loc[group.timestamp].to_numpy(dtype=int))
                sources[str(path)] = sha(path)
            predictions = []
            for model in models:
                for seed in seeds[model]:
                    directory = directory_for(model, protocol, fold, seed, new_root)
                    if model == 'Original-Paper-RF':
                        result = read_json(directory / 'REPLAY.json')
                        score = result['metrics']
                        files = (directory / 'REPLAY.json', directory / 'predictions.csv')
                    else:
                        complete = read_json(directory / 'COMPLETE.json')
                        identity_root = ARCH if model in ('E-GRU-SQRT-VAL', 'LSTM', 'Transformer', 'Transformer-LSTM') else (
                            new_root if model in cfg['models'] else CLASSIC)
                        assert complete['identity_sha256'] == sha(identity_root / 'run_identity.json')
                        for name, digest in complete['outputs'].items():
                            assert sha(directory / name) == digest
                        result = read_json(directory / 'result.json')
                        score = result['metrics']
                        files = (directory / 'COMPLETE.json', directory / 'result.json', directory / 'predictions.csv')
                    for p in files:
                        if model not in cfg['models']:
                            assert str(p.resolve()) in proofs and sha(p) == proofs[str(p.resolve())], str(p)
                        sources[str(p)] = sha(p)
                    table = pd.read_csv(directory / 'predictions.csv')
                    pd.testing.assert_frame_equal(table[['cow_id', 'timestamp', 'label']], expected)
                    pred = table.prediction.to_numpy(dtype=int)
                    assert np.isin(pred, np.arange(1, 8)).all()
                    for name, value in metrics(expected.label.to_numpy(dtype=int), pred).items():
                        np.testing.assert_allclose(value, score[name], rtol=0, atol=2e-15)
                    predictions.append(pred)
            unique.extend(zip(expected.cow_id, expected.timestamp))
            rows.append(dict(fold=fold, table=expected, labels=expected.label.to_numpy(dtype=int),
                predictions=np.stack(predictions), times=expected.timestamp.to_numpy() // 300))
        assert len(unique) == len(set(unique))
        blocks = np.unique(np.concatenate([r['times'] for r in rows]))
        rng = np.random.default_rng(20261014)
        draws = rng.multinomial(len(blocks), np.full(len(blocks), 1 / len(blocks)), size=2000)
        cow_rng = np.random.default_rng(20261014)
        time_samples = np.zeros((2000, len(models)))
        cow_samples = np.zeros_like(time_samples) if protocol == 'OS' else None
        fold_scores, fold_classes = {m: [] for m in models}, {m: [] for m in models}
        for row in rows:
            grouped = group_confusions(row['labels'], row['predictions'], np.searchsorted(blocks, row['times']), len(blocks))
            totals = grouped.sum(1)
            sampled_cm = np.einsum('rb,sbij->rsij', draws, grouped)
            assert (sampled_cm.sum((-1, -2)) > 0).all()
            sampled = f1(sampled_cm).mean(-1)
            if cow_samples is not None:
                cows, ids = np.unique(row['table'].cow_id, return_inverse=True)
                assert len(cows) == 2
                cow_cm = group_confusions(row['labels'], row['predictions'], ids, 2)
                cow_draws = cow_rng.multinomial(2, np.full(2, .5), size=2000)
                sampled_cow = f1(np.einsum('rb,sbij->rsij', cow_draws, cow_cm)).mean(-1)
            for index, model in enumerate(models):
                cls = f1(totals[model_slices[model]])
                values = cls.mean(-1)
                fold_scores[model].append(values)
                fold_classes[model].append(cls)
                time_samples[:, index] += sampled[:, model_slices[model]].mean(1) / 5
                if cow_samples is not None:
                    cow_samples[:, index] += sampled_cow[:, model_slices[model]].mean(1) / 5
                fold_rows.append(dict(protocol=protocol, fold=row['fold'], model=model,
                    macro_f1=float(values.mean()), seed_count=len(seeds[model]), targets=len(row['labels'])))
                for offset, seed in enumerate(seeds[model]):
                    seed_rows.append(dict(protocol=protocol, fold=row['fold'], model=model, seed=seed, macro_f1=float(values[offset])))
                    cm = totals[model_slices[model]][offset]
                    cm_rows.extend(dict(protocol=protocol, fold=row['fold'], model=model, seed=seed,
                        true_class=i + 1, predicted_class=j + 1, count=int(cm[i, j])) for i in range(7) for j in range(7))
        scores = {}
        for model in models:
            values = np.stack(fold_scores[model])
            classes = np.stack(fold_classes[model]).mean((0, 1))
            scores[model] = dict(mean=float(values.mean()), per_seed_fivefold_mean=values.mean(0).tolist(),
                per_class_mean=classes.tolist(), seeds=seeds[model],
                seed_population_std=float(values.mean(0).std()) if len(seeds[model]) > 1 else None)
            expected_scores = prior_six['protocols'][protocol]['scores'].get(model) or prior_arch['protocols'][protocol]['scores'].get(model)
            if expected_scores is not None:
                np.testing.assert_allclose(scores[model]['mean'], expected_scores['mean'], rtol=0, atol=2e-15)
            class_rows.extend(dict(protocol=protocol, model=model, class_id=i + 1, f1=float(value)) for i, value in enumerate(classes))
        pairs = [(m, 'Original-Paper-RF') for m in models if m != 'Original-Paper-RF']
        pairs += [(m, 'E-GRU-SQRT-VAL') for m in models if m not in ('Original-Paper-RF', 'E-GRU-SQRT-VAL')]
        pairs += [('Stats-RF-Window', 'Stats-RF-CurrentOnly')]
        comparisons = {}
        for left, right in pairs:
            li, ri = models.index(left), models.index(right)
            entry = dict(delta=scores[left]['mean'] - scores[right]['mean'],
                conditional_time_CI95=np.quantile(time_samples[:, li] - time_samples[:, ri], [.025, .975]).tolist(),
                conditional_cow_CI95=None if cow_samples is None else np.quantile(cow_samples[:, li] - cow_samples[:, ri], [.025, .975]).tolist(),
                per_fold_delta=(np.stack(fold_scores[left]).mean(1) - np.stack(fold_scores[right]).mean(1)).tolist())
            comparisons[f'{left}_minus_{right}'] = entry
            pair_rows.append(dict(protocol=protocol, left=left, right=right, delta=entry['delta'],
                time_low=entry['conditional_time_CI95'][0], time_high=entry['conditional_time_CI95'][1],
                cow_low=None if cow_samples is None else entry['conditional_cow_CI95'][0],
                cow_high=None if cow_samples is None else entry['conditional_cow_CI95'][1]))
        results[protocol] = dict(scores=scores, comparisons=comparisons, unique_Test_targets=len(unique), shared_time_blocks=len(blocks))
    for path, digest in sources.items():
        assert sha(path) == digest, path
    write_json(output / 'verified_summary.json', dict(protocols=results, models=models, sources=sources,
        actual_formal_models=len(seed_rows), new_blind_evaluation=False, historical_Test_exposure=True,
        interval_seed=20261014, replicates=2000, multiple_comparison_correction=False,
        independent_new_results_audit='pending', mainline_automatically_replaced=False))
    for name, data in [('fold_results', fold_rows), ('seed_results', seed_rows), ('class_results', class_rows),
        ('confusion_matrices', cm_rows), ('paired_comparisons', pair_rows)]:
        pd.DataFrame(data).to_csv(output / f'{name}.csv', index=False)
    text = ['# 已有九模型总览' if args.existing_only else '# 十一模型完整比较', '',
        '每模型先平均自身实际seed再五折等权；原RF单seed，其余三seed。', '',
        '| 模型 | TS宏F1% | OS宏F1% |', '| --- | ---: | ---: |']
    for model in models:
        text.append(f"| {model} | {results['TS']['scores'][model]['mean'] * 100:.6f} | {results['OS']['scores'][model]['mean'] * 100:.6f} |")
    text += ['', '本轮VAL-refit配方与原20轮主线分开；已有Test曝光，条件区间不包含重训/选型/新环境，未校正多重比较。',
        '本总览用seed20261014重新统一计算区间；旧报告各自的seed及区间保持原件，不冒称两套bootstrap输出逐位相同。',
        '既有审计证据引用与新结果审计范围分列，主线不自动替换。', '']
    (output / 'RESULTS.md').write_text('\n'.join(text), encoding='utf-8')
    write_json(output / 'COMPLETE.json', dict(status='saved_prediction_rescore_complete', models=len(models),
        actual_formal_models=len(seed_rows), comparisons=len(pair_rows), new_fits=0,
        outputs={p.name: sha(p) for p in output.iterdir() if p.is_file() and p.name != 'COMPLETE.json'}))
    print('EXTENDED_MODEL_OVERVIEW_COMPLETE', len(models), len(seed_rows), flush=True)


if __name__ == '__main__':
    main()
