"""Reuse the original paper RF checkpoints in the audited classic comparison."""
import copy
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import sklearn
from sklearn.ensemble import RandomForestClassifier

from q3_sensor.evaluation import metrics
from q3_sensor.protocol import sha
from scripts.sensor_os_protocol import RF_FEATURES, load_author_daily
from scripts.summarize_sensor_architecture_comparison_20261011 import f1, group_confusions
from scripts.run_sensor_history_controls_20261011 import read_json, write_json

CONFIG = Path('configs/sensor_classic_with_original_rf_20261014.json')


def verify_outputs(directory, complete):
    for name, digest in complete['outputs'].items():
        assert sha(directory / name) == digest, str(directory / name)


def author_frames(author, source, protocol, fold):
    official = read_json(f'configs/official_{protocol.lower()}.json')
    if protocol == 'TS':
        return author.data_loader_s2('uwb_hd_aa_pre_loader', str(source),
            copy.deepcopy(official['group_1']), copy.deepcopy(official['group_2']),
            copy.deepcopy(official['folds'][fold]), val=False, timestamp=True)
    return author.data_loader_s1('uwb_hd_aa_pre_loader', str(source),
        copy.deepcopy(official['folds'][fold]), val=False, timestamp=True)


def main():
    if Path.cwd() != Path('/q3/mmcows-sensor-behavior-20261007'):
        raise RuntimeError('Scientific computations are cloud-only')
    cfg = read_json(CONFIG)
    base = read_json(cfg['base_config'])
    environment = read_json('configs/sensor_classic_env_20261012.json')
    actual_versions = dict(python=platform.python_version(), torch=torch.__version__,
        numpy=np.__version__, sklearn=sklearn.__version__)
    for name, version in actual_versions.items():
        assert version == environment[name], f'Environment version changed: {name}={version}'
    assert cfg['new_fits'] == 0 and cfg['original_RF']['seeds_per_fold'] == [42]
    assert cfg['bootstrap'] == base['bootstrap']
    assert cfg['base_seeds_per_fold'] == base['formal_seeds']
    old = Path(cfg['base_results'])
    audit = read_json(cfg['base_audit'])
    assert audit['deterministic_status'] == 'PASS'
    prior_hashes = {str(Path(name).resolve()): digest.removeprefix('sha256:')
        for name, digest in audit['audited_input_hashes'].items()}
    old_summary = read_json(old / 'verified_summary.json')
    assert old_summary['original_GT_verified'] and old_summary['saved_metrics_reconstructed']
    identity = read_json(old / 'run_identity.json')
    for name, digest in identity['sources'].items():
        assert sha(name) == digest, name
    assert read_json(old / 'FORMAL_COMPLETE.json')['expected_jobs'] == 120
    root = Path(cfg['output'])
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg['threads'])
    torch.set_grad_enabled(False)
    # A reuse/rescore stage must never fit or update a model.
    def forbid_fit(*args, **kwargs):
        raise RuntimeError('No RF fitting is permitted in this reuse stage')
    RandomForestClassifier.fit = forbid_fit
    paths = {CONFIG, Path(__file__), Path(cfg['base_config']), Path(cfg['base_audit']),
        Path('configs/sensor_classic_env_20261012.json'),
        old / 'run_identity.json', old / 'METHOD_LOCK.json', old / 'FORMAL_COMPLETE.json',
        old / 'verified_summary.json', Path(base['reuse_architecture_root']) / 'run_identity.json',
        Path('scripts/sensor_os_protocol.py'), Path('q3_sensor/evaluation.py'),
        Path('scripts/summarize_sensor_architecture_comparison_20261011.py'),
        Path('reports/sensor_baselines_20261007/upstream/data_loader.py'),
        Path(base['source']) / 'source_receipt.json'}
    prior_required = {old / 'run_identity.json', old / 'METHOD_LOCK.json', old / 'verified_summary.json'}
    historical_RF_hashes = {}
    for protocol, evidence in cfg['original_RF']['historical_audits'].items():
        assert sha(evidence['path']) == evidence['sha256']
        historical = read_json(evidence['path'])[evidence['hash_field']]
        names = [str(Path(cfg['original_RF']['reuse_roots'][protocol]) / fold / 'RF.pt') for fold in base['folds']]
        names += ['reports/sensor_baselines_20261007/upstream/data_loader.py',
            'reports/sensor_baselines_20261007/upstream/rf_classifer.py']
        for name in names:
            assert name in historical, f'Missing historical RF evidence: {name}'
            digest = historical[name].removeprefix('sha256:')
            assert sha(name) == digest, f'Original RF historical input changed: {name}'
            historical_RF_hashes[name] = digest
            paths.add(Path(name))
        paths.add(Path(evidence['path']))
    paths.update(Path(name) for name in identity['sources'])
    raw_receipt = read_json(Path(base['source']) / 'source_receipt.json')
    for record in raw_receipt['members']:
        path = Path(base['source']) / Path(record['member']).relative_to('sensor_data')
        assert sha(path) == record['sha256']
        paths.add(path)
    for name in ('fold_results', 'seed_results', 'class_results', 'confusion_matrices', 'fit_costs'):
        paths.add(old / f'{name}.csv')
        prior_required.add(old / f'{name}.csv')
    jobs = {}
    for protocol in base['protocols']:
        paths.add(Path(f'configs/official_{protocol.lower()}.json'))
        prior_required.add(Path(f'configs/official_{protocol.lower()}.json'))
        paths.add(Path(cfg['original_RF']['reuse_roots'][protocol]) / 'run_identity.json')
        for fold in base['folds']:
            cache = Path(base['reuse_architecture_root']) / 'data/formal' / protocol / fold
            paths.update(cache / name for name in ('DATA_RECEIPT.json', 'targets.jsonl'))
            prior_required.update(cache / name for name in ('DATA_RECEIPT.json', 'targets.jsonl'))
            parent = Path(cfg['original_RF']['reuse_roots'][protocol]) / fold
            paths.update(parent / name for name in ('RF.pt', 'predictions.csv', 'targets.jsonl', 'COMPLETE.json'))
            for model in cfg['models'][1:]:
                for seed in base['formal_seeds']:
                    directory = (Path(base['reuse_architecture_root']) / 'formal' / protocol / fold / 'GRU' / f'seed{seed}'
                        if model == base['reference_model'] else old / 'formal' / protocol / fold / model / f'seed{seed}')
                    complete = read_json(directory / 'COMPLETE.json')
                    expected_identity = (base['reuse_architecture_identity_sha256']
                        if model == base['reference_model'] else sha(old / 'run_identity.json'))
                    assert complete['identity_sha256'] == expected_identity
                    verify_outputs(directory, complete)
                    jobs[(protocol, fold, model, seed)] = directory
                    paths.update(directory / name for name in complete['outputs'])
                    paths.add(directory / 'COMPLETE.json')
                    checkpoint_name = ('checkpoint.pt' if model in (base['reference_model'], 'CurrentRepeat-GRU-VAL')
                        else 'checkpoint.joblib')
                    prior_required.update(directory / name for name in ('COMPLETE.json', 'predictions.csv', 'result.json', checkpoint_name))
    before = {str(p): sha(p) for p in sorted(paths)}
    for path in prior_required:
        assert str(path.resolve()) in prior_hashes, f'Missing prior Test replay binding: {path}'
    prior_bound = 0
    for name, digest in before.items():
        prior_digest = prior_hashes.get(str(Path(name).resolve()))
        if prior_digest is not None:
            assert digest == prior_digest, f'Input changed since prior final audit: {name}'
            prior_bound += 1
    run_identity = dict(config=cfg, sources=before, base_identity_sha256=sha(old / 'run_identity.json'),
        base_audit_sha256=sha(cfg['base_audit']), fits_performed=0, environment=actual_versions)
    if (root / 'run_identity.json').exists():
        assert read_json(root / 'run_identity.json') == run_identity
    else:
        write_json(root / 'run_identity.json', run_identity)
    write_json(root / 'REUSE_IDENTITY_CHECK.json', dict(prior_audit_inputs_bound=prior_bound,
        required_prior_inputs=len(prior_required), missing_prior_inputs=[], changed_prior_inputs=[],
        historical_RF_checkpoints_bound=10, historical_RF_and_author_code_hashes=historical_RF_hashes))
    if (root / 'COMPLETE.json').exists():
        verify_outputs(root, read_json(root / 'COMPLETE.json'))
        print('SIX_MODEL_ALREADY_COMPLETE', flush=True)
        return
    write_json(root / 'PROGRESS.json', dict(stage='original_RF_replay_and_rescore', RF_completed=0, RF_expected=10))
    author, _ = load_author_daily(Path(base['source']))
    models = cfg['models']
    seeds = {model: ([42] if model == 'Original-Paper-RF' else base['formal_seeds']) for model in models}
    slices, at = {}, 0
    for model in models:
        slices[model] = slice(at, at + len(seeds[model]))
        at += len(seeds[model])
    outputs, fold_rows, seed_rows, class_rows, cm_rows, cost_rows, pair_rows = {}, [], [], [], [], [], []
    replay_receipts = []
    for protocol in base['protocols']:
        loaded, unique = [], []
        for fold in base['folds']:
            cache = Path(base['reuse_architecture_root']) / 'data/formal' / protocol / fold
            receipt = read_json(cache / 'DATA_RECEIPT.json')
            assert sha(cache / 'targets.jsonl') == receipt['outputs']['targets.jsonl']
            targets = pd.read_json(cache / 'targets.jsonl', lines=True, convert_dates=False)
            expected = targets.loc[targets.split.eq('test'), ['cow_id', 'timestamp', 'label']].reset_index(drop=True)
            for cow, records in expected.groupby('cow_id'):
                gt_path = Path(base['source']) / f'behavior_labels/individual/C{cow:02d}_0725.csv'
                gt = pd.read_csv(gt_path).set_index('timestamp').behavior
                assert gt.index.is_unique
                np.testing.assert_array_equal(records.label.to_numpy(), gt.loc[records.timestamp].to_numpy(dtype=int))
            parent = Path(cfg['original_RF']['reuse_roots'][protocol]) / fold
            parent_complete = read_json(parent / 'COMPLETE.json')
            assert sha(parent / 'targets.jsonl') == parent_complete['target_sha256']
            if 'checkpoint_sha256' in parent_complete:
                assert sha(parent / 'RF.pt') == parent_complete['checkpoint_sha256']['RF.pt']
            parent_targets = pd.read_json(parent / 'targets.jsonl', lines=True, convert_dates=False)
            pd.testing.assert_frame_equal(targets[['cow_id', 'timestamp', 'label', 'split']],
                parent_targets[['cow_id', 'timestamp', 'label', 'split']])
            parent_predictions = pd.read_csv(parent / 'predictions.csv')
            pd.testing.assert_frame_equal(parent_predictions[['cow_id', 'timestamp', 'label']], expected)
            train_frame, val_frame, test_frame = author_frames(author, Path(base['source']), protocol, fold)
            assert len(val_frame) == 0
            for frame, role in ((train_frame, 'train'), (test_frame, 'test')):
                keys = frame[['cow_id', 'timestamp', 'behavior']].copy()
                keys.columns = ['cow_id', 'timestamp', 'label']
                np.testing.assert_array_equal(keys.to_numpy(), keys.astype('int64').to_numpy())
                keys = keys.astype('int64').reset_index(drop=True)
                pd.testing.assert_frame_equal(keys, targets.loc[targets.split.eq(role), keys.columns].reset_index(drop=True))
            checkpoint = torch.load(parent / 'RF.pt', map_location='cpu', weights_only=False)
            assert checkpoint['features'] == RF_FEATURES == cfg['original_RF']['features']
            rf = checkpoint['model']
            params = rf.get_params()
            for name in ('n_estimators', 'class_weight', 'random_state'):
                assert params[name] == cfg['original_RF'][name]
            assert isinstance(rf, RandomForestClassifier) and len(rf.estimators_) == 100 and rf.n_features_in_ == 7
            np.testing.assert_array_equal(rf.classes_, np.arange(1, 8))
            rf.n_jobs = cfg['threads']
            started = time.monotonic()
            original_prediction = rf.predict(test_frame[RF_FEATURES].to_numpy()).astype(int)
            replay_seconds = time.monotonic() - started
            np.testing.assert_array_equal(original_prediction, parent_predictions.RF.to_numpy(dtype=int))
            original_metrics = metrics(expected.label.to_numpy(), original_prediction)
            for name, value in original_metrics.items():
                np.testing.assert_allclose(value, parent_complete['records']['RF'][name], rtol=0, atol=2e-15)
            normalized = expected.assign(prediction=original_prediction)
            reused = root / 'reused_original_RF' / protocol / fold / 'seed42'
            reused.mkdir(parents=True, exist_ok=True)
            normalized.to_csv(reused / 'predictions.csv', index=False)
            replay_receipts.append(dict(protocol=protocol, fold=fold, seed=42, checkpoint=str(parent / 'RF.pt'),
                checkpoint_sha256=sha(parent / 'RF.pt'), Test_targets=len(expected), prediction_mismatches=0,
                seconds=replay_seconds, RF_params={name: params[name] for name in ('n_estimators', 'class_weight', 'random_state')},
                metrics=original_metrics, input_features=RF_FEATURES, no_fit=True))
            write_json(reused / 'REPLAY.json', replay_receipts[-1])
            predictions = []
            for model in models:
                for seed in seeds[model]:
                    if model == 'Original-Paper-RF':
                        pred = original_prediction
                        cost_rows.append(dict(protocol=protocol, fold=fold, model=model, seed=seed, seconds=None,
                            replay_seconds=replay_seconds, cost_scope='original_fit_cost_unavailable_Test_replay_separate'))
                    else:
                        directory = jobs[(protocol, fold, model, seed)]
                        table = pd.read_csv(directory / 'predictions.csv')
                        pd.testing.assert_frame_equal(table[['cow_id', 'timestamp', 'label']], expected)
                        pred = table.prediction.to_numpy(dtype=int)
                        result = read_json(directory / 'result.json')
                        for name, value in metrics(expected.label.to_numpy(), pred).items():
                            np.testing.assert_allclose(value, result['metrics'][name], rtol=0, atol=2e-15)
                        cost_rows.append(dict(protocol=protocol, fold=fold, model=model, seed=seed, seconds=result['seconds'],
                            feature_seconds=result.get('feature_seconds'), classifier_fit_seconds=result.get('classifier_fit_seconds'),
                            replay_seconds=None, cost_scope='reused_full_fit_and_evaluation_record'))
                    assert np.isin(pred, np.arange(1, 8)).all()
                    predictions.append(pred)
            labels = expected.label.to_numpy(dtype=int)
            unique.extend(zip(expected.cow_id, expected.timestamp))
            loaded.append(dict(fold=fold, table=expected, labels=labels, predictions=np.stack(predictions),
                times=expected.timestamp.to_numpy() // cfg['bootstrap']['block_seconds']))
            write_json(root / 'PROGRESS.json', dict(stage='original_RF_replay_and_rescore', RF_completed=len(replay_receipts), RF_expected=10))
            print('ORIGINAL_RF_REPLAY_COMPLETE', protocol, fold, len(expected), flush=True)
        assert len(unique) == len(set(unique))
        blocks = np.unique(np.concatenate([row['times'] for row in loaded]))
        rng = np.random.default_rng(cfg['bootstrap']['seed'])
        draws = rng.multinomial(len(blocks), np.full(len(blocks), 1 / len(blocks)), size=cfg['bootstrap']['replicates'])
        cow_rng = np.random.default_rng(cfg['bootstrap']['seed'])
        bt = np.zeros((len(draws), len(models)))
        bc = np.zeros_like(bt) if protocol == 'OS' else None
        fold_scores, fold_classes = {m: [] for m in models}, {m: [] for m in models}
        for row in loaded:
            grouped = group_confusions(row['labels'], row['predictions'], np.searchsorted(blocks, row['times']), len(blocks))
            totals = grouped.sum(1)
            sampled_cm = np.einsum('rb,sbij->rsij', draws, grouped)
            assert (sampled_cm.sum((-1, -2)) > 0).all()
            sampled = f1(sampled_cm).mean(-1)
            if bc is not None:
                cows, cow_ids = np.unique(row['table'].cow_id, return_inverse=True)
                assert len(cows) == 2
                cow_cm = group_confusions(row['labels'], row['predictions'], cow_ids, 2)
                cow_draws = cow_rng.multinomial(2, np.full(2, .5), size=len(draws))
                cow_sampled = f1(np.einsum('rb,sbij->rsij', cow_draws, cow_cm)).mean(-1)
            for index, model in enumerate(models):
                cls = f1(totals[slices[model]])
                macro = cls.mean(-1)
                fold_scores[model].append(macro)
                fold_classes[model].append(cls)
                bt[:, index] += sampled[:, slices[model]].mean(1) / 5
                if bc is not None:
                    bc[:, index] += cow_sampled[:, slices[model]].mean(1) / 5
                fold_rows.append(dict(protocol=protocol, fold=row['fold'], model=model, macro_f1=float(macro.mean()),
                    targets=len(row['labels']), seed_count=len(seeds[model])))
                for offset, seed in enumerate(seeds[model]):
                    seed_rows.append(dict(protocol=protocol, fold=row['fold'], model=model, seed=seed, macro_f1=float(macro[offset])))
                    cm = totals[slices[model]][offset]
                    for truth in range(7):
                        for prediction in range(7):
                            cm_rows.append(dict(protocol=protocol, fold=row['fold'], model=model, seed=seed,
                                true_class=truth + 1, predicted_class=prediction + 1, count=int(cm[truth, prediction])))
        scores = {}
        for model in models:
            values = np.stack(fold_scores[model])
            classes = np.stack(fold_classes[model]).mean((0, 1))
            scores[model] = dict(mean=float(values.mean()), per_seed_fivefold_mean=values.mean(0).tolist(),
                seed_population_std=float(values.mean(0).std()) if len(seeds[model]) > 1 else None,
                seeds=seeds[model], per_class_mean=classes.tolist())
            class_rows.extend(dict(protocol=protocol, model=model, class_id=k + 1, f1=float(value)) for k, value in enumerate(classes))
            if model != 'Original-Paper-RF':
                np.testing.assert_allclose(scores[model]['mean'], old_summary['protocols'][protocol]['scores'][model]['mean'], rtol=0, atol=2e-15)
        comparisons = {}
        for left, right in cfg['primary_comparisons']:
            li, ri = models.index(left), models.index(right)
            key = f'{left}_minus_{right}'
            record = dict(delta=scores[left]['mean'] - scores[right]['mean'],
                conditional_time_CI95=np.quantile(bt[:, li] - bt[:, ri], [.025, .975]).tolist(),
                conditional_cow_CI95=None if bc is None else np.quantile(bc[:, li] - bc[:, ri], [.025, .975]).tolist(),
                per_fold_delta=(np.stack(fold_scores[left]).mean(1) - np.stack(fold_scores[right]).mean(1)).tolist())
            comparisons[key] = record
            if key in old_summary['protocols'][protocol]['comparisons']:
                previous = old_summary['protocols'][protocol]['comparisons'][key]
                for name in ('delta', 'conditional_time_CI95', 'per_fold_delta'):
                    np.testing.assert_allclose(record[name], previous[name], rtol=0, atol=2e-15)
                if bc is not None:
                    np.testing.assert_allclose(record['conditional_cow_CI95'], previous['conditional_cow_CI95'], rtol=0, atol=2e-15)
            pair_rows.append(dict(protocol=protocol, left=left, right=right, delta=record['delta'],
                time_CI95_low=record['conditional_time_CI95'][0], time_CI95_high=record['conditional_time_CI95'][1],
                cow_CI95_low=None if bc is None else record['conditional_cow_CI95'][0],
                cow_CI95_high=None if bc is None else record['conditional_cow_CI95'][1]))
        outputs[protocol] = dict(scores=scores, comparisons=comparisons, unique_Test_targets=len(unique), shared_time_blocks=len(blocks))
    assert len(replay_receipts) == 10 and len(seed_rows) == 160
    after = {name: sha(name) for name in before}
    assert before == after, 'Inputs changed during six-model evaluation'
    write_json(root / 'input_hashes_after.json', after)
    write_json(root / 'original_RF_replay.json', dict(models=10, prediction_mismatches=0, receipts=replay_receipts))
    write_json(root / 'verified_summary.json', dict(protocols=outputs, models=models, sources=before,
        fits_performed=0, original_RF_actual_checkpoint_replay=True, actual_formal_models=160,
        base_150_checkpoint_replay_audit=cfg['base_audit'], independent_audit='pending',
        historical_Test_exposure=True, addition_after_Test_exposure=True, new_blind_evaluation=False,
        multiple_comparison_correction=False, mainline_automatically_replaced=False))
    for name, rows in [('fold_results', fold_rows), ('seed_results', seed_rows), ('class_results', class_rows),
        ('confusion_matrices', cm_rows), ('fit_costs', cost_rows), ('paired_comparisons', pair_rows)]:
        pd.DataFrame(rows).to_csv(root / f'{name}.csv', index=False)
    lines = ['# 原论文RF正式接入六模型对比', '',
        '官方TS/OS各五折；RF每折seed42，其余每折seed17/23/29。各模型先平均自身seed再平均五折。', '',
        '| 模型 | TS宏F1% | OS宏F1% | 每折实际seed数 |', '| --- | ---: | ---: | ---: |']
    for model in models:
        lines.append(f"| {model} | {outputs['TS']['scores'][model]['mean']*100:.6f} | {outputs['OS']['scores'][model]['mean']*100:.6f} | {len(seeds[model])} |")
    lines += ['', '原论文发布值TS73.700%/OS71.400%仅作参考，非第七个实测模型。',
        '原RF与新模型的输入/历史表示不同，属于完整方法比较，不能只归因于GRU结构。',
        'RF对比在Test曝光后追加；区间条件于已有数据/配方/模型，未校正多重比较，不代表新盲测。',
        '原RF旧拟合耗时未知，重放耗时另列；本轮VAL-refit锚点与原20轮主线分开。',
        '新增RF及六模型汇总独立审计待完成。', '']
    (root / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.name != 'COMPLETE.json')
    write_json(root / 'COMPLETE.json', dict(status='RF_replay_and_six_model_rescore_complete',
        RF_models=10, actual_formal_models=160, new_fits=0, independent_audit='pending',
        identity_sha256=sha(root / 'run_identity.json'), outputs={str(p.relative_to(root)): sha(p) for p in files}))
    print('SIX_MODEL_WITH_ORIGINAL_RF_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
