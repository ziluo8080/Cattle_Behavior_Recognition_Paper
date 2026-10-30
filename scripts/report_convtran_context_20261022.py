"""Saved TRAIN/VAL evidence only: independent NumPy recount and paired report."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

CLASSES = ('walking', 'standing', 'feeding_up', 'feeding_down', 'licking', 'drinking', 'lying')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def scores(cm):
    cm = np.asarray(cm, dtype=np.int64)
    assert cm.shape == (7, 7) and (cm >= 0).all()
    denominator = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * cm.diagonal(), denominator, out=np.zeros(7, float), where=denominator > 0)
    return f1, float(f1.mean())


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def report(out, cfg):
    out = Path(out); results = {}; curves = []; finals = []; comparisons = {}; epoch_recounts = prediction_recounts = 0
    for protocol in cfg['protocols']:
        rows = [json.loads(line) for line in (out / 'data' / protocol / 'targets.jsonl').read_text().splitlines()]
        assert {r['split'] for r in rows} == {'train', 'val'}
        truth = np.asarray([r['label'] for r in rows if r['split'] == 'val'])
        assert np.isin(truth, range(1, 8)).all()
        results[protocol] = {}; comparisons[protocol] = {}
        for arm in cfg['arms']:
            directory = out / 'groups' / protocol / arm
            done, history = read(directory / 'COMPLETE.json'), read(directory / 'HISTORY.json')
            assert done['status'] == 'PASS' and done['checkpoint_prediction_replays'] == 6
            if arm == 'Original_ConvTran':
                assert done['baseline_replay_passed'] is True
            for seed in cfg['seeds']:
                records = history[str(seed)]
                assert [r['epoch'] for r in records] == list(range(1, 21))
                for record in records:
                    assert np.isfinite(record['train_ce'])
                    for view in ('raw', 'ema'):
                        metric = record['validation'][view]
                        f1, macro = scores(metric['confusion_matrix'])
                        np.testing.assert_allclose(f1, metric['f1'], atol=1e-14, rtol=0)
                        assert abs(macro - metric['macro_f1']) < 1e-14
                        assert np.asarray(metric['confusion_matrix']).sum() == len(truth)
                        assert np.isfinite(metric['weighted_ce'])
                        epoch_recounts += 1
                        curves.append(dict(protocol=protocol, arm=arm, seed=seed, epoch=record['epoch'], view=view,
                            train_ce=record['train_ce'], val_ce=metric['weighted_ce'], macro_f1_percent=100*macro,
                            **{f'{c}_f1_percent':100*f for c, f in zip(CLASSES, f1)}))
                for view in ('raw', 'ema'):
                    prediction = np.load(directory / f'seed{seed}_{view}_final20_predictions.npy', allow_pickle=False)
                    assert prediction.shape == truth.shape and np.isin(prediction, range(1, 8)).all()
                    cm = np.zeros((7, 7), np.int64)
                    np.add.at(cm, (truth-1, prediction-1), 1)
                    metric = done['final'][str(seed)][view]
                    np.testing.assert_array_equal(cm, metric['confusion_matrix'])
                    assert metric == records[-1]['validation'][view]
                    f1, macro = scores(cm)
                    assert abs(macro - metric['macro_f1']) < 1e-14
                    prediction_recounts += 1
                    finals.append(dict(protocol=protocol, arm=arm, seed=seed, view=view, macro_f1_percent=macro*100,
                        parameters=done['parameters'], **{f'{c}_f1_percent':100*f for c, f in zip(CLASSES, f1)}))
            results[protocol][arm] = {}
            for view in ('raw', 'ema'):
                selected = [r for r in finals if r['protocol'] == protocol and r['arm'] == arm and r['view'] == view]
                values = [r['macro_f1_percent'] for r in selected]
                results[protocol][arm][view] = dict(mean=float(np.mean(values)), seed_sd=float(np.std(values, ddof=1)),
                    per_seed=values, classes={c:float(np.mean([r[f'{c}_f1_percent'] for r in selected])) for c in CLASSES})
        for arm in cfg['arms']:
            main = results[protocol][cfg['primary']]['raw']; control = results[protocol][arm]['raw']
            delta = np.asarray(main['per_seed']) - np.asarray(control['per_seed'])
            comparisons[protocol][arm] = dict(per_seed_delta_pp=delta.tolist(), mean_delta_pp=float(delta.mean()),
                positive_seeds=int((delta > 0).sum()), class_mean_delta_pp={c:main['classes'][c]-control['classes'][c] for c in CLASSES})
    rules = cfg['pilot_review_rules']; per_protocol = {}
    for protocol in cfg['protocols']:
        pair = comparisons[protocol]['Original_ConvTran']; classes = pair['class_mean_delta_pp']
        per_protocol[protocol] = bool(pair['mean_delta_pp'] > rules['macro_mean_delta_pp_strictly_greater_than'] and
            pair['positive_seeds'] >= rules['minimum_positive_seed_pairs_each_protocol'] and
            classes['drinking'] >= rules['drinking_mean_delta_pp_at_least'] and
            classes['walking'] >= rules['walking_mean_delta_pp_at_least'] and
            classes['licking'] >= rules['licking_mean_delta_pp_at_least'])
    summary = dict(status='COMPLETE_STOP_FOR_REVIEW', results=results, primary_minus_controls_raw=comparisons,
        pilot_progression_rule_pass=per_protocol, final1pp_established=False, Test_allowed=False, automatic_expansion=False,
        epoch_confusion_recounts=epoch_recounts, final_prediction_recounts=prediction_recounts)
    save(out / 'SUMMARY.json', summary)
    write_csv(out / 'FIXED20_RESULTS.csv', finals); write_csv(out / 'LEARNING_CURVES.csv', curves)
    lines = ['# 双时间尺度ConvTran开发结果', '', '固定20轮raw为主，EMA单列；三种子均值；仅fold_1开发VAL。', '',
        '| 方法 | TS raw/% | OS raw/% | TS EMA/% | OS EMA/% |', '|---|---:|---:|---:|---:|']
    for arm in cfg['arms']:
        values = [results[p][arm][v]['mean'] for v in ('raw', 'ema') for p in cfg['protocols']]
        lines.append('| ' + arm + ' | ' + ' | '.join(f'{v:.4f}' for v in values) + ' |')
    lines += ['', f'最终预测重计{prediction_recounts}份；逐轮混淆矩阵重计{epoch_recounts}份。',
              '既定开发推进规则（非最终验收）：' + str(per_protocol),
              '完整模型与删交互版的差值用于检验交互块增量；未控制全部参数量解释。完成后停止，等待研究决定。']
    (out / 'RESULTS_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/sensor_convtran_context_20261022_v1.json')
    args = parser.parse_args(); cfg = read(args.config)
    result = report(cfg['output'], cfg)
    print(json.dumps({k:result[k] for k in ('status', 'pilot_progression_rule_pass', 'final_prediction_recounts')}))
