"""Independent NumPy recount of fixed20 development evidence and locked rules."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.report_convtran_context_20261022 import CLASSES, read, save, scores, write_csv


def audit_group(directory, truth, cfg, protocol, arm, source, curves, finals):
    done, history = read(directory / 'COMPLETE.json'), read(directory / 'HISTORY.json')
    assert done['status'] == 'PASS' and done['checkpoint_prediction_replays'] == 6
    if source == 'new' and arm == 'TwoRate_Concat': assert done['baseline_replay_passed'] is True
    assert set(history) == {str(s) for s in cfg['seeds']}
    for seed in cfg['seeds']:
        records = history[str(seed)]
        assert [r['epoch'] for r in records] == list(range(1, 21))
        for record in records:
            assert np.isfinite(record['train_ce'])
            for view in ('raw', 'ema'):
                metric = record['validation'][view]
                f1, macro = scores(metric['confusion_matrix'])
                np.testing.assert_allclose(f1, metric['f1'], atol=1e-14, rtol=0)
                assert abs(macro-metric['macro_f1']) < 1e-14
                assert np.asarray(metric['confusion_matrix']).sum() == len(truth)
                assert np.isfinite(metric['weighted_ce'])
                curves.append(dict(protocol=protocol, arm=arm, source=source, seed=seed,
                    epoch=record['epoch'], view=view, train_ce=record['train_ce'], val_ce=metric['weighted_ce'],
                    macro_f1_percent=100*macro,
                    **{f'{c}_f1_percent':100*f for c,f in zip(CLASSES,f1)}))
        for view in ('raw', 'ema'):
            pred = np.load(directory / f'seed{seed}_{view}_final20_predictions.npy', allow_pickle=False)
            assert pred.shape == truth.shape and np.isin(pred, range(1,8)).all()
            cm = np.zeros((7,7), np.int64); np.add.at(cm, (truth-1,pred-1), 1)
            metric = done['final'][str(seed)][view]
            np.testing.assert_array_equal(cm, metric['confusion_matrix'])
            assert metric == records[-1]['validation'][view]
            f1, macro = scores(cm)
            assert abs(macro-metric['macro_f1']) < 1e-14
            finals.append(dict(protocol=protocol, arm=arm, source=source, seed=seed, view=view,
                macro_f1_percent=100*macro, parameters=done['parameters'],
                **{f'{c}_f1_percent':100*f for c,f in zip(CLASSES,f1)}))
    result = {}
    for view in ('raw','ema'):
        rows = [r for r in finals if r['protocol']==protocol and r['arm']==arm and r['source']==source and r['view']==view]
        assert [r['seed'] for r in rows] == cfg['seeds']
        values = [r['macro_f1_percent'] for r in rows]
        result[view] = dict(mean=float(np.mean(values)), seed_sd=float(np.std(values,ddof=1)),
            per_seed=values, classes={c:float(np.mean([r[f'{c}_f1_percent'] for r in rows])) for c in CLASSES})
    return result


def pair(candidate, reference):
    delta = np.asarray(candidate['per_seed'])-np.asarray(reference['per_seed'])
    return dict(mean_delta_pp=float(delta.mean()), per_seed_delta_pp=delta.tolist(),
        positive_seeds=int((delta>0).sum()),
        class_mean_delta_pp={c:candidate['classes'][c]-reference['classes'][c] for c in CLASSES})


def decide(results, cfg):
    rules = cfg['candidate_rules']; comparisons = {}; eligible = {}; gates = {}
    for arm in ('FullResidual_A','EndpointResidual_B'):
        gates[arm] = {}; comparisons[arm] = {}
        for protocol in cfg['protocols']:
            values = results[protocol]
            # Selection is raw-only. EMA remains descriptive, never a fallback.
            comparisons[arm][protocol] = {ref:pair(values[arm]['raw'],values[ref]['raw'])
                for ref in ('TwoRate_Concat','Original_ConvTran','TwoRate_Interaction','FullResidual_A','EndpointResidual_B')}
            cat = comparisons[arm][protocol]['TwoRate_Concat']
            old = comparisons[arm][protocol]['Original_ConvTran']
            checks = dict(
                concat_mean=cat['mean_delta_pp']>rules['vs_concat_macro_delta_pp_strictly_greater_than'],
                concat_pairs=cat['positive_seeds']>=rules['minimum_positive_seed_pairs_each_protocol'],
                original_mean=old['mean_delta_pp']>rules['vs_original_macro_delta_pp_strictly_greater_than'],
                original_pairs=old['positive_seeds']>=rules['minimum_positive_seed_pairs_each_protocol'],
                drinking=old['class_mean_delta_pp']['drinking']>=rules['vs_original_drinking_delta_pp_at_least'],
                walking=old['class_mean_delta_pp']['walking']>=rules['vs_original_walking_delta_pp_at_least'],
                licking=old['class_mean_delta_pp']['licking']>=rules['vs_original_licking_delta_pp_at_least'])
            gates[arm][protocol] = dict(checks=checks, passed=all(checks.values()))
        eligible[arm] = all(gates[arm][p]['passed'] for p in cfg['protocols'])
    b_over_a = all(comparisons['EndpointResidual_B'][p]['FullResidual_A']['mean_delta_pp']>0
        and comparisons['EndpointResidual_B'][p]['FullResidual_A']['positive_seeds']>=2 for p in cfg['protocols'])
    if all(eligible.values()): chosen = 'EndpointResidual_B' if b_over_a else 'FullResidual_A'
    elif eligible['FullResidual_A']: chosen = 'FullResidual_A'
    elif eligible['EndpointResidual_B']: chosen = 'EndpointResidual_B'
    else: chosen = None
    return dict(selected_development_candidate=chosen, eligible=eligible, gates=gates,
                B_over_A_interface_increment_rule=b_over_a, comparisons_raw=comparisons)


def report(cfg):
    out, reference = Path(cfg['output']), Path(cfg['reference_output'])
    results, curves, finals = {}, [], []
    for protocol in cfg['protocols']:
        target = out / 'data' / protocol / 'targets.jsonl'
        assert target.read_bytes() == (reference / 'data' / protocol / 'targets.jsonl').read_bytes()
        rows = [json.loads(line) for line in target.read_text().splitlines()]
        assert {r['split'] for r in rows} == {'train','val'}
        truth = np.asarray([r['label'] for r in rows if r['split']=='val'])
        assert np.isin(truth,range(1,8)).all()
        results[protocol] = {}
        for arm in cfg['arms']:
            results[protocol][arm] = audit_group(out/'groups'/protocol/arm,truth,cfg,protocol,arm,'new',curves,finals)
        for arm in ('Original_ConvTran','TwoRate_Interaction'):
            results[protocol][arm] = audit_group(reference/'groups'/protocol/arm,truth,cfg,protocol,arm,'reused',curves,finals)
        # Independently verify the reproduced concat evidence against the saved prior trajectories.
        new = read(out/'groups'/protocol/'TwoRate_Concat'/'HISTORY.json')
        old = read(reference/'groups'/protocol/'TwoRate_Concat'/'HISTORY.json')
        for seed in cfg['seeds']:
            for n,o in zip(new[str(seed)],old[str(seed)]):
                assert n['train_ce']==o['train_ce'] and n['validation']==o['validation']
            for view in ('raw','ema'):
                name = f'seed{seed}_{view}_final20_predictions.npy'
                np.testing.assert_array_equal(np.load(out/'groups'/protocol/'TwoRate_Concat'/name),
                                               np.load(reference/'groups'/protocol/'TwoRate_Concat'/name))
    summary = dict(status='COMPLETE_STOP_FOR_REVIEW', results=results, **decide(results,cfg),
        final1pp_established=False, Test_allowed=False, automatic_expansion=False,
        final_prediction_recounts=len(finals), epoch_confusion_recounts=len(curves),
        new_final_prediction_recounts=sum(r['source']=='new' for r in finals),
        new_epoch_confusion_recounts=sum(r['source']=='new' for r in curves))
    save(out/'SUMMARY.json',summary)
    write_csv(out/'FIXED20_RESULTS.csv',finals); write_csv(out/'LEARNING_CURVES.csv',curves)
    lines = ['# 双时间尺度残差交互：固定20轮开发结果','',
        '仅TS/OS fold_1；三种子均值；raw20为主、EMA20单列。旧原ConvTran与旧交互复用封存结果。','',
        '| 方法 | TS raw/% | OS raw/% | TS EMA/% | OS EMA/% |','|---|---:|---:|---:|---:|']
    for arm in ('Original_ConvTran','TwoRate_Concat','TwoRate_Interaction','FullResidual_A','EndpointResidual_B'):
        values = [results[p][arm][v]['mean'] for v in ('raw','ema') for p in cfg['protocols']]
        lines.append('| '+arm+' | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    lines += ['', '预定规则选择的开发候选：'+str(summary['selected_development_candidate']),
        'A/B合格状态：'+str(summary['eligible']),
        'B对A整体读出接口增量规则：'+str(summary['B_over_A_interface_increment_rule']),
        '本结论不是跨折稳定性或最终1pp验收。不能以EMA、最高中间轮或单协议替代本轮raw20结论。',
        f'新结果重计{summary["new_final_prediction_recounts"]}份末轮预测和{summary["new_epoch_confusion_recounts"]}份逐轮混淆矩阵；含复用参照共{len(finals)}和{len(curves)}份。']
    (out/'RESULTS_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config',default='configs/sensor_convtran_endpoint_pilot_20261022_v1.json')
    cfg=read(parser.parse_args().config); summary=report(cfg)
    print(json.dumps({k:summary[k] for k in ('status','eligible','selected_development_candidate')}))
