"""Freeze every endpoint before one label-scoring pass; no checkpoint selection."""
from fractions import Fraction
from pathlib import Path
import json

import numpy as np
from scripts.sensor_learned_fusion_data_20261023 import read, save, sha
from scripts.sensor_xgb_b_fusion_20261022 import validate_probabilities, temper, fuse
from scripts.report_convtran_context_20261022 import CLASSES, write_csv
from scripts.report_tree_anchored_residual_20261024 import metrics


def read_probability(path):
    probability = np.load(path, allow_pickle=False)
    validate_probabilities(probability)  # Validate without another normalization.
    return probability


def report(cfg):
    out = Path(cfg['output'])
    assert not (out / 'PREDICTIONS_FROZEN.json').exists()
    frozen = {}
    cases = {}
    for p in cfg['protocols']:
        for k in cfg['inners']:
            base = out / 'baseline' / p / f'inner{k}'
            px = read_probability(base / 'XGB.npy')
            case = {}
            for s in cfg['seeds']:
                folder = out / 'neural' / p / f'inner{k}' / f'seed{s}'
                complete = read(folder / 'COMPLETE.json')
                assert (complete['status'], complete['protocol'], complete['inner'], complete['seed'], complete['epoch']) == ('COMPLETE', p, k, s, 20)
                for path, digest in complete['files'].items():
                    assert sha(path) == digest, path
                    frozen[path] = digest
                h = read(folder / 'history.json')
                assert [r['epoch'] for r in h] == list(range(1, 21)) and all(np.isfinite(r['train_ce']) for r in h)
                frozen[str(folder / 'COMPLETE.json')] = sha(folder / 'COMPLETE.json')
                case['XGB', 'ema', s] = px
                for n, source in [('OriginalB', 'B'), ('OriginalGate', 'Gate')]:
                    path = base / f'{source}_{s}.npy'
                    case[n, 'ema', s] = read_probability(path)
                    frozen[str(path)] = sha(path)
                for v in ['ema', 'raw']:
                    probability = read_probability(folder / f'held_{v}.npy')
                    gate = fuse(temper(px, 2), temper(probability, 1), 'margin_select', .7)
                    path = out / 'predictions' / p / f'inner{k}' / f'ZeroGate_{v}_{s}.npy'
                    assert not path.exists()
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, gate)
                    frozen[str(path)] = sha(path)
                    case['NativeZeroHead', v, s] = probability
                    case['ZeroGate', v, s] = gate
            frozen[str(base / 'XGB.npy')] = sha(base / 'XGB.npy')
            gold = out / 'data' / p / f'inner{k}' / 'DEV_TARGETS.jsonl'
            frozen[str(gold)] = sha(gold)
            cases[p, k] = case
    save(out / 'PREDICTIONS_FROZEN.json', dict(status='PASS', files=frozen, new_models=18, new_held_scores_before_freeze=0))
    results = {}
    fractions = {}
    endpoint_rows, paired, class_rows, error_rows = [], [], [], []
    for (p, k), case in cases.items():
        gold = out / 'data' / p / f'inner{k}' / 'DEV_TARGETS.jsonl'
        truth = np.asarray([json.loads(line)['label'] for line in gold.read_text().splitlines()])
        for (method, view, seed), probability in case.items():
            assert len(probability) == len(truth)
            m, fraction, _ = metrics(truth, probability)
            results[p, k, method, view, seed] = m
            fractions[p, k, method, view, seed] = fraction
            endpoint_rows.append(dict(protocol=p, inner=k, method=method, view=view, seed=seed, **m))
            for i, behavior in enumerate(CLASSES):
                class_rows.append(dict(protocol=p, inner=k, method=method, view=view, seed=seed, behavior=behavior,
                                       F1_percent=m['class_f1_percent'][i], precision_percent=m['precision_percent'][i], recall_percent=m['recall_percent'][i], support=m['support'][i]))
        for seed in cfg['seeds']:
            for method, reference in [('NativeZeroHead', 'OriginalB'), ('ZeroGate', 'OriginalGate'), ('ZeroGate', 'XGB')]:
                delta = fractions[p, k, method, 'ema', seed] - fractions[p, k, reference, 'ema', seed]
                paired.append(dict(protocol=p, inner=k, seed=seed, method=method, reference=reference, delta_F1_pp=float(delta)))
            x = case['XGB', 'ema', seed].argmax(1) + 1
            a = case['ZeroGate', 'ema', seed].argmax(1) + 1
            for i, behavior in enumerate(CLASSES, 1):
                error_rows.append(dict(protocol=p, inner=k, seed=seed, behavior=behavior,
                    corrected_XGB_errors=int(((x != truth) & (a == truth) & (truth == i)).sum()),
                    damaged_XGB_correct=int(((x == truth) & (a != truth) & (truth == i)).sum()),
                    removed_XGB_false_positives=int(((x == i) & (truth != i) & (a != i)).sum()),
                    introduced_false_positives=int(((x != i) & (truth != i) & (a == i)).sum())))
    methods = [('XGB', 'ema'), ('OriginalB', 'ema'), ('OriginalGate', 'ema')] + [(m, v) for m in ['NativeZeroHead', 'ZeroGate'] for v in ['ema', 'raw']]
    summary, exact = {}, {}
    for method, view in methods:
        name = method if view == 'ema' else method + '_raw'
        summary[name], exact[name] = {}, {}
        for p in cfg['protocols']:
            inner_means = [sum((fractions[p, k, method, view, s] for s in cfg['seeds']), Fraction()) / 3 for k in cfg['inners']]
            mean = sum(inner_means, Fraction()) / 3
            exact[name][p] = mean
            per_class = np.mean([results[p, k, method, view, s]['class_f1_percent'] for k in cfg['inners'] for s in cfg['seeds']], axis=0)
            summary[name][p] = dict(mean_percent=float(mean), per_inner_mean_percent=[float(v) for v in inner_means],
                                    class_f1_percent=dict(zip(CLASSES, per_class.tolist())))
    decisions = {}
    for p in cfg['protocols']:
        gate_pairs = [r for r in paired if r['protocol'] == p and r['method'] == 'ZeroGate' and r['reference'] == 'OriginalGate']
        x_pairs = [r for r in paired if r['protocol'] == p and r['method'] == 'ZeroGate' and r['reference'] == 'XGB']
        b_pairs = [r for r in paired if r['protocol'] == p and r['method'] == 'NativeZeroHead']
        inner_wins = sum(sum((fractions[p, k, 'ZeroGate', 'ema', s] - fractions[p, k, 'OriginalGate', 'ema', s] for s in cfg['seeds']), Fraction()) > 0 for k in cfg['inners'])
        gate_wins = sum(r['delta_F1_pp'] > 0 for r in gate_pairs)
        x_wins = sum(r['delta_F1_pp'] > 0 for r in x_pairs)
        b_wins = sum(r['delta_F1_pp'] > 0 for r in b_pairs)
        decisions[p] = dict(delta_gate_pp=float(exact['ZeroGate'][p] - exact['OriginalGate'][p]), delta_XGB_pp=float(exact['ZeroGate'][p] - exact['XGB'][p]),
             delta_B_pp=float(exact['NativeZeroHead'][p] - exact['OriginalB'][p]), positive_gate_pairs=gate_wins, positive_XGB_pairs=x_wins,
             positive_B_pairs=b_wins, positive_gate_inner_means=inner_wins,
             gate_success=exact['ZeroGate'][p] > max(exact['OriginalGate'][p], exact['XGB'][p]) and inner_wins >= 2 and gate_wins >= 6 and x_wins >= 6,
             initialization_success=exact['NativeZeroHead'][p] > exact['OriginalB'][p] and b_wins >= 6)
    target_ok = all(summary['ZeroGate']['OS']['class_f1_percent'][c] >= summary['XGB']['OS']['class_f1_percent'][c] for c in ['licking', 'drinking'])
    gate_ok = all(d['gate_success'] for d in decisions.values())
    result = dict(status='COMPLETE_STOP_FOR_REVIEW', scope=cfg['scope'], primary='fixed_epoch20_EMA', aggregation=cfg['aggregation'], summary=summary,
                  decisions=decisions, candidate_supported=gate_ok, initialization_supported=all(d['initialization_success'] for d in decisions.values()),
                  OS_target_classes_preserved=target_ok, full_target_solution_supported=gate_ok and target_ok,
                  recommendation='Review development candidate; no automatic formal refit' if gate_ok else 'Retain original shared fusion; do not expand fixed zero-head recipe',
                  scientific_new_fits=18, original_fivefold_unchanged=True, new_blind_evaluation=False, automatic_expansion=False)
    save(out / 'PER_SEED_METRICS.json', endpoint_rows)
    write_csv(out / 'PAIRED_RESULTS.csv', paired)
    write_csv(out / 'PER_CLASS_RESULTS.csv', class_rows)
    write_csv(out / 'CLASS_ERROR_CHANGES.csv', error_rows)
    save(out / 'SUMMARY.json', result)
    lines = ['# 零初始化匹配复核完成', '', '仅outer TRAIN内部三个开发inner；每inner三种子后等权平均。不是完整五折或新盲测。', '',
             '| 方法（EMA20） | TS宏F1/% | OS宏F1/% |', '|---|---:|---:|']
    for name in ['OriginalGate', 'XGB', 'OriginalB', 'NativeZeroHead', 'ZeroGate']:
        lines.append(f"| {name} | {summary[name]['TS']['mean_percent']:.6f} | {summary[name]['OS']['mean_percent']:.6f} |")
    lines += ['', f'候选满足双协议稳定条件：{gate_ok}。OS舔舐与饮水均不低于XGB：{target_ok}。',
              '原主线及五折结果不变；固定候选不跨协议、精度或checkpoint补选，不自动扩大。']
    (out / 'RESULTS_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result
