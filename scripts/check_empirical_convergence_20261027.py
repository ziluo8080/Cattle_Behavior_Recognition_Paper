"""A red-capable check of saved, actual DEV trajectories (no inference)."""
import argparse
import json
from pathlib import Path
import numpy as np


def assess(history):
    # Two consecutive five-epoch blocks, fixed before new experiments.
    assert len(history) >= 10
    tail = history[-10:]
    ce = np.array([r['dev_ema']['weighted_CE'] for r in tail], dtype=float)
    f1 = np.array([r['dev_ema']['macro_f1_percent'] for r in tail], dtype=float)
    relative_ce_drift = float((ce[5:].mean() - ce[:5].mean()) / ce[:5].mean())
    return {
        'CE_relative_block_drift': relative_ce_drift,
        'F1_block_drift_pp': float(f1[5:].mean() - f1[:5].mean()),
        'F1_tail_range_pp': float(np.ptp(f1)),
        'CE_stationary_within_2_percent': bool(abs(relative_ce_drift) <= .02),
        'F1_stationary_within_0_5pp': bool(abs(f1[5:].mean()-f1[:5].mean()) <= .5),
    }


def strong_assess(history):
    """Twenty actual training epochs; positive LR and unchanged hard-label metrics."""
    if len(history) < 40:
        return dict(passed=False, reason='minimum40epochs')
    windows = [history[-20:-10], history[-10:]]
    checks = []
    for window in windows:
        item = assess(window)
        for field, tolerance in [('train_ce', .05), ('fit_ema', .02)]:
            values = np.array([r[field]['weighted_CE'] if field=='fit_ema' else r[field] for r in window])
            drift = float((values[5:].mean()-values[:5].mean())/values[:5].mean())
            item[field+'_relative_block_drift'] = drift
            item[field+'_stationary'] = abs(drift) <= tolerance
        f = np.array([r['fusion_ema']['macro_f1_percent'] for r in window])
        item['fusion_F1_block_drift_pp'] = float(f[5:].mean()-f[:5].mean())
        item['fusion_F1_tail_range_pp'] = float(np.ptp(f))
        item['CE_within5percent_of_run_minimum'] = float(np.mean([r['dev_ema']['weighted_CE'] for r in window])) <= min(r['dev_ema']['weighted_CE'] for r in history)*1.05
        item['raw_parameter_update_small'] = max(r['raw_relative_parameter_change'] for r in window) <= .002
        item['EMA_parameter_update_small'] = max(r['EMA_relative_parameter_change'] for r in window) <= .002
        item['positive_learning_rate'] = min(r['learning_rate'] for r in window) >= 3e-6
        item['passed'] = bool(item['CE_stationary_within_2_percent'] and item['F1_stationary_within_0_5pp']
                              and item['F1_tail_range_pp'] <= 1.0 and abs(item['fusion_F1_block_drift_pp']) <= .5
                              and item['fusion_F1_tail_range_pp'] <= 1.0 and item['train_ce_stationary']
                              and item['fit_ema_stationary'] and item['CE_within5percent_of_run_minimum']
                              and item['raw_parameter_update_small'] and item['EMA_parameter_update_small']
                              and item['positive_learning_rate'])
        checks.append(item)
    return dict(passed=all(w['passed'] for w in checks), windows=checks,
                scope='Observed empirical optimization and validation stationarity; no global optimum claim.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases = []
    for path in sorted((args.root/'trajectories').rglob('history.json')):
        history = json.loads(path.read_text(encoding='utf-8-sig'))
        result = assess(history)
        result.update(case=path.relative_to(args.root).as_posix())
        result['passed'] = result['CE_stationary_within_2_percent'] and result['F1_stationary_within_0_5pp']
        cases.append(result)
    assert len(cases) == 18, f'Expected18 completed trajectories; got{len(cases)}'
    report = dict(status='PASS' if all(c['passed'] for c in cases) else 'FAIL',
                  scope='Empirical DEV stationarity on each observed case; not proof of a global optimum.',
                  criteria=dict(tail_epochs=10, block_epochs=5, max_abs_relative_CE_drift=.02,
                                max_abs_F1_block_drift_pp=.5, all18_required=True),
                  cases=cases, passed=sum(c['passed'] for c in cases), total=len(cases))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(f"EMPIRICAL_CONVERGENCE_{report['status']} {report['passed']}/18 cases")
    print('CE drift range:', min(c['CE_relative_block_drift'] for c in cases), max(c['CE_relative_block_drift'] for c in cases))
    raise SystemExit(0 if report['status']=='PASS' else 1)


if __name__ == '__main__':
    main()
