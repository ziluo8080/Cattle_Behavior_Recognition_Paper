"""Source-aware empirical stationarity, independent of model/training runtime."""
import math
from statistics import mean


def assess(history, cfg):
    if len(history) < 40:
        return {'passed': False, 'reason': 'minimum40epochs'}
    stage = history[-1]['stage']
    stage_rows = [r for r in history if r['stage'] == stage]
    if len(stage_rows) < cfg['minimum_stage_epochs']:
        return {'passed': False, 'reason': 'minimum40epochsInCurrentStage'}
    original_min = min(r['dev_ema']['weighted_CE'] for r in history)
    checks = []
    for rows in [history[-20:-10], history[-10:]]:
        item = {}
        passed = all(r['stage'] == stage for r in rows)
        for key, limit in [('train_ce', cfg['max_abs_train_CE_relative_block_drift']),
                           ('fit_ema', cfg['max_abs_fit_EMA_CE_relative_block_drift']),
                           ('dev_ema', cfg['max_abs_DEV_CE_relative_block_drift'])]:
            values = [r[key] if key == 'train_ce' else r[key]['weighted_CE'] for r in rows]
            drift = (mean(values[5:]) - mean(values[:5])) / mean(values[:5])
            item[key+'_relative_block_drift'] = drift
            passed = passed and math.isfinite(drift) and abs(drift) <= limit
        f1 = [r['dev_ema']['macro_f1_percent'] for r in rows]
        item['DEV_F1_block_drift_pp'] = mean(f1[5:]) - mean(f1[:5])
        item['DEV_F1_range_pp'] = max(f1)-min(f1)
        item['DEV_CE_near_original_minimum'] = mean([r['dev_ema']['weighted_CE'] for r in rows]) <= original_min*cfg['DEV_CE_within_run_minimum_factor']
        item['positive_LR'] = min(r['learning_rate'] for r in rows) >= cfg['minimum_positive_learning_rate']
        item['maximum_raw_parameter_update'] = max(r['raw_relative_parameter_change'] for r in rows)
        item['maximum_EMA_parameter_update'] = max(r['EMA_relative_parameter_change'] for r in rows)
        passed = (passed and abs(item['DEV_F1_block_drift_pp']) <= cfg['max_abs_DEV_F1_block_drift_pp']
                  and item['DEV_F1_range_pp'] <= cfg['max_DEV_F1_window_range_pp']
                  and item['DEV_CE_near_original_minimum'] and item['positive_LR']
                  and item['maximum_raw_parameter_update'] <= cfg['max_relative_parameter_update']
                  and item['maximum_EMA_parameter_update'] <= cfg['max_relative_parameter_update'])
        item['passed'] = bool(passed)
        checks.append(item)
    return {'passed': all(w['passed'] for w in checks), 'windows': checks,
            'source_aware_DEV_CE_minimum': original_min,
            'scope': 'empirical optimization and validation stationarity, not a global optimum certificate'}
