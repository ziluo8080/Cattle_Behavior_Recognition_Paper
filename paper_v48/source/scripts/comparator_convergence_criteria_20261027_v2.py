"""Stricter cross-window and terminal guards, preserving the V1 history."""
from statistics import mean
from scripts.comparator_convergence_criteria_20261027 import assess as original_assess


def assess(history, cfg):
    result=original_assess(history,cfg)
    if 'windows' not in result:return result
    rows=history[-20:];a,b=rows[:10],rows[10:];checks={}
    for key,limit in [('train_ce',cfg['max_abs_train_CE_relative_block_drift']),
                      ('fit_ema',cfg['max_abs_fit_EMA_CE_relative_block_drift']),
                      ('dev_ema',cfg['max_abs_DEV_CE_relative_block_drift'])]:
        values=[r[key] if key=='train_ce' else r[key]['weighted_CE'] for r in rows]
        drift=(mean(values[10:])-mean(values[:10]))/mean(values[:10])
        checks[key+'_cross_window_relative_drift']=drift
        checks[key+'_cross_window_pass']=abs(drift)<=limit
    values=[r['dev_ema']['weighted_CE'] for r in rows]
    best=min(r['dev_ema']['weighted_CE'] for r in history)
    checks['DEV_CE_relative_range']=(max(values)-min(values))/min(values)
    checks['DEV_CE_range_pass']=checks['DEV_CE_relative_range']<=cfg['max_DEV_CE_window_relative_range']
    checks['terminal_DEV_CE_near_original_minimum']=values[-1]<=best*cfg['terminal_DEV_CE_within_run_minimum_factor']
    f=[r['dev_ema']['macro_f1_percent'] for r in rows]
    checks['DEV_F1_cross_window_drift_pp']=mean(f[10:])-mean(f[:10])
    checks['DEV_F1_cross_window_pass']=abs(checks['DEV_F1_cross_window_drift_pp'])<=cfg['max_abs_cross_window_DEV_F1_drift_pp']
    checks['DEV_F1_full20_range_pp']=max(f)-min(f)
    checks['DEV_F1_full20_range_pass']=checks['DEV_F1_full20_range_pp']<=cfg['max_cross_window_DEV_F1_range_pp']
    result['cross_window_terminal_checks']=checks
    gates=[v for k,v in checks.items() if k.endswith('_pass') or k=='terminal_DEV_CE_near_original_minimum']
    result['passed']=bool(result['passed'] and all(gates))
    return result
