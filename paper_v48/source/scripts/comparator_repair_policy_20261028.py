"""Locked early stopping and paired guard; never relabel as terminal stationarity."""
import math


def selected(history):
    if not history or any(not math.isfinite(r['dev_ema']['weighted_CE']) for r in history):
        raise ValueError('Missing or nonfinite history')
    return min(history, key=lambda r: (r['dev_ema']['weighted_CE'], r['epoch']))


def stopping(history, cfg):
    selected(history)
    meaningful = history[0]['dev_ema']['weighted_CE']
    last = history[0]['epoch']
    for row in history[1:]:
        if row['dev_ema']['weighted_CE'] < meaningful * (1-cfg['relative_improvement']):
            meaningful, last = row['dev_ema']['weighted_CE'], row['epoch']
    end = history[-1]['epoch']
    saturated = end >= cfg['minimum_epochs'] and end-last >= cfg['patience_epochs']
    return dict(stop=saturated or end >= cfg['maximum_epochs'], validation_plateau=saturated,
                last_material_improvement_epoch=last, selected_epoch=selected(history)['epoch'],
                reason='VALIDATION_PLATEAU' if saturated else 'MAXIMUM_BUDGET' if end >= cfg['maximum_epochs'] else 'CONTINUE')


def compare(candidate, baseline, cfg):
    a, b = candidate['dev_ema'], baseline['dev_ema']
    f1_delta = a['macro_f1_percent']-b['macro_f1_percent']
    ce_delta = a['weighted_CE']/b['weighted_CE']-1
    a_gap = candidate['fit_ema']['macro_f1_percent']-a['macro_f1_percent']
    b_gap = baseline['fit_ema']['macro_f1_percent']-b['macro_f1_percent']
    checks = dict(DEV_CE_not_materially_worse=ce_delta <= cfg['max_relative_CE_increase'],
                  DEV_F1_not_materially_worse=f1_delta >= -cfg['max_F1_drop_pp'],
                  selected_F1_gap_growth_bounded=a_gap-b_gap <= cfg['max_F1_gap_growth_pp'])
    return dict(paired_guard_pass=all(checks.values()), checks=checks, selected_CE_relative_change=ce_delta,
                selected_F1_change_pp=f1_delta, selected_Fit_DEV_gap_change_pp=a_gap-b_gap,
                boundary='Reused development data; no independent generalization or no-overfit proof')


def allow_confirmation(receipts):
    if len(receipts) != 2 or {r['protocol'] for r in receipts} != {'TS', 'OS'}:
        return False
    return all(r['stopping']['validation_plateau'] and r['comparison']['paired_guard_pass'] for r in receipts)
