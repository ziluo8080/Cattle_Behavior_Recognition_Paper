"""Observed-DEV deterioration screen. No blanket absolute train/DEV-gap rule."""
from statistics import mean
import math


def normalize(history, tree=False):
    rows=[]
    for i,r in enumerate(history):
        fit=r.get('fit_eval',r.get('train')) if tree else r['fit_ema']
        dev=r['dev'] if tree else r['dev_ema']
        row=dict(index=i,step=r['rounds'] if tree else r['epoch'],stage='boosting' if tree else r['stage'],
                 fit_ce=fit['weighted_CE'],dev_ce=dev['weighted_CE'],fit_f1=fit['macro_f1_percent'],dev_f1=dev['macro_f1_percent'])
        if any(not isinstance(v,(int,float)) or not math.isfinite(v) for k,v in row.items() if k in ['fit_ce','dev_ce','fit_f1','dev_f1']):
            raise ValueError('Non-finite or absent comparable Fit/DEV metrics')
        if row['dev_ce']<=0 or row['fit_ce']<0:raise ValueError('Invalid CE')
        rows.append(row)
    return rows


def summarize(rows):
    return dict(start=rows[0]['step'],end=rows[-1]['step'],stage=rows[0]['stage'],
                **{k:mean(r[k] for r in rows) for k in ['fit_ce','dev_ce','fit_f1','dev_f1']})


def assess(history,cfg,tree=False):
    rows=normalize(history,tree)
    if len(rows)<max(cfg['reference_rows'],cfg['tail_rows']):
        return dict(passed=False,status='INSUFFICIENT_EVIDENCE')
    windows=[];n=cfg['reference_rows']
    for i in range(len(rows)-n+1):
        w=rows[i:i+n]
        if all(r['stage']==w[0]['stage'] for r in w):windows.append(summarize(w))
    tailrows=rows[-cfg['tail_rows']:]
    if not windows or any(r['stage']!=tailrows[-1]['stage'] for r in tailrows):
        return dict(passed=False,status='INSUFFICIENT_SINGLE_STAGE_TAIL')
    if sum(r['stage']==rows[-1]['stage'] for r in rows)<cfg['minimum_stage_rows']:
        return dict(passed=False,status='INSUFFICIENT_STAGE_HISTORY')
    current_windows=[w for w in windows if w['stage']==rows[-1]['stage']]
    reference=min(current_windows,key=lambda w:(w['dev_ce'],w['end']))
    f1_reference=max(current_windows,key=lambda w:(w['dev_f1'],-w['end']))
    global_reference=min(windows,key=lambda w:(w['dev_ce'],w['end']))
    global_f1_reference=max(windows,key=lambda w:(w['dev_f1'],-w['end']))
    tail=summarize(tailrows);terminal=rows[-1]
    ce_tail=tail['dev_ce']/reference['dev_ce']-1
    ce_terminal=terminal['dev_ce']/reference['dev_ce']-1
    f1_tail_drop=f1_reference['dev_f1']-tail['dev_f1']
    f1_terminal_drop=f1_reference['dev_f1']-terminal['dev_f1']
    ce_gap_ref=reference['dev_ce']-reference['fit_ce'];ce_gap_tail=tail['dev_ce']-tail['fit_ce']
    f1_gap_ref=reference['fit_f1']-reference['dev_f1'];f1_gap_tail=tail['fit_f1']-tail['dev_f1']
    gap_ce_growth=(ce_gap_tail-ce_gap_ref)/reference['dev_ce']
    gap_f1_growth=f1_gap_tail-f1_gap_ref
    ce_limit=reference['dev_ce']*(1+cfg['max_DEV_CE_deterioration_relative'])+cfg['max_DEV_CE_deterioration_absolute']
    gap_limit=max(cfg['max_CE_gap_growth_absolute'],cfg['max_CE_gap_growth_relative_to_reference_DEV_CE']*reference['dev_ce'])
    terminal_ce_gap_growth=(terminal['dev_ce']-terminal['fit_ce'])-ce_gap_ref
    terminal_f1_gap_growth=(terminal['fit_f1']-terminal['dev_f1'])-f1_gap_ref
    checks=dict(tail_DEV_CE_not_materially_worse=tail['dev_ce']<=ce_limit,
                terminal_DEV_CE_not_materially_worse=terminal['dev_ce']<=ce_limit,
                tail_DEV_F1_not_materially_worse=f1_tail_drop<=cfg['max_DEV_F1_drop_pp'],
                terminal_DEV_F1_not_materially_worse=f1_terminal_drop<=cfg['max_DEV_F1_drop_pp'],
                tail_CE_gap_growth_bounded=ce_gap_tail-ce_gap_ref<=gap_limit,
                terminal_CE_gap_growth_bounded=terminal_ce_gap_growth<=gap_limit,
                tail_F1_gap_growth_bounded=gap_f1_growth<=cfg['max_F1_gap_growth_pp'],
                terminal_F1_gap_growth_bounded=terminal_f1_gap_growth<=cfg['max_F1_gap_growth_pp'])
    global_ce_limit=global_reference['dev_ce']*(1+cfg['max_DEV_CE_deterioration_relative'])+cfg['max_DEV_CE_deterioration_absolute']
    checks.update(tail_DEV_CE_respects_source_history=tail['dev_ce']<=global_ce_limit,
                  terminal_DEV_CE_respects_source_history=terminal['dev_ce']<=global_ce_limit,
                  tail_DEV_F1_respects_source_history=global_f1_reference['dev_f1']-tail['dev_f1']<=cfg['max_DEV_F1_drop_pp'],
                  terminal_DEV_F1_respects_source_history=global_f1_reference['dev_f1']-terminal['dev_f1']<=cfg['max_DEV_F1_drop_pp'])
    return dict(passed=all(checks.values()),status='OBSERVED_DEV_SCREEN_PASS' if all(checks.values()) else 'GENERALIZATION_HOLD',
                checks=checks,reference_CE_window=reference,reference_F1_window=f1_reference,tail=tail,terminal=terminal,
                global_reference_CE_window=global_reference,global_reference_F1_window=global_f1_reference,
                relative_tail_DEV_CE_deterioration=ce_tail,relative_terminal_DEV_CE_deterioration=ce_terminal,
                tail_DEV_F1_drop_pp=f1_tail_drop,terminal_DEV_F1_drop_pp=f1_terminal_drop,
                fit_DEV_CE_gap_at_reference=ce_gap_ref,fit_DEV_CE_gap_tail=ce_gap_tail,
                fit_DEV_F1_gap_at_reference_pp=f1_gap_ref,fit_DEV_F1_gap_tail_pp=f1_gap_tail,
                relative_CE_gap_growth=gap_ce_growth,F1_gap_growth_pp=gap_f1_growth,
                CE_gap_growth_limit=gap_limit,terminal_CE_gap_growth=terminal_ce_gap_growth,terminal_F1_gap_growth_pp=terminal_f1_gap_growth,
                boundary='No material progressive deterioration under these observed-DEV tolerances; not a proof of no overfitting')
