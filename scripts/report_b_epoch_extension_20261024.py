"""Freeze all duration endpoints, then report development optima and stability."""
from fractions import Fraction
from pathlib import Path
import json
import numpy as np

from scripts.sensor_learned_fusion_data_20261023 import read, save, sha
from scripts.sensor_xgb_b_fusion_20261022 import temper, fuse
from scripts.report_zero_head_replication_20261024 import read_probability
from scripts.report_tree_anchored_residual_20261024 import metrics
from scripts.report_convtran_context_20261022 import CLASSES, write_csv


def choose_epochs(means, endpoints):
    """All entries are exact Fractions; selection is EMA only."""
    best = {p: max(endpoints, key=lambda e: (means[e, p], -e)) for p in ['TS', 'OS']}
    average = max(endpoints, key=lambda e: (means[e, 'TS'] + means[e, 'OS'], -e))
    eligible = [e for e in endpoints if all(means[e, p] >= means[20, p] for p in ['TS', 'OS'])]
    shared = max(eligible, key=lambda e: (min(means[e, p] - means[20, p] for p in ['TS', 'OS']),
                                          sum(means[e, p] - means[20, p] for p in ['TS', 'OS']), -e))
    return dict(best_epoch_by_protocol=best, best_equal_protocol_average_epoch=average,
                shared_recommended_epoch=shared, shared_eligible_epochs=eligible)


def report(cfg):
    out, parent = Path(cfg['output']), Path(cfg['parent'])
    assert not (out / 'PREDICTIONS_FROZEN.json').exists()
    frozen, cases, losses = {}, {}, []
    for p in cfg['protocols']:
        for k in cfg['inners']:
            base = parent / 'baseline' / p / f'inner{k}'
            px = read_probability(base / 'XGB.npy')
            frozen[str(base / 'XGB.npy')] = sha(base / 'XGB.npy')
            case = {}
            for seed in cfg['seeds']:
                folder = out / 'trajectories' / p / f'inner{k}' / f'seed{seed}'
                done = read(folder / 'COMPLETE.json')
                assert (done['status'], done['protocol'], done['inner'], done['seed'], done['final_epoch'], done['new_epochs']) == ('COMPLETE', p, k, seed, 100, 80)
                assert done['source_unchanged'] and done['optimizer_EMA_RNG_not_reset']
                assert sha(folder/'history.json') == done['history_sha256'] and sha(folder/'latest.pt') == done['latest_sha256']
                for name in ['history.json', 'latest.pt', 'COMPLETE.json']:
                    frozen[str(folder/name)] = sha(folder/name)
                history = read(folder/'history.json')
                assert [r['epoch'] for r in history] == list(range(1, 101))
                assert all(np.isfinite(r['train_ce']) and r['learning_rate'] == .001 for r in history)
                losses += [dict(protocol=p, inner=k, seed=seed, **row) for row in history]
                case['XGB', 'ema', seed] = px
                for epoch in cfg['endpoints']:
                    d = folder / f'epoch{epoch}'
                    assert sha(d/'COMPLETE.json') == done['endpoints'][str(epoch)]
                    record = read(d/'COMPLETE.json')
                    assert record['status'] == 'COMPLETE' and record['epoch'] == epoch and record['rng_preserved'] and record['serialized_raw_EMA_exact']
                    for name, expected in record['files'].items():
                        assert sha(name) == expected, name
                        frozen[name] = expected
                    frozen[str(d/'COMPLETE.json')] = sha(d/'COMPLETE.json')
                    for view in ['ema', 'raw']:
                        prob = read_probability(d/f'held_{view}.npy')
                        gate = fuse(temper(px, 2.), temper(prob, 1.), 'margin_select', .7)
                        path = out/'predictions'/p/f'inner{k}'/f'Fusion_epoch{epoch}_{view}_{seed}.npy'
                        assert not path.exists()
                        path.parent.mkdir(parents=True,exist_ok=True);np.save(path,gate);frozen[str(path)] = sha(path)
                        case[f'B_epoch{epoch}',view,seed] = prob
                        case[f'Fusion_epoch{epoch}',view,seed] = gate
                        if epoch == 20 and view == 'ema':
                            original = base/f'Gate_{seed}.npy'
                            np.testing.assert_array_equal(gate,np.load(original,allow_pickle=False));frozen[str(original)] = sha(original)
            gold = parent/'data'/p/f'inner{k}'/'DEV_TARGETS.jsonl';frozen[str(gold)] = sha(gold);cases[p,k] = case
    save(out/'PREDICTIONS_FROZEN.json',dict(status='PASS',files=frozen,trajectories=18,endpoints=cfg['endpoints'],new_scores_before_freeze=0))
    values, details = {}, {}
    rows, classes, pairs = [], [], []
    for (p,k),case in cases.items():
        gold = parent/'data'/p/f'inner{k}'/'DEV_TARGETS.jsonl'
        truth = np.asarray([json.loads(line)['label'] for line in gold.read_text().splitlines()])
        for (method,view,seed),prob in case.items():
            result,exact,_ = metrics(truth,prob);values[p,k,method,view,seed] = exact;details[p,k,method,view,seed] = result
            rows.append(dict(protocol=p,inner=k,method=method,view=view,seed=seed,**result))
            for i,c in enumerate(CLASSES):
                classes.append(dict(protocol=p,inner=k,method=method,view=view,seed=seed,behavior=c,F1_percent=result['class_f1_percent'][i],
                                    precision_percent=result['precision_percent'][i],recall_percent=result['recall_percent'][i],support=result['support'][i]))
        for e in cfg['endpoints']:
            for seed in cfg['seeds']:
                for method,reference in [(f'Fusion_epoch{e}','Fusion_epoch20'),(f'Fusion_epoch{e}','XGB'),(f'B_epoch{e}','B_epoch20')]:
                    pairs.append(dict(protocol=p,inner=k,seed=seed,epoch=e,method=method,reference=reference,
                        delta_F1_pp=float(values[p,k,method,'ema',seed]-values[p,k,reference,'ema',seed])))
    specs = [('XGB','ema')]+[(f'{kind}_epoch{e}',v) for e in cfg['endpoints'] for kind in ['B','Fusion'] for v in ['ema','raw']]
    summary, means = {}, {}
    for method,view in specs:
        name = method if view == 'ema' else method+'_raw';summary[name] = {}
        for p in cfg['protocols']:
            inner = [sum((values[p,k,method,view,s] for s in cfg['seeds']),Fraction())/3 for k in cfg['inners']]
            mean = sum(inner,Fraction())/3;means[name,p] = mean
            ms = [details[p,k,method,view,s] for k in cfg['inners'] for s in cfg['seeds']]
            summary[name][p] = dict(mean_percent=float(mean),per_inner_mean_percent=[float(x) for x in inner],
                class_f1_percent=dict(zip(CLASSES,np.mean([m['class_f1_percent'] for m in ms],axis=0).tolist())))
    decision = choose_epochs({(e,p):means[f'Fusion_epoch{e}',p] for e in cfg['endpoints'] for p in cfg['protocols']},cfg['endpoints'])
    epoch_table, stability = [], {}
    for e in cfg['endpoints']:
        record = dict(epoch=e);stability[str(e)] = {}
        for p in cfg['protocols']:
            name = f'Fusion_epoch{e}'
            record[p+'_B_F1_percent'] = float(means[f'B_epoch{e}',p]);record[p+'_Fusion_F1_percent'] = float(means[name,p])
            record[p+'_delta_epoch20_pp'] = float(means[name,p]-means['Fusion_epoch20',p]);record[p+'_delta_XGB_pp'] = float(means[name,p]-means['XGB',p])
            ds = [values[p,k,name,'ema',s]-values[p,k,'Fusion_epoch20','ema',s] for k in cfg['inners'] for s in cfg['seeds']]
            dx = [values[p,k,name,'ema',s]-values[p,k,'XGB','ema',s] for k in cfg['inners'] for s in cfg['seeds']]
            inner_wins = sum(sum(ds[3*k:3*k+3],Fraction())>0 for k in cfg['inners'])
            stability[str(e)][p] = dict(delta_epoch20_pp=record[p+'_delta_epoch20_pp'],delta_XGB_pp=record[p+'_delta_XGB_pp'],
                 positive_pairs_epoch20=sum(d>0 for d in ds),positive_pairs_XGB=sum(d>0 for d in dx),positive_inner_means=inner_wins,
                 stable_gain_epoch20=sum(ds,Fraction())>0 and sum(d>0 for d in ds)>=6 and inner_wins>=2,
                 stable_gain_XGB=sum(dx,Fraction())>0 and sum(d>0 for d in dx)>=6)
        stability[str(e)]['OS_target_classes_preserved'] = all(summary[f'Fusion_epoch{e}']['OS']['class_f1_percent'][c]>=summary['XGB']['OS']['class_f1_percent'][c] for c in ['licking','drinking'])
        epoch_table.append(record)
    selected = decision['shared_recommended_epoch']
    result = dict(status='COMPLETE_STOP_FOR_REVIEW',scope=cfg['scope'],primary='EMA registered endpoints',summary=summary,
        epoch_comparison=epoch_table,selection=decision,stability=stability,
        shared_has_stable_gain=selected!=20 and all(stability[str(selected)][p]['stable_gain_epoch20'] for p in cfg['protocols']),
        shared_exceeds_XGB_both_protocols=all(stability[str(selected)][p]['delta_XGB_pp']>0 for p in cfg['protocols']),
        tested_optimum_only=True,upper_bound_optimal_by_protocol={p:decision['best_epoch_by_protocol'][p]==100 for p in cfg['protocols']},
        new_scientific_trajectories=18,new_training_epochs=1440,reused_source_epochs=360,original_fivefold_unchanged=True,
        automatic_expansion=False,new_blind_evaluation=False)
    save(out/'SUMMARY.json',result);save(out/'PER_SEED_METRICS.json',rows)
    write_csv(out/'EPOCH_COMPARISON.csv',epoch_table);write_csv(out/'PER_CLASS_RESULTS.csv',classes);write_csv(out/'PAIRED_RESULTS.csv',pairs);write_csv(out/'TRAIN_LOSS_CURVES.csv',losses)
    lines=['# 原B连续训练轮数验证','','三个TRAIN内部开发inner、三种子，EMA主；非正式五折或新盲测。','',
           '| 总轮数 | TS融合F1/% | OS融合F1/% | TS较20轮/pp | OS较20轮/pp |','|---:|---:|---:|---:|---:|']
    for r in epoch_table:lines.append(f"| {r['epoch']} | {r['TS_Fusion_F1_percent']:.6f} | {r['OS_Fusion_F1_percent']:.6f} | {r['TS_delta_epoch20_pp']:+.6f} | {r['OS_delta_epoch20_pp']:+.6f} |")
    lines += ['',f'各协议开发最优轮：{decision["best_epoch_by_protocol"]}；共享推荐轮：{selected}。',
              f'共享轮次稳定增益条件：{result["shared_has_stable_gain"]}。仅在所测端点和恒定学习率范围内解释，不自动正式refit或扩大。']
    (out/'RESULTS_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return result
