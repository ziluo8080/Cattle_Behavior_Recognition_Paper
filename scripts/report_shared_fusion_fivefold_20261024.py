"""Final frozen-prediction Test evaluation; no training or selection paths."""
import csv
from decimal import Decimal,localcontext
from fractions import Fraction
from pathlib import Path

import numpy as np

from scripts.shared_fusion_fivefold_common_20261024 import read,save,sha,case_dir,check_method_lock,check_data
from scripts.sensor_xgb_b_fusion_20261022 import validate_probabilities,temper,fuse
from scripts.report_convtran_context_20261022 import CLASSES,write_csv


def evaluate(truth,probability):
    probability=validate_probabilities(probability);prediction=probability.argmax(1)+1
    assert len(truth)==len(prediction) and np.isin(truth,range(1,8)).all()
    cm=np.zeros((7,7),np.int64);np.add.at(cm,(truth-1,prediction-1),1)
    fractions=[]
    for i in range(7):
        denominator=int(cm[i,:].sum()+cm[:,i].sum())
        fractions.append(Fraction(2*int(cm[i,i]),denominator) if denominator else Fraction(0))
    macro=sum(fractions,Fraction())/7*100
    return dict(macro_f1_percent=float(macro),class_f1_percent=[float(x*100) for x in fractions],
        confusion_matrix=cm.tolist(),targets=len(truth)),macro


def exact_decimal(value):
    with localcontext() as c:c.prec=60;return str(Decimal(value.numerator)/Decimal(value.denominator))


def report(cfg,queue):
    out=Path(cfg['output']);failed=[j for j in queue['jobs'] if j['status']!='completed']
    if len(queue['jobs'])!=40 or failed:
        result=dict(status='INCOMPLETE_STOP_FOR_REPAIR',failed_jobs=failed,Test_scores_computed=0,final1pp_established=False)
        save(out/'SUMMARY.json',result);return result
    check_method_lock(cfg,full_sources=True);frozen={};cases={}
    for protocol in cfg['protocols']:
        for fold in cfg['folds']:
            check_data(cfg,protocol,fold);directory=case_dir(cfg,protocol,fold)
            tree=out/'tree'/protocol/fold;tr=read(tree/'COMPLETE.json')
            assert (tr['protocol'],tr['fold'],tr['rounds'],tr['fit_count'],tr['features'])==(protocol,fold,500,1,866)
            assert tr['status']=='COMPLETE' and tr['Test_scores_computed']==0
            assert sha(tr['model_path'])==tr['model_sha256'] and sha(tr['probability_path'])==tr['probability_sha256']
            px=validate_probabilities(np.load(tr['probability_path'],allow_pickle=False))
            frozen.update({str(tree/'COMPLETE.json'):sha(tree/'COMPLETE.json'),tr['model_path']:tr['model_sha256'],tr['probability_path']:tr['probability_sha256']})
            case={'XGB':px}
            for seed in cfg['seeds']:
                bdir=out/'B'/protocol/fold/f'seed{seed}';br=read(bdir/'COMPLETE.json')
                assert (br['protocol'],br['fold'],br['seed'],br['epoch'])==(protocol,fold,seed,20)
                assert br['status']=='COMPLETE' and br['Test_scores_computed']==0
                assert sha(bdir/'latest.pt')==br['checkpoint_sha256'] and sha(bdir/'history.json')==br['history_sha256']
                history=read(bdir/'history.json');assert [r['epoch'] for r in history]==list(range(1,21))
                assert all(np.isfinite(r['Fit_CE']) for r in history)
                frozen.update({str(bdir/'COMPLETE.json'):sha(bdir/'COMPLETE.json'),str(bdir/'latest.pt'):br['checkpoint_sha256'],str(bdir/'history.json'):br['history_sha256']})
                for view in ('ema','raw'):
                    record=br['predictions'][view];assert sha(record['path'])==record['sha256']
                    pb=validate_probabilities(np.load(record['path'],allow_pickle=False));assert pb.shape==px.shape
                    fusion=fuse(temper(px,2.),temper(pb,1.),'margin_select',.7)
                    path=out/'predictions'/protocol/fold/f'seed{seed}_{view}.npy';path.parent.mkdir(parents=True,exist_ok=True)
                    if path.exists():np.testing.assert_array_equal(np.load(path,allow_pickle=False),fusion)
                    else:np.save(path,fusion)
                    frozen[record['path']]=record['sha256'];frozen[str(path)]=sha(path)
                    case[f'B_{view}_{seed}']=pb;case[f'Fusion_{view}_{seed}']=fusion
            frozen[str(directory/'TEST_TARGETS.jsonl')]=sha(directory/'TEST_TARGETS.jsonl');cases[protocol,fold]=case
    save(out/'PREDICTIONS_FROZEN.json',dict(status='PASS',files=frozen,completed_neural_models=30,completed_tree_models=10,
        all_Test_probabilities_frozen_before_new_Test_scores=True))
    records=[];results={};exact={};legacy={};paired=[]
    for protocol in cfg['protocols']:
        results[protocol]={};exact[protocol]={};legacy[protocol]={}
        for fold in cfg['folds']:
            target=case_dir(cfg,protocol,fold)/'TEST_TARGETS.jsonl'
            import json
            rows=[json.loads(line) for line in target.read_text().splitlines()]
            truth=np.asarray([r['label'] for r in rows],np.int64);case=cases[protocol,fold];fold_results={};fold_exact={}
            for seed in cfg['seeds']:
                for view in ('ema','raw'):
                    for method,key in [('Fusion',f'Fusion_{view}_{seed}'),('B',f'B_{view}_{seed}'),('XGB','XGB')]:
                        metric,fraction=evaluate(truth,case[key]);name=f'{method}_{view}_{seed}'
                        fold_results[name]=metric;fold_exact[name]=fraction
                        records.append(dict(protocol=protocol,fold=fold,seed=seed,method=method,view=view,
                            macro_f1_percent=metric['macro_f1_percent'],**dict(zip(CLASSES,metric['class_f1_percent']))))
                old_path=Path(cfg['historical_GRU'])/protocol/fold/'GRU'/f'seed{seed}'
                with (old_path/'predictions.csv').open(newline='') as f:old=list(csv.DictReader(f))
                np.testing.assert_array_equal([[int(r[k]) for k in ('cow_id','timestamp','label')] for r in old],
                    [[r[k] for k in ('cow_id','timestamp','label')] for r in rows])
                pred=np.asarray([int(r['prediction']) for r in old]);assert np.isin(pred,range(1,8)).all();one_hot=np.eye(7)[pred-1]
                metric,fraction=evaluate(truth,one_hot)
                expected=read(old_path/'result.json')['metrics']
                np.testing.assert_array_equal(metric['confusion_matrix'],expected['confusion_matrix'])
                assert abs(metric['macro_f1_percent']-expected['macro_f1']*100)<1e-12
                fold_results[f'Historical_GRU_{seed}']=metric;fold_exact[f'Historical_GRU_{seed}']=fraction
                paired.append(dict(protocol=protocol,fold=fold,seed=seed,
                    delta_pp=float(fold_exact[f'Fusion_ema_{seed}']-fraction),
                    **{c:fold_results[f'Fusion_ema_{seed}']['class_f1_percent'][i]-metric['class_f1_percent'][i] for i,c in enumerate(CLASSES)}))
            results[protocol][fold]=fold_results;exact[protocol][fold]=fold_exact
    threshold=read(cfg['acceptance_targets']);summary={};fold_rows=[];class_rows=[]
    for protocol in cfg['protocols']:
        summary[protocol]={}
        for method,view in [('Fusion','ema'),('Fusion','raw'),('B','ema'),('B','raw'),('XGB','ema'),('Historical_GRU',None)]:
            prefix=method if view is None else f'{method}_{view}'
            name=method if method=='XGB' else prefix
            per_fold=[];per_seed={s:[] for s in cfg['seeds']};class_fold=[];fold_fractions=[]
            for fold in cfg['folds']:
                keys=[f'{prefix}_{s}' for s in cfg['seeds']]
                mean_fraction=sum((exact[protocol][fold][k] for k in keys),Fraction())/len(keys)
                fold_fractions.append(mean_fraction);per_fold.append(float(mean_fraction))
                classes=np.mean([results[protocol][fold][k]['class_f1_percent'] for k in keys],axis=0);class_fold.append(classes)
                seed_values=[]
                for s,k in zip(cfg['seeds'],keys):per_seed[s].append(results[protocol][fold][k]['macro_f1_percent']);seed_values.append(results[protocol][fold][k]['macro_f1_percent'])
                fold_rows.append(dict(protocol=protocol,fold=fold,method=name,macro_f1_percent=float(mean_fraction),seed_sd_pp=float(np.std(seed_values,ddof=1))))
            overall=sum(fold_fractions,Fraction())/5
            entry=dict(mean_percent=float(overall),mean_percent_unrounded_decimal=exact_decimal(overall),
                per_fold_percent=per_fold,fold_sample_sd_pp=float(np.std(per_fold,ddof=1)),
                per_seed_fivefold_mean_percent={str(s):float(np.mean(v)) for s,v in per_seed.items()},
                class_f1_percent=dict(zip(CLASSES,np.mean(class_fold,axis=0).tolist())))
            summary[protocol][name]=entry
            for c,v in entry['class_f1_percent'].items():class_rows.append(dict(protocol=protocol,method=name,behavior=c,macro_averaged_class_F1_percent=v))
            if name=='Fusion_ema':
                threshold_decimal=Decimal(threshold['protocols'][protocol]['effective_acceptance_percent'])
                entry['exact_plus_one_threshold_percent']=str(threshold_decimal)
                entry['numeric_plus_one_passed']=overall>=Fraction(threshold_decimal)
                entry['gain_over_frozen_historical_GRU_pp']=float(overall-Fraction(Decimal(threshold['protocols'][protocol]['frozen_reference_percent'])))
        frozen_ref=Decimal(threshold['protocols'][protocol]['frozen_reference_percent'])
        assert abs(Decimal(summary[protocol]['Historical_GRU']['mean_percent_unrounded_decimal'])-frozen_ref)<Decimal('1e-11')
    final=dict(status='COMPLETE_STOP_FOR_REVIEW',protocols=summary,
        final1pp_established=all(summary[p]['Fusion_ema']['numeric_plus_one_passed'] for p in cfg['protocols']),
        protocols_completed=2,folds_per_protocol=5,neural_seeds_per_fold=3,deterministic_tree_fits_per_fold=1,
        aggregation=cfg['aggregation'],primary='Fusion_ema20',secondary='Fusion_raw20',
        Test_used_for_selection=False,Test_evaluated=True,formal_fit_scope=cfg['fit_scope'],
        historical_Test_exposure=True,new_blind_evaluation=False,automatic_method_change=False)
    save(out/'CASE_RESULTS.json',results);save(out/'SUMMARY.json',final)
    write_csv(out/'PER_SEED_RESULTS.csv',records);write_csv(out/'PER_FOLD_RESULTS.csv',fold_rows)
    write_csv(out/'PER_CLASS_RESULTS.csv',class_rows);write_csv(out/'PAIRED_VS_HISTORICAL_GRU.csv',paired)
    lines=['# 原共享融合：五折正式复评完成','','每折三种子先平均，再对五折等权平均；主终点固定EMA20。','',
        '| 方法 | TS宏F1/% | OS宏F1/% |','|---|---:|---:|']
    for name in ('Historical_GRU','XGB','B_ema','Fusion_ema','Fusion_raw'):
        lines.append(f"| {name} | {summary['TS'][name]['mean_percent']:.6f} | {summary['OS'][name]['mean_percent']:.6f} |")
    lines+=['',f"双协议相对固定历史GRU +1pp 数值验收：{final['final1pp_established']}。",'逐折/种子、七类以及配对差值见CSV。',
        '采用官方五折划分与历史GRU一致的TRAIN+VAL复拟合口径；目标身份完全相同。',
        '历史数据已参与开发，不能称为全新盲测。Test未用于本轮方法、参数、轮次或视图选择。']
    (out/'RESULTS_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return final
