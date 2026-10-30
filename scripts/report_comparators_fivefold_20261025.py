"""Evaluate all formal comparators only after all 150 jobs and predictions freeze."""
import csv
import json
from fractions import Fraction
from pathlib import Path
import numpy as np
from scripts.sensor_learned_fusion_data_20261023 import read,save,sha
from scripts.report_shared_fusion_fivefold_20261024 import evaluate,exact_decimal
from scripts.report_convtran_context_20261022 import CLASSES

def write_csv(p,rows):
    with Path(p).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def report(cfg):
    out=Path(cfg['output']);parent=Path(cfg['parent']);assert not (out/'PREDICTIONS_FROZEN.json').exists();frozen={};models=cfg['neural_models']+[cfg['cpu_model']]
    for p in cfg['protocols']:
        for f in cfg['folds']:
            for m in models:
                for s in cfg['seeds']:
                    d=out/'models'/p/f/m/f'seed{s}';r=read(d/'COMPLETE.json');assert (r['status'],r['protocol'],r['fold'],r['seed'],r['model'])==('COMPLETE',p,f,s,m)
                    assert r['Test_scores_computed']==0 and r['Test_labels_parsed'] is False
                    for name,h in r['files'].items():assert sha(name)==h,name;frozen[name]=h
                    frozen[str(d/'COMPLETE.json')]=sha(d/'COMPLETE.json')
                    if m in cfg['neural_models']:
                        assert r['epoch']==20 and r['raw_EMA_replay_exact'];hist=read(d/'HISTORY.json');assert [x['epoch'] for x in hist]==list(range(1,21)) and all(np.isfinite(x['fit_ce']) for x in hist)
                    else:assert r['features']==49890 and r['serialized_predictions_exact'] and r['relative_system_residual']<=1e-6
    for p in cfg['protocols']:
        for f in cfg['folds']:
            gold=parent/'data'/p/f/'TEST_TARGETS.jsonl';frozen[str(gold)]=sha(gold)
    save(out/'PREDICTIONS_FROZEN.json',dict(status='PASS',files=frozen,models=150,neural_models=120,rocket_models=30,all_predictions_frozen_before_Test_scores=True))
    rows=[];class_rows=[];values={};case_results={};legacy=read(parent/'CASE_RESULTS.json')
    for p in cfg['protocols']:
        for f in cfg['folds']:
            targets=[json.loads(x) for x in (parent/'data'/p/f/'TEST_TARGETS.jsonl').read_text().splitlines()];y=np.asarray([r['label'] for r in targets])
            for m in models:
                for s in cfg['seeds']:
                    d=out/'models'/p/f/m/f'seed{s}'
                    for v in (['raw','ema'] if m in cfg['neural_models'] else ['fixed_final']):
                        prob=np.load(d/f'test_{v}.npy') if v!='fixed_final' else np.eye(7)[np.load(d/'test_predictions.npy')-1]
                        metric,value=evaluate(y,prob);values[p,f,s,m,v]=value;case_results[p,f,s,m,v]=metric
                        rows.append(dict(protocol=p,fold=f,seed=s,method=m,view=v,primary=v==cfg['primary_views'][m],**metric))
                        for c,name in enumerate(CLASSES):class_rows.append(dict(protocol=p,fold=f,seed=s,method=m,view=v,behavior=name,F1_percent=metric['class_f1_percent'][c]))
    summary={};fold_rows=[];paired=[]
    for m in models:
        for v in (['raw','ema'] if m in cfg['neural_models'] else ['fixed_final']):
            key=m+'_'+v;summary[key]={}
            for p in cfg['protocols']:
                fm=[];cf=[]
                for f in cfg['folds']:
                    vals=[values[p,f,s,m,v] for s in cfg['seeds']];mean=sum(vals,Fraction())/3;fm.append(mean)
                    fold_rows.append(dict(protocol=p,fold=f,method=m,view=v,macro_f1_percent=float(mean),seed_sd_pp=float(np.std([float(x) for x in vals],ddof=1))))
                    cf.append(np.mean([case_results[p,f,s,m,v]['class_f1_percent'] for s in cfg['seeds']],axis=0))
                    for s in cfg['seeds']:
                        old=legacy[p][f][f'Fusion_ema_{s}'];cm=np.asarray(old['confusion_matrix']);oldfrac=sum((Fraction(2*int(cm[c,c]),int(cm[c].sum()+cm[:,c].sum())) if cm[c].sum()+cm[:,c].sum() else Fraction() for c in range(7)),Fraction())*100/7
                        paired.append(dict(protocol=p,fold=f,seed=s,method=m,view=v,comparator_minus_original_fusion_pp=float(values[p,f,s,m,v]-oldfrac)))
                mean=sum(fm,Fraction())/5
                summary[key][p]=dict(mean_percent=float(mean),unrounded_decimal=exact_decimal(mean),per_fold_percent=[float(x) for x in fm],fold_sample_sd_pp=float(np.std([float(x) for x in fm],ddof=1)),class_f1_percent=dict(zip(CLASSES,np.mean(cf,axis=0).tolist())))
    old_summary=read(parent/'SUMMARY.json');table={}
    for p in cfg['protocols']:
        table[p]={'Historical_GRU':old_summary['protocols'][p]['Historical_GRU']['mean_percent']}
        for m in models:table[p][m]=summary[m+'_'+cfg['primary_views'][m]][p]['mean_percent']
        table[p]['Original_Shared_Fusion']=old_summary['protocols'][p]['Fusion_ema']['mean_percent']
    result=dict(status='COMPLETE_STOP_FOR_REVIEW',main_table_percent=table,summary=summary,primary_views=cfg['primary_views'],view_selection_source=str(out/'VIEW_SELECTION.json'),fit_scope=cfg['fit_scope'],aggregation=cfg['aggregation'],new_fits=150,protocols=2,folds=5,seeds=3,Test_used_for_selection=False,historical_Test_exposure=True,new_blind_evaluation=False,automatic_method_change=False)
    save(out/'SUMMARY.json',result);save(out/'PER_SEED_METRICS.json',rows);write_csv(out/'PER_FOLD.csv',fold_rows);write_csv(out/'PER_CLASS.csv',class_rows);write_csv(out/'PAIRED_VS_FUSION.csv',paired)
    lines=['# 五种对照正式五折：完成','','同原共享融合TRAIN＋VAL复拟合口径；四神经主视图EMA20，raw20单列，MultiRocket固定最终解。','', '|方法|TS宏F1/%|OS宏F1/%|','|---|---:|---:|']
    for m in table['TS']:lines.append(f'|{m}|{table["TS"][m]:.6f}|{table["OS"][m]:.6f}|')
    lines+=['','每折三种子先平均，再五折等权平均。视图由既有开发结果先行固定，未按本轮Test选择。历史Test已有曝光，不称全新盲测。','增强XGB和B单独继续作为原组件消融，不加入本主表；原融合保持原封存版本。']
    (out/'RESULTS_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return result
