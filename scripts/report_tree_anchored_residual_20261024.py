"""One fixed final scoring pass on the original TRAIN-internal held targets."""
from fractions import Fraction
from pathlib import Path
import json
import numpy as np

from scripts.tree_anchored_common_20261024 import read,save,sha,check_receipt
from scripts.sensor_xgb_b_fusion_20261022 import validate_probabilities,temper,fuse
from scripts.report_convtran_context_20261022 import CLASSES,write_csv


def metrics(truth,probability):
    p=validate_probabilities(probability);pred=p.argmax(1)+1
    cm=np.bincount((truth-1)*7+pred-1,minlength=49).reshape(7,7)
    scores=[]
    for i in range(7):
        den=int(cm[i,:].sum()+cm[:,i].sum());scores.append(Fraction(2*int(cm[i,i]),den) if den else Fraction())
    macro=sum(scores,Fraction())/7*100
    precision=np.divide(np.diag(cm),cm.sum(0),out=np.zeros(7),where=cm.sum(0)>0)*100
    recall=np.divide(np.diag(cm),cm.sum(1),out=np.zeros(7),where=cm.sum(1)>0)*100
    return dict(macro_f1_percent=float(macro),class_f1_percent=[float(v*100) for v in scores],precision_percent=precision.tolist(),
        recall_percent=recall.tolist(),confusion_matrix=cm.tolist(),support=cm.sum(1).tolist()),macro,scores


def report(cfg,state):
    out=Path(cfg['output']);assert len(state['jobs'])==22 and all(j['status']=='completed' for j in state['jobs'])
    frozen={};predictions={}
    for protocol in cfg['protocols']:
        check_receipt(out/'data'/protocol);case={}
        for name,digest in read(out/'baseline'/protocol/'REPLAY.json')['files'].items():assert sha(name)==digest;frozen[name]=digest
        px=validate_probabilities(np.load(out/'baseline'/protocol/'XGB.npy',allow_pickle=False))
        for seed in cfg['seeds']:
            for name,kind in [('OriginalGate','Gate'),('OriginalB','B')]:
                case[name,'ema',seed]=np.load(out/'baseline'/protocol/f'{kind}_{seed}.npy',allow_pickle=False)
            case['XGB','ema',seed]=px
            for arm in cfg['arms']:
                folder=out/'neural'/protocol/arm/f'seed{seed}';complete=read(folder/'COMPLETE.json')
                assert (complete['protocol'],complete['arm'],complete['seed'],complete['epoch'])==(protocol,arm,seed,20)
                assert sha(folder/'latest.pt')==complete['checkpoint_sha256'] and sha(folder/'history.json')==complete['history_sha256']
                h=read(folder/'history.json');assert [r['epoch'] for r in h]==list(range(1,21)) and all(np.isfinite(r['Fit_CE']) and np.isfinite(r['Fit_KL']) for r in h)
                frozen.update({str(folder/'latest.pt'):complete['checkpoint_sha256'],str(folder/'history.json'):complete['history_sha256'],str(folder/'COMPLETE.json'):sha(folder/'COMPLETE.json')})
                for view in ['ema','raw']:
                    r=complete['predictions'][view];assert sha(r['path'])==r['sha256'];frozen[r['path']]=r['sha256']
                    probability=validate_probabilities(np.load(r['path'],allow_pickle=False));case[arm,view,seed]=probability
                    if arm=='ZeroHead_B':
                        gate=fuse(temper(px,2.),temper(probability,1.),'margin_select',.7)
                        path=out/'predictions'/protocol/f'ZeroHeadGate_{view}_{seed}.npy';path.parent.mkdir(parents=True,exist_ok=True)
                        assert not path.exists();np.save(path,gate);frozen[str(path)]=sha(path);case['ZeroHeadGate',view,seed]=gate
        truth=out/'data'/protocol/'DEV_TARGETS.jsonl';frozen[str(truth)]=sha(truth);predictions[protocol]=case
    save(out/'PREDICTIONS_FROZEN.json',dict(status='PASS',files=frozen,registered_neural_models=18,registered_teacher_models=4,held_scores_before_freeze=0))
    results={};fractions={};rows=[];paired=[];classes=[];metric_rows=[];class_errors=[];keep_vs_ce=[]
    for protocol,case in predictions.items():
        targets=[json.loads(line) for line in (out/'data'/protocol/'DEV_TARGETS.jsonl').read_text().splitlines()]
        truth=np.asarray([r['label'] for r in targets]);assert np.isin(truth,np.arange(1,8)).all()
        results[protocol]={};fractions[protocol]={}
        for (arm,view,seed),probability in case.items():
            metric,value,class_fractions=metrics(truth,probability);results[protocol][arm,view,seed]=metric;fractions[protocol][arm,view,seed]=(value,class_fractions)
            metric_rows.append(dict(protocol=protocol,method=arm,view=view,seed=seed,**metric))
            rows.append(dict(protocol=protocol,method=arm,view=view,seed=seed,macro_f1_percent=float(value),**dict(zip(CLASSES,metric['class_f1_percent']))))
        for arm in cfg['arms']:
            for seed in cfg['seeds']:
                a=case[arm,'ema',seed].argmax(1)+1;x=case['XGB','ema',seed].argmax(1)+1
                m=results[protocol][arm,'ema',seed];base=results[protocol]['XGB','ema',seed]
                paired.append(dict(protocol=protocol,method=arm,seed=seed,
                    delta_original_gate_pp=m['macro_f1_percent']-results[protocol]['OriginalGate','ema',seed]['macro_f1_percent'],
                    delta_XGB_pp=m['macro_f1_percent']-base['macro_f1_percent'],
                    delta_ZeroHeadGate_pp=m['macro_f1_percent']-results[protocol]['ZeroHeadGate','ema',seed]['macro_f1_percent'],
                    corrected_XGB_errors=int(((x!=truth)&(a==truth)).sum()),damaged_XGB_correct=int(((x==truth)&(a!=truth)).sum())))
                for i,c in enumerate(CLASSES,1):
                    trueclass=truth==i
                    class_errors.append(dict(protocol=protocol,method=arm,seed=seed,behavior=c,
                        corrected_XGB_errors_trueclass=int(((x!=truth)&(a==truth)&trueclass).sum()),
                        damaged_XGB_correct_trueclass=int(((x==truth)&(a!=truth)&trueclass).sum()),
                        removed_XGB_false_positives=int(((x==i)&(truth!=i)&(a!=i)).sum()),
                        introduced_false_positives=int(((x!=i)&(truth!=i)&(a==i)).sum())))
        for seed in cfg['seeds']:
            ce=results[protocol]['TreeResidual_CE','ema',seed];keep=results[protocol]['TreeResidual_KeepCorrect','ema',seed]
            a=case['TreeResidual_CE','ema',seed].argmax(1)+1;b=case['TreeResidual_KeepCorrect','ema',seed].argmax(1)+1
            x=case['XGB','ema',seed].argmax(1)+1
            keep_vs_ce.append(dict(protocol=protocol,seed=seed,delta_macro_F1_pp=keep['macro_f1_percent']-ce['macro_f1_percent'],
                **{c:keep['class_f1_percent'][i]-ce['class_f1_percent'][i] for i,c in enumerate(CLASSES)},
                XGB_correct_damaged_CE=int(((x==truth)&(a!=truth)).sum()),XGB_correct_damaged_KeepCorrect=int(((x==truth)&(b!=truth)).sum()),
                CE_errors_corrected_by_KeepCorrect=int(((a!=truth)&(b==truth)).sum()),CE_correct_damaged_by_KeepCorrect=int(((a==truth)&(b!=truth)).sum())))
    summary={};exact_means={}
    methods=[(n,'ema') for n in ['OriginalGate','XGB','OriginalB']]+[(n,v) for n in cfg['arms']+['ZeroHeadGate'] for v in ['ema','raw']]
    for method,view in methods:
        name=method if view=='ema' else method+'_raw';summary[name]={};exact_means[name]={}
        for protocol in cfg['protocols']:
            values=[fractions[protocol][method,view,s][0] for s in cfg['seeds']];mean=sum(values,Fraction())/3
            exact_means[name][protocol]=mean
            ms=[results[protocol][method,view,s] for s in cfg['seeds']]
            cf=np.mean([m['class_f1_percent'] for m in ms],axis=0)
            summary[name][protocol]=dict(mean_percent=float(mean),seed_sd_pp=float(np.std([float(v) for v in values],ddof=1)),
                per_seed_percent=[float(v) for v in values],class_f1_percent=dict(zip(CLASSES,cf.tolist())),
                precision_percent=dict(zip(CLASSES,np.mean([m['precision_percent'] for m in ms],axis=0).tolist())),
                recall_percent=dict(zip(CLASSES,np.mean([m['recall_percent'] for m in ms],axis=0).tolist())),support=ms[0]['support'])
            for i,c in enumerate(CLASSES):classes.append(dict(protocol=protocol,method=name,behavior=c,F1_percent=float(cf[i]),precision_percent=summary[name][protocol]['precision_percent'][c],recall_percent=summary[name][protocol]['recall_percent'][c],support=ms[0]['support'][i]))
    candidates={}
    for arm in cfg['arms'][1:]:
        details={};gains=[]
        for protocol in cfg['protocols']:
            mean=exact_means[arm][protocol];gate=exact_means['OriginalGate'][protocol];tree=exact_means['XGB'][protocol]
            subset=[r for r in paired if r['method']==arm and r['protocol']==protocol]
            gain=mean-max(gate,tree);gains.append(float(gain))
            details[protocol]=dict(delta_original_gate_pp=float(mean-gate),delta_XGB_pp=float(mean-tree),
                positive_original_gate_pairs=sum(r['delta_original_gate_pp']>0 for r in subset),positive_XGB_pairs=sum(r['delta_XGB_pp']>0 for r in subset),
                success=mean>gate and mean>tree and sum(r['delta_original_gate_pp']>0 for r in subset)>=2 and sum(r['delta_XGB_pp']>0 for r in subset)>=2)
        target_ok=all(summary[arm]['OS']['class_f1_percent'][c]>=summary['XGB']['OS']['class_f1_percent'][c] for c in ['licking','drinking'])
        primary=all(r['success'] for r in details.values())
        candidates[arm]=dict(protocols=details,both_protocol_macro_success=primary,OS_target_classes_preserved=target_ok,
            OS_target_problem_solved=primary and target_ok,eligible_for_followup=primary,
            min_stronger_baseline_gain_pp=min(gains),mean_stronger_baseline_gain_pp=float(np.mean(gains)))
    eligible=[a for a,r in candidates.items() if r['eligible_for_followup']]
    chosen=max(eligible,key=lambda a:(candidates[a]['min_stronger_baseline_gain_pp'],candidates[a]['mean_stronger_baseline_gain_pp'],-cfg['arms'].index(a))) if eligible else 'OriginalGate'
    result=dict(status='COMPLETE_STOP_FOR_REVIEW',primary='fixed_epoch20_EMA',scope=cfg['scope'],
        summary=summary,candidates=candidates,shared_best=chosen,new_fusion_supported=chosen!='OriginalGate',
        original_fivefold_method_unchanged=True,new_formal_Test_read=False,outer_VAL_used_for_fit_or_score=False,
        historical_data_exposure=True,new_blind_evaluation=False,neural_fits=18,nested_teacher_fits=4,
        total_neural_epochs=360,automatic_expansion=False,
        full_target_solution_supported=chosen!='OriginalGate' and candidates[chosen]['OS_target_problem_solved'],
        KeepCorrect_minus_CE=[r for r in keep_vs_ce])
    save(out/'SUMMARY.json',result);write_csv(out/'PER_SEED_RESULTS.csv',rows);write_csv(out/'PAIRED_RESULTS.csv',paired);write_csv(out/'PER_CLASS_RESULTS.csv',classes)
    save(out/'PER_SEED_METRICS.json',metric_rows);write_csv(out/'CLASS_ERROR_CHANGES.csv',class_errors);write_csv(out/'KEEP_VS_CE.csv',keep_vs_ce)
    lines=['# 树先验动态残差：TRAIN内部小试完成','','| 方法（EMA20） | TS宏F1/% | OS宏F1/% |','|---|---:|---:|']
    for name in ['OriginalGate','XGB','OriginalB','ZeroHead_B','ZeroHeadGate']+cfg['arms'][1:]:lines.append(f"| {name} | {summary[name]['TS']['mean_percent']:.6f} | {summary[name]['OS']['mean_percent']:.6f} |")
    lines += ['',f'预注册共享选择：{chosen}。',
        '仅原outer TRAIN内部固定inner2开发留出；不是原outer VAL、正式Test或新盲测。',
        'TS留出舔舐仅8条；XGBoost每协议一个固定full-Fit教师，跨种子引用不是独立树拟合。',
        '只胜过弱控制不代表融合成功；KeepCorrect机制须另看与CE的匹配差值和纠正/损伤，不能由单独涨点归因。',
        '旧原共享融合及其五折结果保留，本批不自动扩大或重新正式评分。']
    (out/'RESULTS_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return result
