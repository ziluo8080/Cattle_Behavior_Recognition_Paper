"""Stream all frozen dense checkpoints into curves, then select development stops."""
from fractions import Fraction
from pathlib import Path
import json
import numpy as np

from scripts.sensor_learned_fusion_data_20261023 import read,save,sha
from scripts.sensor_xgb_b_fusion_20261022 import temper,fuse
from scripts.report_zero_head_replication_20261024 import read_probability
from scripts.report_tree_anchored_residual_20261024 import metrics
from scripts.report_convtran_context_20261022 import CLASSES,write_csv

ARMS=['Constant','Cosine']


def choose_stops(means,epochs):
    candidates=[(a,e) for a in ARMS for e in epochs]
    simpler=lambda a,e:(-e,-ARMS.index(a))
    best={p:max(candidates,key=lambda ae:(means[ae[0],ae[1],p],*simpler(*ae))) for p in ['TS','OS']}
    average=max(candidates,key=lambda ae:(sum(means[ae[0],ae[1],p] for p in ['TS','OS']),*simpler(*ae)))
    eligible=[ae for ae in candidates if all(means[ae[0],ae[1],p]>=means['Constant',20,p] for p in ['TS','OS'])]
    shared=max(eligible,key=lambda ae:(min(means[ae[0],ae[1],p]-means['Constant',20,p] for p in ['TS','OS']),
               sum(means[ae[0],ae[1],p]-means['Constant',20,p] for p in ['TS','OS']),*simpler(*ae)))
    pack=lambda ae:dict(arm=ae[0],stop_epoch=ae[1],schedule_horizon_epochs=40)
    return dict(best_by_protocol={p:pack(ae) for p,ae in best.items()},best_equal_protocol_average=pack(average),
                shared_recommendation=pack(shared),shared_eligible=[pack(ae) for ae in eligible])


def report(cfg):
    out,parent=Path(cfg['output']),Path(cfg['data_parent'])
    assert not (out/'PREDICTIONS_FROZEN.json').exists()
    frozen={};registry=[];train_rows=[]
    for p in cfg['protocols']:
        for k in cfg['inners']:
            baseline=parent/'baseline'/p/f'inner{k}';px=read_probability(baseline/'XGB.npy')
            frozen[str(baseline/'XGB.npy')]=sha(baseline/'XGB.npy')
            gold=parent/'data'/p/f'inner{k}'/'DEV_TARGETS.jsonl';frozen[str(gold)]=sha(gold)
            receipt=parent/'data'/p/f'inner{k}'/'RECEIPT.json';frozen[str(receipt)]=sha(receipt)
            for seed in cfg['seeds']:
                registry.append(dict(protocol=p,inner=k,seed=seed,method='XGB',view='ema',path=str(baseline/'XGB.npy')))
                for arm in cfg['arms']:
                    folder=out/'trajectories'/p/f'inner{k}'/arm/f'seed{seed}';done=read(folder/'COMPLETE.json')
                    assert (done['status'],done['protocol'],done['inner'],done['arm'],done['seed'],done['epoch'])==('COMPLETE',p,k,arm,seed,40)
                    for name,digest in done['files'].items():assert sha(name)==digest,name;frozen[name]=digest
                    frozen[str(folder/'COMPLETE.json')]=sha(folder/'COMPLETE.json')
                    h=read(folder/'history.json');assert [r['epoch'] for r in h]==list(range(1,41))
                    assert all(np.isfinite(r['train_ce']) for r in h);train_rows += [dict(protocol=p,inner=k,arm=arm,seed=seed,**r) for r in h]
                    for e in range(1,41):
                        d=folder/f'epoch{e}';record=read(d/'COMPLETE.json')
                        assert sha(d/'COMPLETE.json')==done['epoch_receipts'][str(e)]
                        assert record['rng_preserved'] and record['EMA_reload_exact'] and record['epoch']==e
                        for name,digest in record['files'].items():assert sha(name)==digest,name;frozen[name]=digest
                        frozen[str(d/'COMPLETE.json')]=sha(d/'COMPLETE.json')
                        if arm=='Constant' and e in [20,30,40]:
                            marker=d/'ORIGINAL_REPLAY.json';assert read(marker)['status']=='PASS';frozen[str(marker)]=sha(marker)
                        for view in (['ema','raw'] if e in cfg['raw_diagnostic_epochs'] else ['ema']):
                            prob=read_probability(d/f'held_{view}.npy')
                            gate=fuse(temper(px,2.),temper(prob,1.),'margin_select',.7)
                            path=out/'predictions'/p/f'inner{k}'/f'{arm}_Fusion_epoch{e}_{view}_{seed}.npy'
                            assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True);np.save(path,gate);frozen[str(path)]=sha(path)
                            registry.append(dict(protocol=p,inner=k,seed=seed,method=f'{arm}_B_epoch{e}',view=view,path=str(d/f'held_{view}.npy')))
                            registry.append(dict(protocol=p,inner=k,seed=seed,method=f'{arm}_Fusion_epoch{e}',view=view,path=str(path)))
                            if arm=='Constant' and e==20 and view=='ema':
                                original=baseline/f'Gate_{seed}.npy';np.testing.assert_array_equal(gate,np.load(original));frozen[str(original)]=sha(original)
    save(out/'PREDICTIONS_FROZEN.json',dict(status='PASS',files=frozen,new_models=36,dense_epochs=40,new_scores_before_freeze=0))
    values={};results={};rows=[];class_rows=[]
    truths={};weights={}
    for p in cfg['protocols']:
        for k in cfg['inners']:
            d=parent/'data'/p/f'inner{k}';truths[p,k]=np.asarray([json.loads(line)['label'] for line in (d/'DEV_TARGETS.jsonl').read_text().splitlines()])
            counts=np.asarray(read(d/'RECEIPT.json')['fit_counts']);weights[p,k]=np.sqrt(counts.sum()/(7*counts)).astype(np.float32).astype(np.float64)
    for item in registry:
        p,k,s,m,v=(item[n] for n in ['protocol','inner','seed','method','view'])
        prob=read_probability(item['path']);truth=truths[p,k];metric,exact,_=metrics(truth,prob)
        pt=prob[np.arange(len(truth)),truth-1];nll=-np.log(np.maximum(pt,np.finfo(np.float64).tiny))
        metric.update(weighted_CE=float(np.average(nll,weights=weights[p,k][truth-1])),NLL=float(nll.mean()),zero_true_probabilities=int((pt==0).sum()))
        values[p,k,m,v,s]=exact;results[p,k,m,v,s]=metric;rows.append(dict(protocol=p,inner=k,seed=s,method=m,view=v,**metric))
        for i,c in enumerate(CLASSES):class_rows.append(dict(protocol=p,inner=k,seed=s,method=m,view=v,behavior=c,F1_percent=metric['class_f1_percent'][i],precision_percent=metric['precision_percent'][i],recall_percent=metric['recall_percent'][i],support=metric['support'][i]))
    methods=[('XGB','ema')]+[(f'{a}_{kind}_epoch{e}',v) for a in ARMS for e in range(1,41) for kind in ['B','Fusion'] for v in (['ema','raw'] if e in cfg['raw_diagnostic_epochs'] else ['ema'])]
    summary={};means={}
    for m,v in methods:
        name=m if v=='ema' else m+'_raw';summary[name]={}
        for p in cfg['protocols']:
            inner=[sum((values[p,k,m,v,s] for s in cfg['seeds']),Fraction())/3 for k in cfg['inners']]
            mean=sum(inner,Fraction())/3;means[name,p]=mean
            ms=[results[p,k,m,v,s] for k in cfg['inners'] for s in cfg['seeds']]
            summary[name][p]=dict(mean_percent=float(mean),per_inner_mean_percent=[float(x) for x in inner],
                weighted_CE=float(np.mean([x['weighted_CE'] for x in ms])),NLL=float(np.mean([x['NLL'] for x in ms])),
                class_f1_percent=dict(zip(CLASSES,np.mean([x['class_f1_percent'] for x in ms],axis=0).tolist())))
    selection=choose_stops({(a,e,p):means[f'{a}_Fusion_epoch{e}',p] for a in ARMS for e in range(1,41) for p in cfg['protocols']},range(1,41))
    curves=[];pairs=[];stability={}
    for a in ARMS:
        for e in range(1,41):
            name=f'{a}_Fusion_epoch{e}';row=dict(arm=a,epoch=e);detail={}
            for p in cfg['protocols']:
                b=f'{a}_B_epoch{e}';ref='Constant_Fusion_epoch20'
                row.update({p+'_B_F1_percent':float(means[b,p]),p+'_Fusion_F1_percent':float(means[name,p]),p+'_DEV_B_EMA_CE':summary[b][p]['weighted_CE'],
                    p+'_delta_original20_pp':float(means[name,p]-means[ref,p]),p+'_delta_XGB_pp':float(means[name,p]-means['XGB',p])})
                ds=[values[p,k,name,'ema',s]-values[p,k,ref,'ema',s] for k in cfg['inners'] for s in cfg['seeds']]
                dx=[values[p,k,name,'ema',s]-values[p,k,'XGB','ema',s] for k in cfg['inners'] for s in cfg['seeds']]
                iw=sum(sum(ds[3*k:3*k+3],Fraction())>0 for k in cfg['inners'])
                detail[p]=dict(delta_original20_pp=float(sum(ds,Fraction())/9),delta_XGB_pp=float(sum(dx,Fraction())/9),positive_pairs_original20=sum(x>0 for x in ds),positive_pairs_XGB=sum(x>0 for x in dx),positive_inner_means=iw,
                    stable_gain=sum(ds,Fraction())>0 and sum(x>0 for x in ds)>=6 and iw>=2)
                for k in cfg['inners']:
                    for s in cfg['seeds']:pairs.append(dict(protocol=p,inner=k,seed=s,arm=a,epoch=e,delta_original20_pp=float(values[p,k,name,'ema',s]-values[p,k,ref,'ema',s]),delta_XGB_pp=float(values[p,k,name,'ema',s]-values[p,k,'XGB','ema',s])))
            detail['OS_target_classes_preserved']=all(summary[name]['OS']['class_f1_percent'][c]>=summary['XGB']['OS']['class_f1_percent'][c] for c in ['licking','drinking'])
            stability[f'{a}_{e}']=detail;curves.append(row)
    selected=selection['shared_recommendation'];key=f"{selected['arm']}_{selected['stop_epoch']}"
    result=dict(status='COMPLETE_STOP_FOR_REVIEW',scope=cfg['scope'],primary='EMA fused development macro-F1, epochs1-40',summary=summary,curves=curves,
        selection=selection,stability=stability,shared_has_stable_gain=all(stability[key][p]['stable_gain'] for p in cfg['protocols']),
        original_constant20_exact=True,original_constant30_40_exact=True,trajectories=36,training_epochs=1440,dense_primary_checkpoints=1440,
        dynamic_early_stopping_executed=False,development_checkpoint_selection=True,original_fivefold_unchanged=True,new_blind_evaluation=False,automatic_expansion=False)
    save(out/'SUMMARY.json',result);save(out/'PER_SEED_METRICS.json',rows)
    write_csv(out/'PER_EPOCH_CURVES.csv',curves);write_csv(out/'PER_CLASS_RESULTS.csv',class_rows);write_csv(out/'PAIRED_RESULTS.csv',pairs);write_csv(out/'TRAIN_LOSS_CURVES.csv',train_rows)
    lines=['# 原B早期曲线与余弦对照：完成','','仅历史TRAIN内部开发数据；完整40轮后选择checkpoint，不追认为原模型已执行早停。','',f'各协议开发最优：{selection["best_by_protocol"]}。',f'共享推荐：{selected}；稳定改善条件：{result["shared_has_stable_gain"]}。',
        '', 'Cosine若选择较早轮，保持40轮调度时钟，不能将衰减压缩到所选轮。原正式五折与20轮主线保留，不自动refit。', '密集曲线见PER_EPOCH_CURVES.csv，原始逐种子指标见PER_SEED_METRICS.json。']
    (out/'RESULTS_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return result
