"""Official fold/seed equal-weight metrics and predeclared paired contrasts."""
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.sensor_motion_architecture_formal_common import read,write,sha,metrics,f1,get_data


def summarize(cfg,jobs,receipt):
    root=Path(cfg['output']);result=dict(protocols={},TEST_scoring=True,new_blindness=False,audit_status='pending');groups=[];flows=[]
    for proto in cfg['protocols']:
        tables={f:get_data(cfg,proto+'/'+f)['val'] for f in cfg['folds']}
        blocks=np.unique(np.concatenate([v.timestamp.to_numpy()//300 for v in tables.values()]));nb=len(blocks)
        draw=np.random.default_rng(20261007).multinomial(nb,np.full(nb,1/nb),size=2000)
        cowdraw={f:np.random.default_rng(20261007+i).multinomial(2,[.5,.5],size=2000) for i,f in enumerate(cfg['folds'])}
        predictions={};eligible={}
        for fold,table in tables.items():
            old=pd.read_csv(Path(cfg['original_formal'][proto])/fold/'predictions.csv')
            pd.testing.assert_frame_equal(old[['cow_id','timestamp','label']],table[['cow_id','timestamp','label']])
            for seed in cfg['seeds']:
                key=(fold,seed);predictions.setdefault('RF',{})[key]=old.RF.to_numpy();predictions.setdefault('Base',{})[key]=old[f'NN_seed{seed}'].to_numpy()
                for arm in ['BiasPair','ContextPair','NoMotionPair','MotionPair']:
                    p=Path(cfg['reused_refiners'])/proto/fold/f'{arm}_seed{seed}'/'predictions.csv'
                    t=pd.read_csv(p);pd.testing.assert_frame_equal(t[['cow_id','timestamp','label']],table[['cow_id','timestamp','label']]);predictions.setdefault(arm,{})[key]=t.prediction.to_numpy()
                case=proto+'/'+fold
                for job in [j for j in jobs if j['protocol']==case and j['seed']==seed]:
                    p=root/case/f"{job['kind']}_seed{seed}";done=read(p/'COMPLETE.json')
                    for name,h in done['files'].items(): assert sha(p/name)==h
                    t=pd.read_csv(p/'predictions.csv');pd.testing.assert_frame_equal(t[['cow_id','timestamp','label']],table[['cow_id','timestamp','label']])
                    names=['global7','pair7','pair3'] if job['kind'] in ['R7Fine','R7Coarse'] else ['pair3'] if job['family']=='Refiner' else ['prediction']
                    for name in names:
                        method=job['kind']+'.'+name if len(names)>1 else job['kind'];predictions.setdefault(method,{})[key]=t[name].to_numpy();eligible[method]=eligible.get(method,True) and done.get('comparison_eligible',True)
        scores={};boots={};cowboots={}
        for method,pp in predictions.items():
            records=[];bb=[];cb=[]
            for (fold,seed),p in sorted(pp.items()):
                t=tables[fold];y=t.label.to_numpy();b=np.searchsorted(blocks,t.timestamp.to_numpy()//300)
                records.append(dict(fold=fold,seed=seed,scores=metrics(y,p)))
                cm=np.bincount(b*49+(y-1)*7+p-1,minlength=nb*49).reshape(nb,7,7)
                bb.append(f1(np.einsum('rb,bjk->rjk',draw,cm)).mean(1))
                cows=sorted(t.cow_id.unique());c=t.cow_id.to_numpy()
                for cow in cows:
                    keep=c==cow;groups.append(dict(protocol=proto,fold=fold,seed=seed,method=method,cow=int(cow),scores=metrics(y[keep],p[keep])))
                if proto=='OS':
                    assert len(cows)==2;ccm=np.stack([np.asarray(metrics(y[c==cow],p[c==cow])['CM']) for cow in cows]);cb.append(f1(np.einsum('rb,bjk->rjk',cowdraw[fold],ccm)).mean(1))
            full=len(records)==15
            pooled={str(seed):metrics(np.concatenate([tables[f].label.to_numpy() for f in cfg['folds']]),np.concatenate([pp[(f,seed)] for f in cfg['folds']])) for seed in cfg['seeds']} if full else {}
            scores[method]=dict(mean=float(np.mean([r['scores']['macro_f1'] for r in records])),f1=np.mean([r['scores']['f1'] for r in records],0).tolist(),cases=records,comparison_eligible=eligible.get(method,True) and full,
                seed_means={str(s):float(np.mean([r['scores']['macro_f1'] for r in records if r['seed']==s])) for s in cfg['seeds']},nonprimary_pooled_per_seed=pooled,
                evaluated_cases=len(records),RF_reference_repeated_across_seeds=method=='RF')
            boots[method]=np.mean(bb,0)
            if proto=='OS':cowboots[method]=np.mean(cb,0)
        contrasts=[('MotionPair',x) for x in ['RF','Base','BiasPair','ContextPair','NoMotionPair','LinearPair','SpeedThresholdPair','RecurrentOnlyMotion','CurrentOnlyMotion','FullUniformMotion','FullBalancedMotion','NoUWBMotion','NoHDMotion','NoAklMotion']]
        contrasts += [('Base',x) for x in ['RecurrentOnly','CurrentOnly','FullUniform','FullBalanced','NoUWB','NoHD','NoAkl']]
        contrasts += [('R7Fine.pair7','R7Fine.global7'),('R7Coarse.pair7','R7Coarse.global7'),('R7Coarse.pair3','R7Fine.pair3'),('MotionPair','R7Coarse.pair3')]
        for b in ['RecurrentOnly','CurrentOnly','FullUniform','FullBalanced','NoUWB','NoHD','NoAkl']:contrasts.append((b+'Motion',b))
        if 'GridIntervalPair' in scores:contrasts.append(('MotionPair','GridIntervalPair'))
        comparisons={}
        for a,b in contrasts:
            coverage_match=set(predictions[a])==set(predictions[b])
            ok=scores[a]['comparison_eligible'] and scores[b]['comparison_eligible'] and coverage_match
            delta=scores[a]['mean']-scores[b]['mean'] if coverage_match else None
            ci=np.quantile(boots[a]-boots[b],[.025,.975]).tolist() if ok else None
            pos=sum(scores[a]['seed_means'][str(s)]>scores[b]['seed_means'][str(s)] for s in cfg['seeds'])
            cmp=dict(delta=delta,CI95=ci,positive_protocol_seed_means=pos if coverage_match else None,comparison_eligible=ok,coverage_match=coverage_match,conditional_positive_support=bool(ok and delta>0 and ci[0]>0 and pos>=2),class_delta=(np.asarray(scores[a]['f1'])-scores[b]['f1']).tolist() if coverage_match else None)
            if proto=='OS' and ok:cmp['conditional_cow_CI95']=np.quantile(cowboots[a]-cowboots[b],[.025,.975]).tolist()
            comparisons[a+'__minus__'+b]=cmp
            for fold,seed in sorted(set(predictions[a])&set(predictions[b])):
                y=tables[fold].label.to_numpy();p,q=predictions[b][(fold,seed)],predictions[a][(fold,seed)];r=(p!=y)&(q==y);h=(p==y)&(q!=y);change=p!=q
                flows.append(dict(protocol=proto,fold=fold,seed=seed,contrast=a+'__minus__'+b,rescued=np.bincount(y[r],minlength=8)[1:].tolist(),harmed=np.bincount(y[h],minlength=8)[1:].tolist(),flow=np.bincount((y[change]-1)*49+(p[change]-1)*7+q[change]-1,minlength=343).reshape(7,7,7).tolist()))
        inter={}
        for base in ['RecurrentOnly','CurrentOnly']:
            value=scores['MotionPair']['mean']-scores['Base']['mean']-scores[base+'Motion']['mean']+scores[base]['mean']
            ci=np.quantile(boots['MotionPair']-boots['Base']-boots[base+'Motion']+boots[base],[.025,.975]).tolist();inter[base]=dict(delta=value,CI95=ci)
        result['protocols'][proto]=dict(scores=scores,comparisons=comparisons,interactions=inter,Grid_no_op_folds=[f for f in cfg['folds'] if receipt['grid'][proto+'/'+f]['no_op']])
    write(root/'verified_summary.json',result);write(root/'groups.json',groups);write(root/'error_flows.json',flows)
    lines=['# 原论文协议下的完整正式消融','','TS/OS各五折×三个seed；Fit为作者TRAIN＋VAL，最终Test矩阵一次统一计分，历史Test已曝光。RF固定每fold seed42作为重复配对参照，不是三次独立RF拟合。','']
    for proto,r in result['protocols'].items():
        lines += [f'## {proto}','','| 方法 | 五折三seed宏F1% | 有效完整比较 |','|---|---:|---|']
        for method,s in r['scores'].items():lines.append(f"| {method} | {s['mean']*100:.6f} | {s['comparison_eligible']} |")
        lines += ['','| 完整主线去一项 | Macro差值pp | 条件95%区间pp |','|---|---:|---|']
        for method in ['Base','RecurrentOnlyMotion','CurrentOnlyMotion','FullUniformMotion','FullBalancedMotion','NoUWBMotion','NoHDMotion','NoAklMotion','LinearPair','SpeedThresholdPair']:
            c=r['comparisons']['MotionPair__minus__'+method];interval='未决' if c['CI95'] is None else str([v*100 for v in c['CI95']]);lines.append(f"| {method} | {c['delta']*100:+.6f} | {interval} |")
        lines += ['','全部对比/逐类/逐fold/seed/cow、辅助pooled、错误流及交互保存于JSON，不隐去负值。']
    lines += ['','固定模型/名单条件区间，不恢复盲性、不声称新牛或场地总体泛化；多项区间未校正，不称全家族显著。Mask改变有效容量；统计改善不等于独特创新。']
    (root/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8');return result
