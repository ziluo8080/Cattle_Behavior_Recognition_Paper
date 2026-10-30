"""Compare frozen lower-LR development runs, retaining all arms and policies."""
from pathlib import Path
import json,csv,statistics
from fractions import Fraction

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def write_csv(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def exact(cm):
    return sum((Fraction(2*cm[i][i],sum(cm[i])+sum(row[i] for row in cm)) if sum(cm[i])+sum(row[i] for row in cm) else Fraction() for i in range(7)),Fraction())/7*100
def hierarchical(items,key):
    inner=[]
    for k in [0,1,2]:
        values=[r[key] for r in items if r['inner']==k]
        assert len(values)==3
        if any(v is None for v in values):return None
        inner.append(statistics.mean(values))
    return statistics.mean(inner)

def report(cfg):
    out=Path(cfg['output']);curves=[];trains=[];cases=[];histories={}
    for p in cfg['protocols']:
        for k in cfg['inners']:
            for arm in cfg['arms']:
                for seed in cfg['seeds']:
                    d=out/'trajectories'/p/f'inner{k}'/arm/f'seed{seed}';done=read(d/'COMPLETE.json');h=read(d/'history.json')
                    assert done['status']=='COMPLETE' and [x['epoch'] for x in h]==list(range(1,21))
                    scores=[exact(x['dev_ema']['confusion_matrix']) for x in h];best=max(range(20),key=lambda i:(scores[i],-i))
                    assert h[best]['epoch']==done['best_epoch']==read(d/'BEST.json')['epoch']
                    histories[p,k,arm,seed]=h
                    for policy,i in [('epoch20',19),('neural_dev_best',best)]:
                        r=h[i];cases.append(dict(protocol=p,inner=k,arm=arm,seed=seed,policy=policy,epoch=r['epoch'],B_F1=r['dev_ema']['macro_f1_percent'],Fusion_F1=r['fusion_ema']['macro_f1_percent'],DEV_CE=r['dev_ema']['weighted_CE'],FIT_EMA_CE=r['fit_ema']['weighted_CE']))
                    trains.extend(dict(protocol=p,inner=k,arm=arm,seed=seed,epoch=r['epoch'],train_ce=r['train_ce'],learning_rate=r['learning_rate'],last_batch_gradient_l2=r['last_batch_gradient_l2']) for r in h)
    old=read(Path(cfg['reference'])/'PER_SEED_METRICS.json');index={(x['protocol'],x['inner'],x['seed'],x['method'],x['view']):x for x in old}
    for p in cfg['protocols']:
        for k in cfg['inners']:
            for seed in cfg['seeds']:
                h=[index[p,k,seed,f'Constant_B_epoch{e}','ema'] for e in range(1,21)]
                best=max(range(20),key=lambda i:(exact(h[i]['confusion_matrix']),-i))
                for policy,i in [('epoch20',19),('neural_dev_best',best)]:
                    b=h[i];g=index[p,k,seed,f'Constant_Fusion_epoch{i+1}','ema']
                    cases.append(dict(protocol=p,inner=k,arm='Constant_1e3',seed=seed,policy=policy,epoch=i+1,B_F1=b['macro_f1_percent'],Fusion_F1=g['macro_f1_percent'],DEV_CE=b['weighted_CE'],FIT_EMA_CE=None))
    for arm in cfg['arms']:
        for epoch in range(1,21):
            row=dict(arm=arm,epoch=epoch)
            for p in cfg['protocols']:
                items=[dict(inner=k,B_F1=histories[p,k,arm,s][epoch-1]['dev_ema']['macro_f1_percent'],Fusion_F1=histories[p,k,arm,s][epoch-1]['fusion_ema']['macro_f1_percent'],CE=histories[p,k,arm,s][epoch-1]['dev_ema']['weighted_CE']) for k in cfg['inners'] for s in cfg['seeds']]
                row[p+'_B_F1_percent']=hierarchical(items,'B_F1');row[p+'_Fusion_F1_percent']=hierarchical(items,'Fusion_F1');row[p+'_DEV_B_EMA_CE']=hierarchical(items,'CE')
            curves.append(row)
    summary={};pairs=[]
    for policy in ['epoch20','neural_dev_best']:
        summary[policy]={}
        for p in cfg['protocols']:
            summary[policy][p]={}
            baseline={(r['inner'],r['seed']):r for r in cases if r['protocol']==p and r['policy']==policy and r['arm']=='Constant_1e3'}
            for arm in ['Constant_1e3']+list(cfg['arms']):
                items=[r for r in cases if r['protocol']==p and r['policy']==policy and r['arm']==arm]
                result={name:hierarchical(items,name) for name in ['B_F1','Fusion_F1','DEV_CE','FIT_EMA_CE']}
                differences=[dict(inner=r['inner'],seed=r['seed'],difference=r['Fusion_F1']-baseline[r['inner'],r['seed']]['Fusion_F1']) for r in items]
                inner=[statistics.mean([r['difference'] for r in differences if r['inner']==k]) for k in cfg['inners']]
                result.update(fusion_delta_original_pp=hierarchical(differences,'difference'),positive_seed_inner_pairs=sum(r['difference']>0 for r in differences),positive_inner_means=sum(v>0 for v in inner),per_inner_delta_pp=inner,selected_epochs=[r['epoch'] for r in items])
                summary[policy][p][arm]=result
                pairs.extend(dict(policy=policy,protocol=p,arm=arm,**r) for r in differences)
    write_csv(out/'PER_EPOCH_CURVES.csv',curves);write_csv(out/'TRAIN_LOSS_CURVES.csv',trains);write_csv(out/'CHECKPOINT_COMPARISON.csv',cases);write_csv(out/'PAIRED_COMPARISON.csv',pairs)
    result=dict(status='COMPLETE_DEVELOPMENT_COMPARISON',registered_trajectories=36,epochs=720,summary=summary,scope=cfg['scope'],notes=['Fixed final-epoch and identically selected neural-DEV-best policies reported separately.','Old baseline same-model FIT-EMA CE not recorded; missing is not imputed from online training loss.','Smaller jitter or best checkpoint performance alone does not certify convergence.','No manuscript or official fivefold result was replaced.'])
    save(out/'SUMMARY.json',result);return result

if __name__=='__main__':report(read('configs/sensor_b_low_lr_20261027_v1.json'))
