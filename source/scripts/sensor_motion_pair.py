"""Cloud-only frozen-base, true coarse-label refinement; final seven-class scoring."""
import json, time, hashlib
from pathlib import Path
START, START_WALL = time.monotonic(), time.time()
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scripts.sensor_motion_pair_model import MotionPairRefiner, coarse_targets
from scripts.sensor_native_motion_features import native_motion, fit_motion_scaler, scale_motion
from scripts.sensor_local_tcn_residual import read, write, sha, f1, metrics, dataset

CONFIG = Path('configs/sensor_motion_pair_20261009.json')
ROOT = Path('/q3/mmcows-sensor-behavior-20261007')
NAMES = ['行走','站立','抬头采食','低头采食','舔舐','饮水','躺卧']


def budget(cfg, stage):
    elapsed = time.monotonic()-START
    if elapsed > cfg['maximum_wall_seconds']:
        write(Path(cfg['output'])/'PARTIAL.json', dict(stage=stage, seconds=elapsed, complete=False))
        raise RuntimeError('Budget exceeded')
    return elapsed


def provenance(cfg):
    parent = read(cfg['parent_configuration'])
    paths = [CONFIG, Path(__file__), Path('scripts/sensor_motion_pair_model.py'),
        Path('scripts/sensor_native_motion_features.py'), Path('scripts/sensor_local_tcn_residual.py'),
        Path('scripts/run_sensor_motion_pair.sh'), Path(cfg['parent_configuration']), Path(cfg['motion_input_admission'])]
    identity = {str(p):sha(p) for p in paths}
    admission=read(cfg['motion_input_admission'])
    assert admission['status']=='PASS'
    for check in admission['checks'].values():
        assert check['partial_coordinate_raw_rows']==0 and check['dt_matches'] and check['pair_flags_match']
    for path,h in admission['hashes'].items():
        if path.startswith('data/'):
            assert sha(path)==h,'Raw same-source admission data changed'
            identity[path]=h
    for proto in cfg['protocols']:
        cache = Path(parent['cache'])/proto
        receipt = read(cache/'COMPLETE.json')
        identity[str(cache/'COMPLETE.json')] = sha(cache/'COMPLETE.json')
        for path,h in receipt['identity']['sources'].items():
            assert sha(path)==h, 'Frozen parent source changed'
            identity[path]=h
        for name,key in [('targets.jsonl','target_sha256'),('normalizer.json','normalizer_sha256'),('fit_statistics.json','statistics_sha256')]:
            assert sha(cache/name)==receipt[key]
            identity[str(cache/name)]=receipt[key]
        for role in ['train','val']:
            for m in ['uwb','hd','akl']:
                for field in ['x','valid','fresh','age']:
                    name=f'{role}/{m}_clean_{field}.npy'
                    assert sha(cache/name)==receipt['arrays'][name]
                    identity[str(cache/name)]=receipt['arrays'][name]
        for seed in cfg['seeds']:
            old=Path(parent['output'])/proto/f'P0_seed{seed}'
            done=read(old/'COMPLETE.json')
            assert done['epochs']==20 and done['selection']=='fixed_final_epoch'
            assert sha(old/'model.pt')==done['model_sha256']
            for name in ['model.pt','COMPLETE.json','clean_predictions.csv']:
                identity[str(old/name)]=sha(old/name)
        budget(cfg,'provenance')
    return parent,identity


def data(parent,proto):
    train,val,stats,trpool,vpool=dataset(parent,proto)
    cache=Path(parent['cache'])/proto
    norm=read(cache/'normalizer.json')
    built=[]
    for rows,pool in [(train,trpool),(val,vpool)]:
        d=pool.arrays['uwb']['clean']
        built.append(native_motion(d['x'],d['valid'],d['age'],rows.timestamp.to_numpy(),rows.start.to_numpy(),norm['uwb']['std']))
    scaler=fit_motion_scaler(built[0][0],built[0][1])
    mt,mv=[scale_motion(features,pair,scaler) for features,pair,_ in built]
    return train,val,stats,trpool,vpool,mt,mv,scaler


def make(cfg,parent,proto,arm,seed,stats):
    old=Path(parent['output'])/proto/f'P0_seed{seed}'/'model.pt'
    ckpt=torch.load(old,map_location='cpu',weights_only=False)
    assert ckpt['fit']=='TRAIN_only' and ckpt['target_sha256']==sha(Path(parent['cache'])/proto/'targets.jsonl')
    return MotionPairRefiner(arm,stats['prior'],ckpt['state_dict'],seed+cfg['training']['branch_seed_offset']).cuda()


def loss_for(output, y7, weights):
    row_weights=weights[y7]
    return (F.cross_entropy(output['auxiliary_logits'],coarse_targets(y7),reduction='none')*row_weights).sum()/row_weights.sum()


def invariant(pred,base):
    for c in range(3,8):
        np.testing.assert_array_equal(pred==c,base==c)


def sanity(cfg,parent):
    train,_,stats,pool,_,mt,_,_=data(parent,'TS')
    batch=pool.batch(np.arange(32))
    motion=torch.from_numpy(mt[:32]).cuda()
    y=torch.tensor(train.label.to_numpy()[:32]-1,device='cuda')
    weights=torch.tensor(stats['CE_weights'],dtype=torch.float32,device='cuda')
    counts,states,records={},{},{}
    for arm in cfg['arms']:
        model=make(cfg,parent,'TS',arm,17,stats).train()
        assert not model.base.training and all(not p.requires_grad for p in model.base.parameters())
        with torch.no_grad():
            output=model(batch,motion)
            np.testing.assert_array_equal(output['prediction'].cpu().numpy(),output['base_logits'].argmax(1).cpu().numpy()+1)
        before={k:v.detach().cpu().clone() for k,v in model.base.state_dict().items()}
        counts[arm]=dict(total=sum(p.numel() for p in model.parameters()),trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
        params=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=.001,weight_decay=.01)
        output=model(batch,motion)
        loss=loss_for(output,y,weights)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None for p in model.base.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        g=float(sum(p.grad.square().sum() for p in params).sqrt())
        assert g>0
        optimizer.step()
        for k,v in model.base.state_dict().items():
            torch.testing.assert_close(v.detach().cpu(),before[k],atol=0,rtol=0)
        out=model(batch,motion)
        invariant(out['prediction'].detach().cpu().numpy(),out['base_logits'].argmax(1).cpu().numpy()+1)
        empty={m:{k:v.clone() for k,v in d.items()} for m,d in batch.items()}
        for d in empty.values():d['valid'].zero_()
        with torch.no_grad():
            fallback=model(empty,motion)
            torch.testing.assert_close(fallback['base_logits'],model.base.prior_logits[None].expand(32,-1),atol=0,rtol=0)
            assert torch.all(fallback['prediction']==model.base.prior_logits.argmax()+1)
        records[arm]=dict(loss=float(loss.detach()),gradient_norm=g,base_frozen_exact=True)
        if arm in ['MotionPair','NoMotionPair']:
            fresh=make(cfg,parent,'TS',arm,17,stats)
            states[arm]={k:v.detach().cpu().clone() for k,v in fresh.residual.state_dict().items()}
            del fresh
        del model
    for k,v in states['MotionPair'].items():torch.testing.assert_close(v,states['NoMotionPair'][k],atol=0,rtol=0)
    assert counts['MotionPair']['trainable']/86695<=cfg['parameter_budget']['maximum_candidate_increase_fraction']
    assert abs(counts['MotionPair']['total']-counts['ContextPair']['total'])/counts['MotionPair']['total']<=.01
    write(Path(cfg['output'])/'SANITY.json',dict(status='PASS',counts=counts,witness=records,sanity_optimizer_steps=4,
        initial_predictions_exact=True,protected_class_indicators_exact=True,Base_always_eval_no_grads=True))
    print('MOTION_PAIR_SANITY_PASS',counts,flush=True)


def train_one(cfg,parent,proto,arm,seed):
    directory=Path(cfg['output'])/proto/f'{arm}_seed{seed}'
    if directory.exists():raise RuntimeError('No fit restart')
    directory.mkdir(parents=True)
    write(directory/'STARTED.json',dict(protocol=proto,arm=arm,seed=seed,budget_elapsed=budget(cfg,'fit_start')))
    train,val,stats,trpool,vpool,mt,mv,scaler=data(parent,proto)
    write(directory/'motion_scaler.json',scaler)
    model=make(cfg,parent,proto,arm,seed,stats)
    torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},directory/'initial_state.pt')
    frozen={k:v.detach().cpu().clone() for k,v in model.base.state_dict().items()}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['training']['learning_rate'],weight_decay=cfg['training']['weight_decay'])
    y=torch.tensor(train.label.to_numpy()-1,device='cuda')
    weights=torch.tensor(stats['CE_weights'],dtype=torch.float32,device='cuda')
    motion=torch.from_numpy(mt)
    rng=np.random.default_rng(seed)
    history,order_hashes=[],[]
    torch.cuda.reset_peak_memory_stats()
    began=time.monotonic()
    for epoch in range(1,21):
        epoch_start=time.monotonic();model.train()
        order=rng.permutation(len(train))
        order_hashes.append(hashlib.sha256(order.astype('<i8').tobytes()).hexdigest())
        numerator=denominator=0.
        for at in range(0,len(order),cfg['training']['batch_size']):
            idx=order[at:at+cfg['training']['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            output=model(trpool.batch(idx),motion[idx].cuda())
            loss=loss_for(output,y[idx],weights)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite coarse CE')
            loss.backward()
            if any(p.grad is not None for p in model.base.parameters()):raise AssertionError('Frozen Base received gradient')
            optimizer.step()
            mass=float(weights[y[idx]].sum());numerator+=float(loss.detach())*mass;denominator+=mass
        history.append(dict(epoch=epoch,train_coarse_CE=numerator/denominator,epoch_seconds=time.monotonic()-epoch_start))
        write(directory/'history.json',history)
        budget(cfg,'epoch')
        print('MOTION_PAIR_EPOCH',proto,arm,seed,epoch,history[-1],flush=True)
        pilot=cfg['pilot']
        if (proto,arm,seed,epoch)==(pilot['protocol'],pilot['arm'],pilot['seed'],pilot['epochs']):
            cost=max(v['epoch_seconds'] for v in history[1:])
            forecast=budget(cfg,'pilot')+pilot['remaining_epochs']*cost*pilot['factor']+pilot['extra_overhead_seconds']
            write(Path(cfg['output'])/'PILOT.json',dict(forecast=forecast,remaining_admitted=forecast<=cfg['maximum_wall_seconds']))
            if forecast>cfg['maximum_wall_seconds']:
                torch.save(dict(state_dict=model.state_dict(),completed_epochs=epoch,resumable=False),directory/'partial_model.pt')
                raise RuntimeError('Budget insufficient')
    for k,v in model.base.state_dict().items():torch.testing.assert_close(v.detach().cpu(),frozen[k],atol=0,rtol=0)
    training_seconds=time.monotonic()-began
    model.eval();predictions=[];base_logits=[];aux_logits=[]
    with torch.no_grad():
        for at in range(0,len(val),cfg['training']['batch_size']):
            idx=np.arange(at,min(at+cfg['training']['batch_size'],len(val)))
            out=model(vpool.batch(idx),torch.from_numpy(mv[idx]).cuda())
            if not torch.isfinite(out['auxiliary_logits']).all():raise RuntimeError('Nonfinite auxiliary scores')
            predictions.append(out['prediction'].cpu().numpy());base_logits.append(out['base_logits'].cpu().numpy());aux_logits.append(out['auxiliary_logits'].cpu().numpy())
    pred,base,aux=np.concatenate(predictions),np.concatenate(base_logits),np.concatenate(aux_logits)
    old=pd.read_csv(Path(parent['output'])/proto/f'P0_seed{seed}'/'clean_predictions.csv')
    pd.testing.assert_frame_equal(old[['cow_id','timestamp','label']],val[['cow_id','timestamp','label']])
    np.testing.assert_array_equal(base.argmax(1)+1,old.prediction.to_numpy())
    invariant(pred,old.prediction.to_numpy())
    table=val[['cow_id','timestamp','label']].copy();table['prediction']=pred;table['base_prediction']=old.prediction.to_numpy()
    table.to_csv(directory/'predictions.csv',index=False)
    np.save(directory/'base_logits.npy',base);np.save(directory/'auxiliary_logits.npy',aux)
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},kind=arm,seed=seed,protocol=proto,
        prior=stats['prior'],fit='TRAIN_only',motion_scaler=scaler,Base_frozen=True),directory/'model.pt')
    files=['model.pt','initial_state.pt','history.json','predictions.csv','base_logits.npy','auxiliary_logits.npy','motion_scaler.json']
    record=dict(scores=metrics(val.label.to_numpy(),pred),order_hashes=order_hashes,epochs=20,training_seconds=training_seconds,
        total_seconds=time.monotonic()-began,parameters=sum(p.numel() for p in model.parameters()),trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),Base_frozen_exact=True,protected_indicators_exact=True,
        files={name:sha(directory/name) for name in files},TEST_scoring=False)
    write(directory/'COMPLETE.json',record)
    print('MOTION_PAIR_FIT_COMPLETE',proto,arm,seed,record['scores']['macro_f1'],flush=True)
    del model
    torch.cuda.empty_cache()


def summarize(cfg,parent):
    root=Path(cfg['output']);result=dict(protocols={},TEST_scoring=False,audit_status='pending');groups=[];flows=[]
    methods=['Base']+cfg['arms']
    for proto in cfg['protocols']:
        targets=pd.read_json(Path(parent['cache'])/proto/'targets.jsonl',lines=True,convert_dates=False)
        val=targets[targets.split.eq('val')].reset_index(drop=True);y=val.label.to_numpy()
        _,blocks=np.unique(val.timestamp.to_numpy()//300,return_inverse=True);nb=int(blocks.max())+1
        draws=np.random.default_rng(20261007).multinomial(nb,np.full(nb,1/nb),size=2000)
        scores,boot,preds={},{},{}
        for arm in methods:
            seed_metrics=[];bs=[];preds[arm]=[]
            for seed in cfg['seeds']:
                path=Path(parent['output'])/proto/f'P0_seed{seed}'/'clean_predictions.csv' if arm=='Base' else root/proto/f'{arm}_seed{seed}'/'predictions.csv'
                table=pd.read_csv(path);pd.testing.assert_frame_equal(table[['cow_id','timestamp','label']],val[['cow_id','timestamp','label']])
                pred=table.prediction.to_numpy();preds[arm].append(pred);seed_metrics.append(metrics(y,pred))
                cm=np.bincount(blocks*49+(y-1)*7+pred-1,minlength=nb*49).reshape(nb,7,7)
                bs.append(f1(np.einsum('rb,bjk->rjk',draws,cm)).mean(1))
                for cow in sorted(val.cow_id.unique()):
                    mask=val.cow_id.to_numpy()==cow
                    groups.append(dict(protocol=proto,arm=arm,seed=seed,cow=int(cow),scores=metrics(y[mask],pred[mask])))
            scores[arm]=dict(mean=float(np.mean([s['macro_f1'] for s in seed_metrics])),per_seed=[s['macro_f1'] for s in seed_metrics],
                            f1=np.mean([s['f1'] for s in seed_metrics],0).tolist(),seed_metrics=seed_metrics)
            boot[arm]=np.mean(bs,0)
        pairs={}
        for control in ['Base','BiasPair','ContextPair','NoMotionPair']:
            delta=scores['MotionPair']['mean']-scores[control]['mean']
            pairs[control]=dict(delta=delta,relative_gain=delta/scores[control]['mean'],
                CI95=np.quantile(boot['MotionPair']-boot[control],[.025,.975]).tolist(),
                class_delta=(np.asarray(scores['MotionPair']['f1'])-scores[control]['f1']).tolist(),
                per_seed_delta=(np.asarray(scores['MotionPair']['per_seed'])-scores[control]['per_seed']).tolist())
            for i,seed in enumerate(cfg['seeds']):
                p,q=preds[control][i],preds['MotionPair'][i];a=(p!=y)&(q==y);b=(p==y)&(q!=y)
                flows.append(dict(protocol=proto,seed=seed,control=control,rescued_by_class=np.bincount(y[a],minlength=8)[1:8].tolist(),
                    harmed_by_class=np.bincount(y[b],minlength=8)[1:8].tolist(),changed_flow=np.bincount((y[p!=q]-1)*49+(p[p!=q]-1)*7+q[p!=q]-1,minlength=343).reshape(7,7,7).tolist()))
        g=cfg['gates']
        tests=dict(base_gain=pairs['Base']['delta']>=g['candidate_macro_gain_vs_Base'],
            pair_classes=min(pairs['Base']['class_delta'][:2])>=-g['pair_class_drop_limit_vs_Base'],
            protected_classes=all(np.array_equal(preds['Base'][i]==c,preds['MotionPair'][i]==c) for i in range(3) for c in range(3,8)),
            bias_gain=pairs['BiasPair']['delta']>=g['candidate_macro_gain_vs_BiasPair'],
            context_gain=pairs['ContextPair']['delta']>=g['candidate_macro_gain_vs_ContextPair'],
            nomotion_gain=pairs['NoMotionPair']['delta']>=g['candidate_macro_gain_vs_NoMotionPair'],
            positive_CI=all(v['CI95'][0]>0 for v in pairs.values()),seed_consistency=sum(d>0 for d in pairs['Base']['per_seed_delta'])>=2)
        tests['passed']=all(tests.values());result['protocols'][proto]=dict(scores=scores,comparisons=pairs,gates=tests)
    result['adoptable']=all(v['gates']['passed'] for v in result['protocols'].values())
    result['status']='supported_pending_audit' if result['adoptable'] else 'method_failed_frozen_pending_audit'
    write(root/'verified_summary.json',result);write(root/'groups.json',groups);write(root/'error_flows.json',flows)
    lines=['# 原生运动类对细化模块验证','','Base冻结；全部原七类GT计分、无TEST评分。','']
    for proto,p in result['protocols'].items():
        lines += [f'## {proto}','','| 方法 | macroF1% | 三seed% |','|---|---:|---|']
        for arm,s in p['scores'].items():lines.append(f"| {arm} | {s['mean']*100:.6f} | "+'/'.join(f'{x*100:.6f}' for x in s['per_seed'])+' |')
        lines += ['','| 对照 | Motion增益pp | 95%区间pp |','|---|---:|---|']
        for c,d in p['comparisons'].items():lines.append(f"| {c} | {d['delta']*100:+.6f} | [{d['CI95'][0]*100:+.6f},{d['CI95'][1]*100:+.6f}] |")
        lines += ['','| 类别 | Base% | Motion% | 差值pp |','|---|---:|---:|---:|']
        for i,name in enumerate(NAMES):lines.append(f"| {name} | {p['scores']['Base']['f1'][i]*100:.6f} | {p['scores']['MotionPair']['f1'][i]*100:.6f} | {p['comparisons']['Base']['class_delta'][i]*100:+.6f} |")
        lines += ['','固定采用门：'+json.dumps(p['gates'],ensure_ascii=False),'']
    lines += [result['status'],'','原生表观位移率不是瞬时真速度；输入按当前数据同源完整观测前提核验，冻结Base和5类输出保持不保证1/2提升。既有曝光fold1/OS两牛/固定模型条件CI；不宣称原创或新场地泛化。']
    (root/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    return result


def main():
    if Path.cwd()!=ROOT or not torch.cuda.is_available():raise RuntimeError('Cloud CUDA required')
    cfg=read(CONFIG);root=Path(cfg['output'])
    if root.exists():raise RuntimeError('Preserve attempt, no reset')
    root.mkdir(parents=True);write(root/'BUDGET_START.json',dict(unix=START_WALL,limit=cfg['maximum_wall_seconds']))
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    try:
        parent,identity=provenance(cfg);write(root/'IDENTITY.json',dict(inputs=identity))
        sanity(cfg,parent)
        pilot=cfg['pilot'];train_one(cfg,parent,pilot['protocol'],pilot['arm'],pilot['seed'])
        for proto in cfg['protocols']:
            for seed in cfg['seeds']:
                for arm in cfg['arms']:
                    if (proto,arm,seed)!=(pilot['protocol'],pilot['arm'],pilot['seed']):train_one(cfg,parent,proto,arm,seed)
        result=summarize(cfg,parent)
        for path,h in identity.items():assert sha(path)==h,'Frozen source changed'
        write(root/'COMPLETE.json',dict(new_refiner_fits=24,refiner_epochs=480,new_Base_fits=0,sanity_optimizer_steps=4,
            seconds=budget(cfg,'complete'),TEST_scoring=False,status=result['status'],Base_frozen_exact=True))
        print('MOTION_PAIR_COMPLETE',result['status'],flush=True)
    except Exception as e:
        reason='budget_insufficient' if str(e)=='Budget insufficient' else ('budget_exceeded' if str(e)=='Budget exceeded' else 'execution_error')
        write(root/'FAILURE.json',dict(reason_code=reason,error=repr(e),seconds=time.monotonic()-START,complete=False))
        raise


if __name__=='__main__':main()
