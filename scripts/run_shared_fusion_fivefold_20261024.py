"""Frozen original B/XGB refits; models never read Test labels or Test scores."""
import argparse
import copy
import gc
import json
import math
import os
import random
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from scripts.shared_fusion_fivefold_common_20261024 import CONFIG,configure,read,save,sha,case_dir,check_method_lock,check_data,code_files
from scripts.sensor_convtran_endpoint_interaction_20261022 import EndpointInteractionConvTran
from scripts.sensor_uwb_event_motion_20261020 import ParameterEMA
from scripts.sensor_xgb_b_fusion_20261022 import validate_probabilities


def core_runtime():
    from scripts import run_convtran_context_20261022 as core
    return core


def load_nn_data(cfg,protocol,fold):
    receipt=check_data(cfg,protocol,fold);directory=case_dir(cfg,protocol,fold)
    names=('dynamic','state','state_valid','context','current')
    with np.load(directory/'FIT.npz',allow_pickle=False) as fit,np.load(directory/'TEST_INPUTS.npz',allow_pickle=False) as test:
        assert 'labels' not in test.files and 'labels' in fit.files
        labels=fit['labels'];prior=fit['prior'];class_weights=fit['class_weights']
        np.testing.assert_array_equal(prior,test['prior'])
        values={k:np.concatenate([fit[k],test[k]],axis=0) for k in names}
    nf,nt=receipt['fit_targets'],receipt['test_targets'];assert labels.shape==(nf,)
    for k,value in values.items():assert len(value)==nf+nt and np.isfinite(value).all(),k
    counts=np.bincount(labels,minlength=7)
    np.testing.assert_array_equal(counts,receipt['fit_class_counts'])
    np.testing.assert_array_equal(prior,counts/counts.sum())
    np.testing.assert_array_equal(class_weights,np.sqrt(counts.sum()/(7*counts)))
    tensors={k:torch.as_tensor(v,device='cuda') for k,v in values.items()}
    # This vector intentionally has Fit length only. No Test truth enters tensors.
    tensors['labels']=torch.as_tensor(labels,dtype=torch.long,device='cuda')
    return dict(tensors=tensors,labels=labels,prior=prior,
        weights=torch.as_tensor(class_weights,dtype=torch.float32,device='cuda'),
        train=np.arange(nf,dtype=np.int64),evaluation=np.arange(nf,nf+nt,dtype=np.int64),
        receipt=receipt,protocol=protocol,fold=fold)


def network(cfg,prior):return EndpointInteractionConvTran(cfg,prior).cuda()


def optimizer(model,cfg):
    return torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],eps=1e-8)


@torch.no_grad()
def infer(model,data,ids,batch_size):
    model.eval();parts=[]
    for start in range(0,len(ids),batch_size):
        logits=model(core_runtime().minibatch(data,ids[start:start+batch_size]));assert torch.isfinite(logits).all()
        parts.append(torch.softmax(logits.double(),dim=-1).cpu().numpy())
    return validate_probabilities(np.concatenate(parts))


def equal(a,b):
    if torch.is_tensor(a):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


def checkpoint(path,model,opt,ema,metadata):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),ema=ema.model.state_dict(),
        ema_updates=ema.updates,rng=core_runtime().rng(),**metadata),tmp);tmp.replace(path)


def restore_check(cfg,data,model,opt,ema,path):
    core=core_runtime();ids=data['train'][:min(256,len(data['train']))]
    checkpoint(path,model,opt,ema,dict(epoch=0))
    expected=core.update(model,opt,ema,data,ids);expected_rng=core.rng()
    payload=torch.load(path,map_location='cpu',weights_only=False)
    other=network(cfg,data['prior']);other.load_state_dict(payload['model']);o=optimizer(other,cfg);o.load_state_dict(payload['optimizer'])
    e=ParameterEMA(other,cfg['ema_decay'],cfg['ema_warmup_steps']);e.model.load_state_dict(payload['ema']);e.updates=payload['ema_updates'];e.model.eval().requires_grad_(False)
    core.restore_rng(payload['rng']);equal(expected,core.update(other,o,e,data,ids))
    equal(expected_rng,core.rng());equal(model.state_dict(),other.state_dict());equal(opt.state_dict(),o.state_dict());equal(ema.model.state_dict(),e.model.state_dict())
    equal(infer(model,data,ids,cfg['batch_size']),infer(other,data,ids,cfg['batch_size']))
    equal(infer(ema.model,data,ids,cfg['batch_size']),infer(e.model,data,ids,cfg['batch_size']))


def xgb_params(cfg):
    p=cfg['xgboost']
    return dict(objective='multi:softprob',num_class=7,tree_method='hist',device='cpu',nthread=p['nthread'],
        eta=p['eta'],max_depth=p['max_depth'],seed=p['seed'],subsample=1.,colsample_bytree=1.,disable_default_eval_metric=1)


def load_tree_data(cfg,protocol,fold):
    receipt=check_data(cfg,protocol,fold);directory=case_dir(cfg,protocol,fold)
    with np.load(directory/'FIT.npz',allow_pickle=False) as z:
        x,y,prior,w=z['features'],z['labels'],z['prior'],z['class_weights']
    with np.load(directory/'TEST_INPUTS.npz',allow_pickle=False) as z:
        assert 'labels' not in z.files
        query=z['features'];missing=~((z['dynamic'][:,:,6]>.5).any(1)|z['state_valid'].any(1))
        np.testing.assert_array_equal(prior,z['prior'])
    counts=np.bincount(y,minlength=7);np.testing.assert_array_equal(counts,receipt['fit_class_counts'])
    np.testing.assert_array_equal(w,np.sqrt(counts.sum()/(7*counts)))
    assert x.shape==(receipt['fit_targets'],866) and query.shape==(receipt['test_targets'],866)
    return x,y,w,prior,query,missing,receipt


def tree(cfg,protocol,fold,preflight=False):
    import xgboost as xgb
    if not preflight:
        from scripts.freeze_shared_fusion_fivefold_20261024 import check_training_seal
        check_training_seal(cfg)
    folder=Path(cfg['output'])/'tree'/protocol/fold
    if not preflight:
        assert not (folder/'STARTED.json').exists(),'No automatic tree retry'
        save(folder/'STARTED.json',dict(pid=os.getpid(),started_unix=time.time()))
    start=time.monotonic();x,y,w,prior,query,missing,receipt=load_tree_data(cfg,protocol,fold);load_seconds=time.monotonic()-start
    train=xgb.DMatrix(x,label=y,weight=w[y],nthread=4)
    rounds=5 if preflight else cfg['xgboost']['rounds'];start=time.monotonic()
    model=xgb.train(xgb_params(cfg),train,num_boost_round=rounds);fit_seconds=time.monotonic()-start
    assert model.num_boosted_rounds()==rounds
    model_path=(Path(cfg['output'])/'preflight'/protocol/fold/'five_tree.json') if preflight else folder/'model.json'
    model_path.parent.mkdir(parents=True,exist_ok=True);model.save_model(model_path)
    other=xgb.Booster();other.load_model(model_path);other.set_param({'nthread':4})
    # Preflight observes Fit inputs only, even for inference timing.
    q=xgb.DMatrix(x[:len(query)] if preflight else query,nthread=4);start=time.monotonic()
    probability=model.predict(q);np.testing.assert_array_equal(probability,other.predict(q));inference_seconds=time.monotonic()-start
    probability=validate_probabilities(probability)
    result=dict(status='PASS' if preflight else 'COMPLETE',protocol=protocol,fold=fold,model='XGBoost',rounds=rounds,
        features=866,fit_count=1,fit_seed=17,fit_targets=len(y),test_targets=len(query),load_seconds=load_seconds,
        fit_seconds=fit_seconds,model_path=model_path.as_posix(),model_sha256=sha(model_path),serialized_replay_exact=True,
        Test_labels_read=False,Test_scores_computed=0,model_Test_predictions_computed=not preflight)
    if preflight:
        result['estimated_job_seconds']=(fit_seconds/5*cfg['xgboost']['rounds']+inference_seconds/5*cfg['xgboost']['rounds']+load_seconds)*1.5+60
        save(model_path.with_name('TREE_PREFLIGHT.json'),result)
        assert result['estimated_job_seconds']<=cfg['budget']['tree_job_seconds']
    else:
        probability[missing]=prior;probability=validate_probabilities(probability)
        path=folder/'test_probabilities.npy';np.save(path,probability)
        result.update(probability_path=path.as_posix(),probability_sha256=sha(path),reused_for_B_seeds=cfg['seeds'],elapsed_seconds=load_seconds+fit_seconds+inference_seconds)
        check_training_seal(cfg);save(folder/'COMPLETE.json',result)
    return result


def neural(cfg,protocol,fold,seed):
    from scripts.freeze_shared_fusion_fivefold_20261024 import check_training_seal
    check_training_seal(cfg);core=core_runtime();lock=core.acquire_gpu()
    folder=Path(cfg['output'])/'B'/protocol/fold/f'seed{seed}'
    assert not (folder/'STARTED.json').exists(),'No automatic neural retry'
    save(folder/'STARTED.json',dict(pid=os.getpid(),started_unix=time.time(),protocol=protocol,fold=fold,seed=seed))
    started=time.monotonic();data=load_nn_data(cfg,protocol,fold);core.seed_all(seed)
    model=network(cfg,data['prior']);assert sum(p.numel() for p in model.parameters())==213119
    opt=optimizer(model,cfg);ema=ParameterEMA(model,cfg['ema_decay'],cfg['ema_warmup_steps']);history=[]
    for epoch in range(1,cfg['epochs']+1):
        order=np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(data['train'])
        numerator=denominator=0.;tick=time.monotonic()
        for start in range(0,len(order),cfg['batch_size']):
            loss,mass=core.update(model,opt,ema,data,order[start:start+cfg['batch_size']]);numerator+=loss*mass;denominator+=mass
        assert all(torch.isfinite(p).all() for p in ema.model.parameters())
        row=dict(epoch=epoch,Fit_CE=numerator/denominator,seconds=time.monotonic()-tick);history.append(row)
        save(folder/'history.json',history)
        checkpoint(folder/'latest.pt',model,opt,ema,dict(epoch=epoch,protocol=protocol,fold=fold,seed=seed,history=history,fit_scope=cfg['fit_scope']))
        save(folder/'PROGRESS.json',dict(protocol=protocol,fold=fold,seed=seed,epoch=epoch,total_epochs=cfg['epochs'],elapsed_seconds=time.monotonic()-started,updated_unix=time.time()))
        print(json.dumps(dict(protocol=protocol,fold=fold,seed=seed,**row)),flush=True)
    predictions={};payload=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
    for view,net,key in [('raw',model,'model'),('ema',ema.model,'ema')]:
        probability=infer(net,data,data['evaluation'],cfg['batch_size'])
        restored=network(cfg,data['prior']).eval().requires_grad_(False);restored.load_state_dict(payload[key])
        np.testing.assert_array_equal(probability,infer(restored,data,data['evaluation'],cfg['batch_size']))
        path=folder/f'test_{view}.npy';np.save(path,probability);predictions[view]=dict(path=path.as_posix(),sha256=sha(path))
        del restored
    check_training_seal(cfg)
    save(folder/'COMPLETE.json',dict(status='COMPLETE',protocol=protocol,fold=fold,seed=seed,model='EndpointResidual_B',epoch=20,
        predictions=predictions,checkpoint_sha256=sha(folder/'latest.pt'),history_sha256=sha(folder/'history.json'),
        fit_targets=len(data['train']),test_targets=len(data['evaluation']),Test_labels_read=False,Test_scores_computed=0,
        serialized_raw_ema_probabilities_exact=True,elapsed_seconds=time.monotonic()-started,fit_scope=cfg['fit_scope']))
    lock.close()


def preflight(cfg):
    check_method_lock(cfg);out=Path(cfg['output']);assert not (out/'PREFLIGHT_STARTED.json').exists()
    save(out/'PREFLIGHT_STARTED.json',dict(pid=os.getpid(),started_unix=time.time()))
    assert read(out/'FIVEFOLD_COVERAGE.json')['status']=='PASS'
    core=core_runtime();lock=core.acquire_gpu();records=[];total=300.
    try:
        for protocol in cfg['protocols']:
            for fold in cfg['folds']:
                start=time.monotonic();data=load_nn_data(cfg,protocol,fold);load_seconds=time.monotonic()-start
                core.seed_all(17);model=network(cfg,data['prior']);opt=optimizer(model,cfg);ema=ParameterEMA(model,cfg['ema_decay'],cfg['ema_warmup_steps'])
                times=[];losses=[];torch.cuda.reset_peak_memory_stats()
                for i in range(3):
                    torch.cuda.synchronize();t=time.monotonic()
                    losses.append(core.update(model,opt,ema,data,data['train'][i*256:(i+1)*256])[0]);torch.cuda.synchronize();times.append(time.monotonic()-t)
                directory=out/'preflight'/protocol/fold;restore_check(cfg,data,model,opt,ema,directory/'resume.pt')
                torch.cuda.synchronize();t=time.monotonic();checkpoint(directory/'timed.pt',model,opt,ema,dict(epoch=0));torch.cuda.synchronize();save_seconds=time.monotonic()-t
                t=time.monotonic()
                for net in (model,ema.model):
                    p=infer(net,data,data['train'][:len(data['evaluation'])],cfg['batch_size']);del p
                torch.cuda.synchronize();infer_seconds=time.monotonic()-t
                estimate=(max(times[1:])*math.ceil(len(data['train'])/256)*20+save_seconds*20+infer_seconds*2+load_seconds)*1.5+60
                assert estimate<=cfg['budget']['neural_job_seconds']
                record=dict(protocol=protocol,fold=fold,Fit_CE=losses,step_seconds=times,load_seconds=load_seconds,
                    estimated_neural_job_seconds=estimate,parameters=sum(p.numel() for p in model.parameters()),
                    next_update_model_optimizer_EMA_RNG_exact=True,serialized_probabilities_exact=True,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),Test_predictions_computed=0,Test_scores_computed=0)
                del model,opt,ema,data,net;gc.collect();torch.cuda.empty_cache()
                tr=tree(cfg,protocol,fold,preflight=True);record['tree']=tr;total+=estimate*3+tr['estimated_job_seconds']
                records.append(record);save(out/'PREFLIGHT_PROGRESS.json',dict(status='RUNNING',records=records));print(json.dumps(record),flush=True)
    finally:lock.close()
    assert total<=cfg['budget']['total_seconds']
    save(out/'PREFLIGHT.json',dict(status='PASS',records=records,conservative_serial_seconds=total,
        implementation_files=code_files(cfg),Test_scores_computed=0,Test_predictions_computed=0))


def witness(cfg,output):
    import xgboost as xgb
    core=core_runtime();lock=core.acquire_gpu();core.seed_all(17)
    n=14;dynamic=torch.randn(n,100,8,device='cuda');dynamic[:,:,6]=1.;dynamic[:,:,7]=.05
    state=torch.randn(n,60,26,device='cuda');prior=np.full(7,1/7)
    data=dict(train=np.arange(n),prior=prior,weights=torch.ones(7,device='cuda'),tensors=dict(dynamic=dynamic,state=state,
        state_valid=torch.ones(n,60,dtype=torch.bool,device='cuda'),context=torch.randn(n,136,device='cuda'),current=state[:,-1],labels=torch.arange(n,device='cuda')%7))
    model=network(cfg,prior);opt=optimizer(model,cfg);ema=ParameterEMA(model,cfg['ema_decay'],cfg['ema_warmup_steps'])
    losses=[core.update(model,opt,ema,data,data['train'])[0] for _ in range(2)]
    output=Path(output);restore_check(cfg,data,model,opt,ema,output.parent/(output.stem+'_B.pt'))
    rng=np.random.default_rng(17);x=rng.normal(size=(70,866)).astype(np.float32);matrix=xgb.DMatrix(x,label=np.arange(70)%7,nthread=4)
    tree_model=xgb.train(xgb_params(cfg),matrix,num_boost_round=5);tree_path=output.parent/(output.stem+'_XGB.json');tree_model.save_model(tree_path)
    replay=xgb.Booster();replay.load_model(tree_path);replay.set_param({'nthread':4});np.testing.assert_array_equal(tree_model.predict(matrix),replay.predict(matrix))
    save(output,dict(status='PASS',scope='synthetic only',neural_parameters=213119,xgb_features=866,classes=7,
        Fit_CE=losses,neural_next_update_model_optimizer_EMA_RNG_exact=True,neural_probability_replay_exact=True,tree_probability_replay_exact=True,
        torch=torch.__version__,numpy=np.__version__,xgboost=xgb.__version__,implementation_files=code_files(cfg),Test_rows_read=0))
    lock.close();print('FIVEFOLD_FUSION_RUNTIME_WITNESS_PASS',flush=True)


def cpu_witness(cfg,output):
    """Independent doc witness without using an occupied GPU; CUDA checked later."""
    import xgboost as xgb
    from torch.nn import functional as F
    torch.set_num_threads(4);random.seed(17);np.random.seed(17);torch.manual_seed(17)
    n=14;prior=np.full(7,1/7);dynamic=torch.randn(n,100,8);dynamic[:,:,6]=1.;dynamic[:,:,7]=.05
    state=torch.randn(n,60,26);batch=dict(dynamic=dynamic,state=state,state_valid=torch.ones(n,60,dtype=torch.bool),
        context=torch.randn(n,136),current=state[:,-1]);labels=torch.arange(n)%7
    model=EndpointInteractionConvTran(cfg,prior);opt=optimizer(model,cfg);ema=ParameterEMA(model,cfg['ema_decay'],cfg['ema_warmup_steps'])
    def step(m,o,e):
        m.train();o.zero_grad(set_to_none=True);loss=F.cross_entropy(m(batch),labels);assert torch.isfinite(loss)
        loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        o.step();e.update(m);return float(loss.detach())
    losses=[step(model,opt,ema) for _ in range(2)]
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);path=output.parent/(output.stem+'_CPU.pt')
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),ema=ema.model.state_dict(),ema_updates=ema.updates,
        torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate()),path)
    expected=step(model,opt,ema);expected_rng=torch.get_rng_state()
    p=torch.load(path,map_location='cpu',weights_only=False);other=EndpointInteractionConvTran(cfg,prior);other.load_state_dict(p['model'])
    oo=optimizer(other,cfg);oo.load_state_dict(p['optimizer']);ee=ParameterEMA(other,cfg['ema_decay'],cfg['ema_warmup_steps'])
    ee.model.load_state_dict(p['ema']);ee.updates=p['ema_updates'];ee.model.eval().requires_grad_(False)
    torch.set_rng_state(p['torch_rng']);np.random.set_state(p['numpy_rng']);random.setstate(p['python_rng'])
    equal(expected,step(other,oo,ee));equal(expected_rng,torch.get_rng_state())
    equal(model.state_dict(),other.state_dict());equal(opt.state_dict(),oo.state_dict());equal(ema.model.state_dict(),ee.model.state_dict())
    model.eval();other.eval()
    with torch.no_grad():equal(model(batch),other(batch));equal(ema.model(batch),ee.model(batch))
    x=np.random.default_rng(17).normal(size=(70,866)).astype(np.float32);matrix=xgb.DMatrix(x,label=np.arange(70)%7,nthread=4)
    t=xgb.train(xgb_params(cfg),matrix,num_boost_round=5);tp=output.parent/(output.stem+'_XGB.json');t.save_model(tp)
    replay=xgb.Booster();replay.load_model(tp);replay.set_param({'nthread':4});np.testing.assert_array_equal(t.predict(matrix),replay.predict(matrix))
    save(output,dict(status='PASS',scope='synthetic CPU only; CUDA not exercised',CUDA_validated=False,
        neural_parameters=sum(p.numel() for p in model.parameters()),xgb_features=866,classes=7,Fit_CE=losses,
        neural_next_update_exact=True,optimizer_EMA_replay_exact=True,tree_probability_replay_exact=True,
        torch=torch.__version__,numpy=np.__version__,xgboost=xgb.__version__,implementation_files=code_files(cfg),Test_rows_read=0))
    print('FIVEFOLD_FUSION_CPU_RUNTIME_WITNESS_PASS',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('witness','cpu_witness','preflight','neural','tree'),required=True)
    parser.add_argument('--protocol',choices=('TS','OS'));parser.add_argument('--fold');parser.add_argument('--seed',type=int,choices=(17,23,29));parser.add_argument('--witness-output')
    args=parser.parse_args();cfg=configure(cuda=args.mode not in ('tree','cpu_witness'))
    try:
        if args.mode=='witness':witness(cfg,args.witness_output or Path(cfg['output'])/'RUNTIME_WITNESS.json')
        elif args.mode=='cpu_witness':cpu_witness(cfg,args.witness_output or Path(cfg['output'])/'CPU_RUNTIME_WITNESS.json')
        elif args.mode=='preflight':preflight(cfg)
        else:
            assert args.protocol in cfg['protocols'] and args.fold in cfg['folds']
            if args.mode=='neural':assert args.seed in cfg['seeds'];neural(cfg,args.protocol,args.fold,args.seed)
            else:tree(cfg,args.protocol,args.fold)
    except Exception:
        save(Path(cfg['output'])/'failures'/f'{args.mode}_{args.protocol}_{args.fold}_{args.seed}_{time.time_ns()}.json',
            dict(status='FAILED',traceback=traceback.format_exc(),unix=time.time()));raise


if __name__=='__main__':main()
