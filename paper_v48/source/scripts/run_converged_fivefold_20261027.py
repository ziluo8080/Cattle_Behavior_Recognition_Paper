"""Sealed two-stage fivefold fits. Test is available only after training locks."""
import argparse,copy,fcntl,gc,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
import xgboost as xgb
from scripts import converged_fivefold_common_20261027 as c
from scripts.run_tree_convergence_20261027 import assess as tree_assess,metrics as tree_metrics

MODULE='scripts.run_converged_fivefold_20261027'


def gpu(cfg):
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'PRIVATE_GPU.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    assert free>=cfg['shared_GPU']['minimum_free_GiB_before_allocation']*1024,'Insufficient free memory; never stop another job'
    torch.cuda.set_per_process_memory_fraction(cfg['shared_GPU']['Torch_memory_fraction'])
    c.native.core.configure()
    return lock


def tree_arrays(cfg,p,f,refit):
    d=c.data_dir(cfg,p,f)
    if refit:
        source=c.read(d/'REFIT_SOURCE.json');path=Path(source['FIT_path']);assert c.sha(path)==source['FIT_sha256']
    else:path=d/'FIT.npz'
    with np.load(path,allow_pickle=False) as z:
        features=np.array(z['features']);truth=np.array(z['labels'])
        assert features.shape==(len(truth),866) and np.isfinite(features).all()
    counts=np.bincount(truth,minlength=7);weights=np.sqrt(counts.sum()/(7*counts))
    fit=xgb.DMatrix(features,label=truth,weight=weights[truth],nthread=4)
    if refit:return fit,truth,weights,None,None
    with np.load(d/'DEV_INPUTS.npz',allow_pickle=False) as z:dev=xgb.DMatrix(z['features'],nthread=4)
    y=np.array([__import__('json').loads(s)['label']-1 for s in (d/'DEV_TARGETS.jsonl').read_text().splitlines()])
    return fit,truth,weights,dev,y


def completed(folder):
    result=c.read(folder/'COMPLETE.json');assert result['status']=='PASS' and result['converged']
    for n,h in result['files'].items():assert c.sha(folder/n)==h,n
    return result


def tree_worker(cfg,p,f,refit):
    c.check_seal(cfg);stage='final_tree' if refit else 'calibration_tree';out=c.case_dir(cfg,stage,p,f)
    assert not (out/'STARTED.json').exists();c.save(out/'STARTED.json',dict(pid=os.getpid(),unix=time.time()))
    fit,y,w,dev,dy=tree_arrays(cfg,p,f,refit);t=cfg['tree'];start=time.monotonic();history=[];model=None
    minimum=t['minimum_rounds']
    if refit:minimum=completed(c.case_dir(cfg,'calibration_tree',p,f))['rounds']
    params=dict(objective='multi:softprob',num_class=7,tree_method='hist',device='cpu',nthread=t['threads'],
        eta=t['eta'],max_depth=t['max_depth'],gamma=t['gamma'],min_child_weight=t['min_child_weight'],
        reg_lambda=t['reg_lambda'],subsample=1.,colsample_bytree=1.,seed=t['seed'],disable_default_eval_metric=1)
    for rounds in range(t['block_rounds'],t['maximum_rounds']+1,t['block_rounds']):
        model=xgb.train(params,fit,num_boost_round=t['block_rounds'],xgb_model=model)
        fp=model.predict(fit);row=dict(rounds=rounds,train=tree_metrics(y,fp,w))
        if not refit:
            dp=model.predict(dev);row['dev']=tree_metrics(dy,dp,w)
        history.append(row)
        if refit:
            rows=history[-5:];ok=rounds>=minimum and len(rows)==5
            if ok:ok=all(abs(b['train']['weighted_CE']/a['train']['weighted_CE']-1)<=.02 and abs(b['train']['macro_f1_percent']-a['train']['macro_f1_percent'])<=.5 for a,b in zip(rows,rows[1:]))
            assessment=dict(passed=bool(ok),scope='FIT-only plateau; calibration independently checked DEV')
        else:assessment=tree_assess(history,cfg['tree_convergence'])
        c.save(out/'history.json',history);c.save(out/'PROGRESS.json',dict(stage=stage,rounds=rounds,convergence=assessment))
        print(dict(stage=stage,protocol=p,fold=f,rounds=rounds,FIT_CE=row['train']['weighted_CE'],converged=assessment['passed']),flush=True)
        assert time.monotonic()-start<cfg['budget']['worker_seconds']
        if assessment['passed']:break
    c.save(out/'CONVERGENCE.json',assessment)
    if not assessment['passed']:raise RuntimeError('Tree did not converge within fixed budget; no fallback')
    model.save_model(out/'model.ubj');reloaded=xgb.Booster();reloaded.load_model(out/'model.ubj');np.testing.assert_array_equal(fp,reloaded.predict(fit))
    np.save(out/'FIT.npy',fp)
    names=['model.ubj','FIT.npy','history.json','CONVERGENCE.json']
    if not refit:np.save(out/'DEV.npy',dp);names.append('DEV.npy')
    c.save(out/'COMPLETE.json',dict(status='PASS',converged=True,rounds=rounds,Test_targets_opened=False,files={n:c.sha(out/n) for n in names}))


def recover(cfg,data,seed,path):
    """Fresh learner, EMA and AdamW from own preselected evaluation anchor."""
    state=c.new_state(cfg,data,seed,lr=cfg['minimum_learning_rate'])
    model=torch.load(path,map_location='cpu',weights_only=False)['model']
    state['model'].load_state_dict(model);state['ema'].model.load_state_dict(model)
    assert not state['opt'].state and state['ema'].updates==0
    return state


def neural_worker(cfg,p,f,seed,refit):
    c.check_seal(cfg);lock=gpu(cfg);stage='final_neural' if refit else 'calibration_neural';out=c.case_dir(cfg,stage,p,f,seed)
    assert not (out/'STARTED.json').exists();c.save(out/'STARTED.json',dict(pid=os.getpid(),unix=time.time()))
    start=time.monotonic();data=(c.load_refit if refit else c.load_selection)(cfg,p,f)
    tree=c.case_dir(cfg,'final_tree' if refit else 'calibration_tree',p,f);completed(tree)
    xf=np.load(tree/'FIT.npy');xd=None if refit else np.load(tree/'DEV.npy')
    plan=None
    if refit:
        calibration=c.case_dir(cfg,'calibration_neural',p,f,seed);completed(calibration);plan=c.read(calibration/'PLAN.json')
    state=c.new_state(cfg,data,seed);history=[];snaps=[];base_epochs=cfg['maximum_epochs'];anchor=None;best=float('inf');rates=[];recovery=False
    # Calibration may stop after40. Refit replays the complete locked schedule,
    # then extends only at its positive floor when FIT is still changing.
    for epoch in range(1,cfg['maximum_epochs']+cfg['recovery_epochs']+1):
        if epoch==cfg['maximum_epochs']+1:
            if refit and not plan['recovery']:break
            recovery=True;base_epochs=cfg['maximum_epochs']
            if refit:assert anchor==plan['recovery_anchor_epoch']
            state=recover(cfg,data,seed,out/'anchor.pt');snaps=[]
        local=epoch-base_epochs if recovery else epoch
        if refit:
            rate=cfg['minimum_learning_rate'] if recovery else c.planned_rate(plan,epoch)
            for group in state['opt'].param_groups:group['lr']=rate
        rate=state['opt'].param_groups[0]['lr']
        if not recovery:rates.append(rate)
        before={n:v.detach().clone() for n,v in state['model'].named_parameters()};ebefore={n:v.detach().clone() for n,v in state['ema'].model.named_parameters()}
        order=np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(data['train']);num=den=0.
        for i in range(0,len(order),cfg['batch_size']):
            loss,mass=c.update(state,data,order[i:i+cfg['batch_size']]);num+=loss*mass;den+=mass
        raw_delta=c.native.relative_change(state['model'],before);ema_delta=c.native.relative_change(state['ema'].model,ebefore)
        snaps.append(c.snapshot(state,local));snaps=snaps[-3:];rng=c.native.core.rng();evaluation=c.evaluation(state,snaps,local)
        fit=c.probabilities(evaluation,data,data['train']);fm=c.score(fit,data['fit_truth'],data);fusion_fit=c.score(c.fused(xf,fit,cfg),data['fit_truth'],data)
        row=dict(epoch=epoch,phase='recovery' if recovery else 'base',local_epoch=local,learning_rate=rate,train_ce=num/den,
            fit_ema=fm,fusion_fit=fusion_fit,raw_relative_parameter_change=raw_delta,EMA_relative_parameter_change=ema_delta,global_step=state['step'])
        if not refit:
            dev=c.probabilities(evaluation,data,data['evaluation']);dm=c.score(dev,data['dev_truth'],data)
            row.update(dev_ema=dm,fusion_ema=c.score(c.fused(xd,dev,cfg),data['dev_truth'],data))
            if not recovery and dm['weighted_CE']<best:
                best=dm['weighted_CE'];anchor=epoch;torch.save(dict(model=evaluation.state_dict(),epoch=epoch),out/'anchor.pt')
            state['scheduler'].step(dm['weighted_CE'])
        elif plan['recovery'] and epoch==plan['recovery_anchor_epoch']:
            anchor=epoch;torch.save(dict(model=evaluation.state_dict(),epoch=epoch),out/'anchor.pt')
        c.native.core.restore_rng(rng);history.append(row)
        assessment=c.fit_assess(history,minimum=plan['base_epochs']) if refit else c.joint_assess(history)
        if refit and plan['recovery'] and epoch<cfg['maximum_epochs']+cfg['recovery_epochs']:assessment=dict(passed=False,reason='locked_recovery_plan_not_complete')
        if recovery and local<cfg['recovery_epochs']:assessment=dict(passed=False,reason='minimum40_recovery_epochs')
        row['next_learning_rate']=state['opt'].param_groups[0]['lr']
        c.save(out/'history.json',history);c.checkpoint(out/'latest.pt',state,snaps,epoch=epoch,phase=row['phase'])
        c.save(out/'PROGRESS.json',dict(stage=stage,protocol=p,fold=f,seed=seed,epoch=epoch,convergence=assessment,elapsed_seconds=time.monotonic()-start))
        print(dict(stage=stage,protocol=p,fold=f,seed=seed,epoch=epoch,LR=rate,TRAIN_CE=row['train_ce'],FIT_CE=fm['weighted_CE'],DEV_CE=None if refit else dm['weighted_CE'],converged=assessment['passed']),flush=True)
        assert np.isfinite(num/den) and time.monotonic()-start<cfg['budget']['worker_seconds']
        if assessment['passed']:
            base_epochs=cfg['maximum_epochs'] if recovery else epoch
            torch.save(dict(model=evaluation.state_dict(),epoch=epoch),out/'evaluation.pt');np.save(out/'FIT.npy',fit)
            if not refit:np.save(out/'DEV.npy',dev)
            del evaluation,before,ebefore;break
        del evaluation,before,ebefore
        if not recovery and refit and not plan['recovery'] and epoch==cfg['maximum_epochs']:break
    c.save(out/'CONVERGENCE.json',assessment)
    if not assessment['passed']:raise RuntimeError('Neural/fusion failed prespecified convergence; no Test and no automatic retry')
    if not refit:
        c.save(out/'PLAN.json',dict(base_epochs=base_epochs,base_learning_rates=rates,recovery=recovery,recovery_anchor_epoch=anchor if recovery else None,final_uses_own_anchor=True,source='TRAIN/VAL only'))
    # Reload terminal evaluation state and verify predictions before declaring success.
    evaluation=c.evaluation(state,snaps,local);evaluation.load_state_dict(torch.load(out/'evaluation.pt',map_location='cpu',weights_only=False)['model'])
    np.testing.assert_array_equal(fit,c.probabilities(evaluation,data,data['train']))
    names=['evaluation.pt','latest.pt','FIT.npy','history.json','CONVERGENCE.json']+([] if refit else ['DEV.npy','PLAN.json'])
    c.save(out/'COMPLETE.json',dict(status='PASS',converged=True,epochs=epoch,recovery=recovery,Test_targets_opened=False,files={n:c.sha(out/n) for n in names}))
    lock.close()


def seal(cfg):
    out=Path(cfg['output']);assert not (out/'TRAINING_SEAL.json').exists();code=c.source_files(cfg);evidence={};data={}
    for name in ['CPU_WITNESS.json','CUDA_WITNESS.json','PREFLIGHT.json']:
        path=out/name;r=c.read(path);assert r['status']=='PASS' and r['code']==code,name;evidence[name]=c.sha(path)
    for p in cfg['protocols']:
        for f in cfg['folds']:
            d=c.data_dir(cfg,p,f);path=d/'RECEIPT.json';r=c.read(path);assert r['status']=='PASS';data[str(path)]=c.sha(path)
            for n,h in r['outputs'].items():assert c.sha(d/n)==h;data[str(d/n)]=h
    correction=out/'ENTRYPOINT_FIX.json'
    adopted={}
    if correction.exists():
        repair=c.read(correction);assert repair['failed_final_fit_updates']==0 and repair['scientific_config_unchanged']
        for path in out.glob('calibration_tree/*/*/COMPLETE.json'):
            completed(path.parent);adopted[str(path)]=c.sha(path)
    c.save(out/'TRAINING_SEAL.json',dict(config=cfg,code=code,data=data,evidence=evidence,unix=time.time(),adopted_completed_calibrations=adopted,entrypoint_fix_sha256=c.sha(correction) if correction.exists() else None))


def test_predictions(cfg):
    """Only called after all80 fits passed. Read inputs, never Test truth here."""
    c.check_seal(cfg);out=Path(cfg['output']);assert not (out/'TRAINING_LOCK.json').exists();models={}
    for p in cfg['protocols']:
        for f in cfg['folds']:
            for stage in ['calibration_tree','final_tree']:completed(c.case_dir(cfg,stage,p,f))
            for seed in cfg['seeds']:completed(c.case_dir(cfg,'calibration_neural',p,f,seed));completed(c.case_dir(cfg,'final_neural',p,f,seed))
            for stage,seeds in [('final_tree',[None]),('final_neural',cfg['seeds'])]:
                for seed in seeds:
                    path=c.case_dir(cfg,stage,p,f,seed)/'COMPLETE.json';models[str(path)]=c.sha(path)
    assert len(models)==40;c.save(out/'TRAINING_LOCK.json',dict(status='PASS',models=models,code=c.source_files(cfg),Test_targets_opened=False,unix=time.time()))
    lock=gpu(cfg);files={}
    for p in cfg['protocols']:
        for f in cfg['folds']:
            source=c.read(c.data_dir(cfg,p,f)/'REFIT_SOURCE.json');path=Path(source['TEST_INPUTS_path']);assert c.sha(path)==source['TEST_INPUTS_sha256']
            with np.load(path,allow_pickle=False) as z:
                assert 'labels' not in z.files;tensors={k:torch.as_tensor(z[k],device='cuda') for k in c.FIELDS};features=np.array(z['features'])
            data=c.load_refit(cfg,p,f);data['tensors']=tensors;data['evaluation']=np.arange(len(features))
            tree=xgb.Booster();tree.load_model(c.case_dir(cfg,'final_tree',p,f)/'model.ubj');px=tree.predict(xgb.DMatrix(features,nthread=4))
            folder=out/'predictions'/p/f;folder.mkdir(parents=True);np.save(folder/'XGB.npy',px)
            for seed in cfg['seeds']:
                state=c.new_state(cfg,data,seed);state['model'].load_state_dict(torch.load(c.case_dir(cfg,'final_neural',p,f,seed)/'evaluation.pt',map_location='cpu',weights_only=False)['model'])
                pb=c.probabilities(state['model'].eval(),data,data['evaluation']);np.save(folder/f'B_seed{seed}.npy',pb);np.save(folder/f'fusion_seed{seed}.npy',c.fused(px,pb,cfg));del state;gc.collect();torch.cuda.empty_cache()
            for path in folder.glob('*.npy'):files[str(path)]=c.sha(path)
            del data,tensors;gc.collect();torch.cuda.empty_cache()
    assert len(files)==70;c.save(out/'PREDICTION_LOCK.json',dict(status='PASS',files=files,training_lock_sha256=c.sha(out/'TRAINING_LOCK.json'),Test_targets_opened=False));lock.close()


def score_once(cfg):
    out=Path(cfg['output']);c.check_seal(cfg);pred=c.read(out/'PREDICTION_LOCK.json');assert c.sha(out/'TRAINING_LOCK.json')==pred['training_lock_sha256']
    for path,h in pred['files'].items():assert c.sha(path)==h
    assert not (out/'SCORING_STARTED.json').exists();c.save(out/'SCORING_STARTED.json',dict(unix=time.time(),pid=os.getpid(),one_attempt=True))
    records=[]
    for p in cfg['protocols']:
        for f in cfg['folds']:
            source=c.read(c.data_dir(cfg,p,f)/'REFIT_SOURCE.json');path=Path(source['TEST_TARGETS_path']);assert c.sha(path)==source['TEST_TARGETS_sha256']
            y=np.array([__import__('json').loads(s)['label']-1 for s in path.read_text().splitlines()]);data=c.load_refit(cfg,p,f,'cpu');folder=out/'predictions'/p/f
            for seed in cfg['seeds']:
                records.append(dict(protocol=p,fold=f,seed=seed,tree=c.score(np.load(folder/'XGB.npy'),y,data),neural=c.score(np.load(folder/f'B_seed{seed}.npy'),y,data),fusion=c.score(np.load(folder/f'fusion_seed{seed}.npy'),y,data)))
    aggregates={p:{m:float(np.mean([r[m]['macro_f1_percent'] for r in records if r['protocol']==p])) for m in ['tree','neural','fusion']} for p in cfg['protocols']}
    c.save(out/'RESULTS.json',dict(status='PASS',records=records,mean_seed_then_mean_fold_F1=aggregates,historical_Test_exposure=True,new_blind_evaluation=False,automatic_manuscript_replacement=False))


def queue(cfg,resume=False):
    c.check_seal(cfg);out=Path(cfg['output']);assert not (out/'QUEUE_STARTED.json').exists();start=time.monotonic();jobs=[]
    for p in cfg['protocols']:
        for f in cfg['folds']:
            jobs.extend([dict(mode='calibration_tree',protocol=p,fold=f,seed=None,status='pending'),dict(mode='final_tree',protocol=p,fold=f,seed=None,status='pending')])
            for seed in cfg['seeds']:
                for mode in ['calibration_neural','final_neural']:jobs.append(dict(mode=mode,protocol=p,fold=f,seed=seed,status='pending'))
    if resume:
        repair=c.read(out/'ENTRYPOINT_FIX.json');assert repair['failed_final_fit_updates']==0
        for j in jobs:
            folder=c.case_dir(cfg,j['mode'],j['protocol'],j['fold'],j['seed'])
            if (folder/'COMPLETE.json').exists():completed(folder);j['status']='completed'
    assert len(jobs)==80;c.save(out/'QUEUE_STARTED.json',dict(pid=os.getpid(),unix=time.time(),jobs=80,resumed_after_loader_fix=resume));child=None
    try:
        for j in jobs:
            if j['status']=='completed':continue
            assert time.monotonic()-start<cfg['budget']['queue_seconds']
            log=out/'logs'/('_'.join(str(j[k]) for k in ['mode','protocol','fold','seed'])+'.log');log.parent.mkdir(exist_ok=True)
            args=[sys.executable,'-s','-u','-m',MODULE,'--mode',j['mode'],'--protocol',j['protocol'],'--fold',j['fold']]
            if j['seed'] is not None:args+=['--seed',str(j['seed'])]
            with log.open('wb') as stream:child=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            j.update(status='running',pid=child.pid,log=str(log));c.save(out/'QUEUE_STATE.json',dict(status='RUNNING',jobs=jobs,elapsed_seconds=time.monotonic()-start))
            child.wait(timeout=cfg['budget']['worker_seconds']+120)
            assert child.returncode==0,f'Failed {log}; scientific failures are never automatically retried'
            completed(c.case_dir(cfg,j['mode'],j['protocol'],j['fold'],j['seed']));j['status']='completed'
            c.save(out/'QUEUE_STATE.json',dict(status='RUNNING',jobs=jobs,elapsed_seconds=time.monotonic()-start))
        test_predictions(cfg);score_once(cfg);c.save(out/'QUEUE_STATE.json',dict(status='COMPLETE',jobs=jobs,elapsed_seconds=time.monotonic()-start))
    except BaseException:
        if child is not None and child.poll() is None:
            # This is exclusively our own child, never the other task's queue.
            child.terminate();child.wait(timeout=30)
        c.save(out/'FAILURE.json',dict(traceback=traceback.format_exc(),unix=time.time()));c.save(out/'QUEUE_STATE.json',dict(status='FAILED_STOP_FOR_INSPECTION',jobs=jobs));raise


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',required=True,choices=['calibration_tree','final_tree','calibration_neural','final_neural','seal','queue','queue_resume','launch','resume']);ap.add_argument('--protocol',choices=['TS','OS']);ap.add_argument('--fold');ap.add_argument('--seed',type=int,choices=[17,23,29]);a=ap.parse_args();cfg=c.config()
    if a.mode=='seal':seal(cfg)
    elif a.mode in ['queue','queue_resume']:queue(cfg,a.mode=='queue_resume')
    elif a.mode in ['launch','resume']:
        c.check_seal(cfg);out=Path(cfg['output']);assert not (out/'DISPATCH.json').exists()
        with (out/'queue.log').open('wb') as stream:child=subprocess.Popen([sys.executable,'-s','-u','-m',MODULE,'--mode','queue_resume' if a.mode=='resume' else 'queue'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        c.save(out/'DISPATCH.json',dict(pid=child.pid,unix=time.time()));print('FIVEFOLD_QUEUE_LAUNCHED',child.pid,flush=True)
    else:
        assert a.protocol in cfg['protocols'] and a.fold in cfg['folds']
        if 'tree' in a.mode:tree_worker(cfg,a.protocol,a.fold,a.mode=='final_tree')
        else:assert a.seed in cfg['seeds'];neural_worker(cfg,a.protocol,a.fold,a.seed,a.mode=='final_neural')


if __name__=='__main__':main()
