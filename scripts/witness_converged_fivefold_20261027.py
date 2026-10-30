"""Synthetic resume and real TRAIN-only cost checks; no Test readers."""
import argparse,copy,gc,time
from pathlib import Path
import numpy as np
import torch
import xgboost as xgb
from scripts import converged_fivefold_common_20261027 as c
from scripts.run_converged_fivefold_20261027 import gpu,recover,tree_arrays
from scripts.run_tree_convergence_20261027 import assess as tree_assess


def equality(a,b):return c.native.base.equal_tree(a,b)


def witness(cfg,cpu):
    device='cpu' if cpu else 'cuda';lock=None if cpu else gpu(cfg);out=Path(cfg['output']);name='CPU_WITNESS' if cpu else 'CUDA_WITNESS'
    if cpu:assert not torch.cuda.is_available(),'CPU witness must hide CUDA devices in its documented invocation'
    receipt=out/(name+'.json')
    if receipt.exists():receipt.rename(out/(name+'_superseded_'+str(time.time_ns())+'.json'))
    work=out/(name+'_work_'+str(time.time_ns()));work.mkdir(parents=True)
    c.native.core.seed_all(17);n=17;dyn=torch.randn(n,100,8,device=device);dyn[:,:,6]=1;sta=torch.randn(n,60,26,device=device)
    data=dict(prior=np.full(7,1/7),weights=torch.ones(7,device=device),train=np.arange(n),evaluation=np.arange(n),fit_truth=np.arange(n)%7,dev_truth=np.arange(n)%7,
        tensors=dict(dynamic=dyn,state=sta,state_valid=torch.ones(n,60,dtype=torch.bool,device=device),context=torch.randn(n,136,device=device),current=sta[:,-1],labels=torch.arange(n,device=device)%7))
    state=c.new_state(cfg,data,17);assert isinstance(state['opt'],torch.optim.AdamW)
    c.update(state,data,data['train'])
    for v in [1.]*12:state['scheduler'].step(v)
    assert 3e-6<=state['opt'].param_groups[0]['lr']<=3e-6*(1+1e-10)
    snaps=[c.snapshot(state,i) for i in range(9,12)];c.checkpoint(work/'state.pt',state,snaps,epoch=11)
    payload=torch.load(work/'state.pt',map_location='cpu',weights_only=False);frozen=copy.deepcopy(payload);rng=c.native.core.rng();live=copy.deepcopy(state['ema'].model.state_dict())
    model=c.evaluation(state,snaps,11);p=c.probabilities(model,data,data['evaluation']);equality(live,state['ema'].model.state_dict());equality(rng,c.native.core.rng())
    expected=c.update(state,data,data['train']);state['scheduler'].step(1.);expected_rng=c.native.core.rng()
    other=c.restore(cfg,data,payload,device);actual=c.update(other,data,data['train']);other['scheduler'].step(1.)
    equality(expected,actual);equality(expected_rng,c.native.core.rng())
    for k in ['model','opt']:equality(state[k].state_dict(),other[k].state_dict())
    equality(state['ema'].model.state_dict(),other['ema'].model.state_dict());equality(state['scheduler'].state_dict(),other['scheduler'].state_dict());equality(frozen,payload)
    torch.save(dict(model=model.state_dict(),epoch=11),work/'anchor.pt');fresh=recover(cfg,data,17,work/'anchor.pt')
    equality(fresh['model'].state_dict(),model.state_dict());equality(fresh['ema'].model.state_dict(),model.state_dict());assert not fresh['opt'].state
    torch.save(model.state_dict(),work/'evaluation.pt');model.load_state_dict(torch.load(work/'evaluation.pt',map_location='cpu',weights_only=False));np.testing.assert_array_equal(p,c.probabilities(model,data,data['evaluation']))
    m=dict(weighted_CE=1.,macro_f1_percent=70.);history=[dict(epoch=i,learning_rate=3e-6,train_ce=1.,fit_ema=m,dev_ema=m,fusion_fit=m,fusion_ema=m,raw_relative_parameter_change=0.,EMA_relative_parameter_change=0.) for i in range(1,41)]
    assert c.joint_assess(history)['passed'] and c.fit_assess(history)['passed'] and not c.joint_assess(history[:39])['passed']
    bad=copy.deepcopy(history)
    for r in bad:r['learning_rate']=0.
    assert not c.joint_assess(bad)['passed'] and not c.fit_assess(bad)['passed']
    bad=copy.deepcopy(history)
    for r in bad[-10:]:r['fusion_fit']=dict(weighted_CE=2.+r['epoch'],macro_f1_percent=70.)
    assert not c.joint_assess(bad)['passed'] and not c.fit_assess(bad)['passed']
    assert c.planned_rate(dict(base_learning_rates=[.0003,.00003]),3)==3e-6
    x=np.random.default_rng(17).normal(size=(28,866)).astype('float32');dm=xgb.DMatrix(x,label=np.arange(28)%7,nthread=4)
    tree=xgb.train(dict(objective='multi:softprob',num_class=7,nthread=4,tree_method='hist',seed=17),dm,num_boost_round=2)
    tree.save_model(work/'tree.ubj');second=xgb.Booster();second.load_model(work/'tree.ubj');np.testing.assert_array_equal(tree.predict(dm),second.predict(dm))
    # Exercise both real loader branches against the actual cache field schema.
    fixture=dict(output=str(work/'fixture'));fd=c.data_dir(fixture,'TS','fold_1');fd.mkdir(parents=True)
    np.savez(fd/'FIT.npz',features=x,labels=np.arange(28)%7);np.savez(fd/'DEV_INPUTS.npz',features=x)
    (fd/'DEV_TARGETS.jsonl').write_text(''.join(__import__('json').dumps(dict(label=int(i%7+1)))+'\n' for i in range(28)))
    c.save(fd/'REFIT_SOURCE.json',dict(FIT_path=str(fd/'FIT.npz'),FIT_sha256=c.sha(fd/'FIT.npz')))
    for refit in [False,True]:
        matrix,y,w,held,dy=tree_arrays(fixture,'TS','fold_1',refit);assert matrix.num_col()==866 and matrix.num_row()==28
    assert not tree_assess([dict(rounds=50,train=m,dev=m)],cfg['tree_convergence'])['passed']
    c.save(out/(name+'.json'),dict(status='PASS',code=c.source_files(cfg),real_data_rows_read=0,CUDA_validated=not cpu,
        AdamW_model_EMA_scheduler_RNG_resume_exact=True,evaluation_preserves_live_model_and_RNG=True,anchor_recovery_is_fresh=True,
        neural_and_fusion_stopping_positive_and_negative_controls=True,tree_roundtrip_exact=True,calibration_and_refit_tree_schema_checked=True,tree_insufficient_rounds_rejected=True,parameters=sum(v.numel() for v in model.parameters())))
    print(name+'_PASS',flush=True)
    if lock:lock.close()


def preflight(cfg):
    out=Path(cfg['output']);assert not (out/'PREFLIGHT.json').exists();lock=gpu(cfg);records=[]
    for p in cfg['protocols']:
        for refit in [False,True]:
            matrix,truth,weights,held,held_truth=tree_arrays(cfg,p,'fold_1',refit);assert matrix.num_col()==866
            del matrix,truth,weights,held,held_truth
            data=(c.load_refit if refit else c.load_selection)(cfg,p,'fold_1');state=c.new_state(cfg,data,17);times=[];losses=[]
            for i in range(3):
                tick=time.monotonic();loss,mass=c.update(state,data,data['train'][256*i:256*(i+1)]);torch.cuda.synchronize();times.append(time.monotonic()-tick);losses.append(loss)
            path=out/f'preflight_{p}_{refit}.pt';c.checkpoint(path,state,[],epoch=0);payload=torch.load(path,map_location='cpu',weights_only=False)
            expected=c.update(state,data,data['train'][:256]);rng=c.native.core.rng();other=c.restore(cfg,data,payload,'cuda');actual=c.update(other,data,data['train'][:256]);equality(expected,actual);equality(rng,c.native.core.rng());equality(state['model'].state_dict(),other['model'].state_dict());equality(state['opt'].state_dict(),other['opt'].state_dict())
            tick=time.monotonic();model=c.evaluation(state,[c.snapshot(state,1)],1);c.probabilities(model,data,data['train'])
            if not refit:c.probabilities(model,data,data['evaluation'])
            torch.cuda.synchronize();infer=time.monotonic()-tick
            estimate=(max(times[1:])*int(np.ceil(len(data['train'])/256))+infer+1.0)*160*1.5+30
            assert estimate<cfg['budget']['worker_seconds'];assert np.isfinite(losses).all()
            records.append(dict(protocol=p,stage='refit' if refit else 'calibration',FIT_rows=len(data['train']),step_seconds=times,inference_seconds=infer,maximum160epoch_estimate_seconds=estimate,peak_GPU_bytes=torch.cuda.max_memory_allocated(),losses=losses,serialized_next_update_exact=True))
            del data,state,other,payload,model;gc.collect();torch.cuda.empty_cache()
    total=sum(r['maximum160epoch_estimate_seconds']*15 for r in records)+10*2*1800
    assert total<cfg['budget']['queue_seconds'],total
    c.save(out/'PREFLIGHT.json',dict(status='PASS',code=c.source_files(cfg),records=records,conservative_queue_estimate_seconds=total,Test_rows_read=0));lock.close();print('FIVEFOLD_PREFLIGHT_PASS',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['cpu','cuda','preflight'],required=True);a=ap.parse_args();cfg=c.config()
    if a.mode=='preflight':preflight(cfg)
    else:witness(cfg,a.mode=='cpu')
