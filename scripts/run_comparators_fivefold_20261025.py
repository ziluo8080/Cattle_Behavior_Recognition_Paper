"""Fixed 150-job formal comparison; prediction-only workers, final shared scorer."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from scripts import run_zero_head_replication_20261024 as base
from scripts.sensor_learned_fusion_data_20261023 import read,save,sha
from scripts import run_shared_fusion_fivefold_20261024 as formal
from scripts.shared_fusion_fivefold_common_20261024 import check_data
from experiments.sensor_comparator_pilot_20261022_v2.models import make_model,synthetic_batch
from scripts import sensor_comparator_ridge_20261025 as ridge

core=base.core
CONFIG=Path('configs/sensor_comparators_fivefold_20261025_v1.json')
MODULE='scripts.run_comparators_fivefold_20261025'

def configure(cuda=False):
    assert Path.cwd()==core.ROOT and (torch.__version__,np.__version__)==('2.4.1+cu121','1.24.4')
    if cuda:core.configure()
    else:torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.mha.set_fastpath_enabled(False);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    cfg=read(CONFIG);assert cfg['epochs']==20 and cfg['neural_fits']==120 and cfg['rocket_fits']==30
    assert cfg['Test_allowed'] and cfg['formal_refit_allowed'] and not cfg['Test_selects_parameters'] and not cfg['early_stopping'] and not cfg['automatic_expansion']
    return cfg

def implementation(cfg):
    files=dict(read(Path(cfg['parent'])/'METHOD_LOCK.json')['code'])
    for p,h in files.items():assert sha(p)==h,p
    paths=[CONFIG,Path(cfg['plan']),Path(cfg['ledger']),Path(cfg['environment']),Path(cfg['model_config']),Path('experiments/sensor_comparator_pilot_20261022_v2/models.py')]
    paths+=list(Path('scripts').glob('*comparators_fivefold*20261025.py'))+[Path('scripts/sensor_comparator_ridge_20261025.py'),Path('scripts/run_zero_head_replication_20261024.py')]
    paths+=list(Path('experiments/sensor_comparator_pilot_20261022_v2/vendor').rglob('*.py'))
    files.update({str(p):sha(p) for p in paths});return files

def model_for(cfg,arm,prior,device):return make_model(arm,read(cfg['model_config']),prior).to(device)

def restore_check(cfg,arm,data,folder,device):
    core.seed_all(17);m=model_for(cfg,arm,data['prior'],device);opt=base.optimizer_for(m,read(cfg['model_config']));ema=base.ParameterEMA(m,.998,200);ids=data['train'][:67]
    base.update(m,opt,ema,data,ids);base.checkpoint(folder/'model.pt',m,opt,ema,epoch=0)
    expected=base.update(m,opt,ema,data,ids);expected_rng=core.rng();payload=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
    other=model_for(cfg,arm,data['prior'],device);other.load_state_dict(payload['model']);oo=base.optimizer_for(other,read(cfg['model_config']));oo.load_state_dict(copy.deepcopy(payload['optimizer']))
    ee=base.ParameterEMA(other,.998,200);ee.model.load_state_dict(payload['ema']);ee.updates=payload['ema_updates'];core.restore_rng(payload['rng'])
    base.equal_tree(expected,base.update(other,oo,ee,data,ids));base.equal_tree(expected_rng,core.rng())
    for x,y in [(m.state_dict(),other.state_dict()),(opt.state_dict(),oo.state_dict()),(ema.model.state_dict(),ee.model.state_dict())]:base.equal_tree(x,y)
    np.testing.assert_array_equal(base.infer(m,data,ids),base.infer(other,data,ids));np.testing.assert_array_equal(base.infer(ema.model,data,ids),base.infer(ee.model,data,ids))
    return dict(model=arm,parameters=sum(p.numel() for p in m.parameters()),model_Adam_EMA_RNG_next_update_exact=True,raw_EMA_reload_exact=True)

def rocket_functions():
    import numba
    numba.set_num_threads(4)
    from experiments.sensor_comparator_pilot_20261022_v2.cpu import mr_functions,mr_fit,mr_transform,numba_seed
    return mr_functions(),mr_fit,mr_transform,numba_seed

def witness(cfg,output,cpu=False):
    device='cpu' if cpu else 'cuda';lock=None if cpu else core.acquire_gpu();out=Path(output);records=[]
    for arm in cfg['neural_models']:
        core.seed_all(17);batch=synthetic_batch(device,67);data=dict(tensors=dict(**batch,labels=torch.arange(67,device=device)%7),prior=np.ones(7)/7,weights=torch.ones(7,device=device),train=np.arange(67))
        records.append(restore_check(cfg,arm,data,out.parent/(out.stem+'_work')/arm,device));del data,batch;gc.collect()
        if not cpu:torch.cuda.empty_cache()
    module,fit,transform,setseed=rocket_functions();rng=np.random.default_rng(17);x=rng.normal(size=(21,8,100));options=read(cfg['model_config'])['multirocket']
    np.random.seed(17);setseed(17);a=fit(module,x,1344,options);np.random.seed(17);setseed(17);b=fit(module,x,1344,options)
    for aa,bb in zip(a,b):
        for v,w in zip(aa,bb):np.testing.assert_array_equal(v,w)
    z1=transform(module,x,a,dict(options,transform_batch_size=5));z2=transform(module,x,a,dict(options,transform_batch_size=21));np.testing.assert_array_equal(z1,z2)
    rw=ridge.witness(device)
    save(out,dict(status='PASS',CUDA_validated=not cpu,real_data_rows_read=0,neural=records,MultiRocket_seed_and_batching_exact=True,ridge=rw,implementation_files=implementation(cfg)))
    if lock:lock.close()
    print('COMPARATORS_FIVEFOLD_WITNESS_PASS',flush=True)

def prepare(cfg):
    out=Path(cfg['output']);assert not (out/'PREPARED.json').exists();parent=Path(cfg['parent']);old=read(parent/'SEAL.json');sources={};cases=[]
    assert read(parent/'SUMMARY.json')['status']=='COMPLETE_STOP_FOR_REVIEW'
    for p,h in old['data_files'].items():assert sha(p)==h;sources[p]=h
    for p,h in old['receipts'].items():assert sha(p)==h;sources[p]=h
    for n in ['SEAL.json','METHOD_LOCK.json','SUMMARY.json','CASE_RESULTS.json','PREDICTIONS_FROZEN.json']:sources[str(parent/n)]=sha(parent/n)
    for p,h in read(parent/'PREDICTIONS_FROZEN.json')['files'].items():assert sha(p)==h;sources[p]=h
    selections={}
    for arm in cfg['neural_models']:
        scores={v:[] for v in ['raw','ema']}
        for protocol in cfg['protocols']:
            f=Path(cfg['development_results'])/'groups'/protocol/arm/'COMPLETE.json';r=read(f);assert r['status']=='PASS';sources[str(f)]=sha(f)
            for v in scores:scores[v]+=[r['final'][str(s)][v]['macro_f1']*100 for s in cfg['seeds']]
        means={v:float(np.mean(values)) for v,values in scores.items()};choice='ema' if means['ema']>means['raw'] else 'raw';assert cfg['primary_views'][arm]==choice
        selections[arm]=dict(development_protocol_seed_average=means,selected=choice)
    save(out/'VIEW_SELECTION.json',dict(status='PASS',rule=cfg['view_selection'],selections=selections,new_Test_scores=0));sources[str(out/'VIEW_SELECTION.json')]=sha(out/'VIEW_SELECTION.json')
    for p in cfg['protocols']:
        for fold in cfg['folds']:
            d=parent/'data'/p/fold;r=read(d/'DATA_RECEIPT.json');assert r['phase']=='formal_refit' and r['TEST_INPUTS_has_labels'] is False
            cases.append(dict(protocol=p,fold=fold,fit_targets=r['fit_targets'],test_targets=r['test_targets']))
    jobs=[dict(protocol=p,fold=f,seed=s,model=m,kind='rocket' if m==cfg['cpu_model'] else 'neural',status='pending') for s in cfg['seeds'] for f in cfg['folds'] for p in cfg['protocols'] for m in cfg['neural_models']+[cfg['cpu_model']]]
    assert len(jobs)==150;save(out/'JOB_MANIFEST.json',dict(jobs=jobs,neural=120,rocket=30));sources[str(out/'JOB_MANIFEST.json')]=sha(out/'JOB_MANIFEST.json')
    save(out/'PREPARED.json',dict(status='PASS',files=sources,cases=cases,old_development_fit_reuse=False,new_fits=0))

def load_rocket(cfg,p,fold):
    pcfg=read(cfg['parent_config']);r=check_data(pcfg,p,fold);d=Path(cfg['parent'])/'data'/p/fold;result={}
    for role,name in [('fit','FIT.npz'),('test','TEST_INPUTS.npz')]:
        with np.load(d/name,allow_pickle=False) as z:
            assert ('labels' in z.files)==(role=='fit')
            values={k:z[k].copy() for k in ['dynamic','state','state_valid','context','current']}
            dyn=values['dynamic'];valid=dyn[:,:,6]>.5;dyn[:,:,:6]=np.where(valid[:,:,None],dyn[:,:,:6],0.);dyn[:,:,7]=np.where(valid,np.clip(dyn[:,:,7],0,.2),.2)
            values['native']=dyn;del values['dynamic']
            result[role]=values
            if role=='fit':result['y']=z['labels'];result['prior']=z['prior'];result['class_weights']=z['class_weights']
    result['receipt']=r;return result

def preflight(cfg):
    out=Path(cfg['output']);assert not (out/'PREFLIGHT.json').exists();lock=core.acquire_gpu();prepared=read(out/'PREPARED.json');pcfg=read(cfg['parent_config']);records=[]
    for p in cfg['protocols']:
        case=max([r for r in prepared['cases'] if r['protocol']==p],key=lambda r:r['fit_targets']);data=formal.load_nn_data(pcfg,p,case['fold'])
        for arm in cfg['neural_models']:
            rec=restore_check(cfg,arm,data,out/'preflight'/p/arm,'cuda');core.seed_all(17);m=model_for(cfg,arm,data['prior'],'cuda');opt=base.optimizer_for(m,read(cfg['model_config']));ema=base.ParameterEMA(m,.998,200)
            torch.cuda.synchronize();tick=time.monotonic()
            for i in range(5):base.update(m,opt,ema,data,data['train'][i*256:(i+1)*256])
            torch.cuda.synchronize();secs=time.monotonic()-tick;tick=time.monotonic();base.infer(m,data,data['train'][:1024]);inf=time.monotonic()-tick
            records.append(dict(protocol=p,fold=case['fold'],seconds5updates=secs,inference1024_seconds=inf,**rec));del m,opt,ema;gc.collect();torch.cuda.empty_cache()
        del data;gc.collect();torch.cuda.empty_cache()
    full=ridge.full_shape_witness(max(r['fit_targets'] for r in prepared['cases']),columns=512)
    mr=[];module,mrfit,transform,setseed=rocket_functions();options=read(cfg['model_config'])['multirocket']
    for p in cfg['protocols']:
        case=max([r for r in prepared['cases'] if r['protocol']==p],key=lambda r:r['fit_targets']);data=load_rocket(cfg,p,case['fold']);y=data['y'];ids=np.r_[tuple(np.flatnonzero(y==c)[:32] for c in range(7))];assert len(ids)==224
        np.random.seed(17);setseed(17);trainparts=[];queryparts=[];branchseconds=[]
        for key in ['native','state']:
            x=np.ascontiguousarray(data['fit'][key].transpose(0,2,1));pars=mrfit(module,x[ids],options['requested_features_per_branch'],options)
            tick=time.monotonic();query=transform(module,x[:1024],pars,options);seconds=time.monotonic()-tick
            trainparts.append(transform(module,x[ids],pars,options));queryparts.append(query);branchseconds.append(seconds)
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import RidgeClassifier
        trainparts += [data['fit']['context'][ids],data['fit']['current'][ids]];queryparts += [data['fit']['context'][:1024],data['fit']['current'][:1024]]
        xtr=np.concatenate(trainparts,-1);xq=np.concatenate(queryparts,-1);assert xtr.shape[1]==49890
        scaler=StandardScaler(copy=False);xtr=scaler.fit_transform(xtr);xq=scaler.transform(xq);weights=data['class_weights'][y[ids]].astype(np.float32).astype(np.float64)
        fit=ridge.fit(xtr,y[ids],weights);ref=RidgeClassifier(alpha=1,solver='cholesky').fit(np.asarray(xtr,np.float64),y[ids],sample_weight=weights)
        difference=float(np.max(np.abs(ridge.predict_scores(xq,fit)-ref.decision_function(np.asarray(xq,np.float64)))));assert difference<1e-6
        np.testing.assert_array_equal(ridge.predict(xq,fit),ref.predict(np.asarray(xq,np.float64)))
        mr.append(dict(protocol=p,branch_transform1024_seconds=branchseconds,features=49890,small_Fit_sklearn_max_score_error=difference,small_Fit_residual=fit['relative_system_residual']))
        del data,xtr,xq,trainparts,queryparts,fit,ref;gc.collect();torch.cuda.empty_cache()
    neural_estimate=0.;rocket_estimate=0.;details=[]
    for case in prepared['cases']:
        p=case['protocol'];n,nt=case['fit_targets'],case['test_targets']
        for arm in cfg['neural_models']:
            r=next(r for r in records if r['protocol']==p and r['model']==arm)
            seconds=(r['seconds5updates']/5*((n+255)//256)*20+r['inference1024_seconds']/1024*nt*4)*1.5+30
            assert seconds<=cfg['budget']['neural_job_seconds'];neural_estimate+=seconds*3;details.append(dict(**case,model=arm,estimated_job_seconds=seconds))
        r=next(r for r in mr if r['protocol']==p)
        seconds=(sum(r['branch_transform1024_seconds'])*(n+nt)/1024+full['gram_seconds']/512*49890*(n/full['n'])**2+full['solve_seconds']*(n/full['n'])**3)*1.7+60
        assert seconds<=cfg['budget']['rocket_job_seconds'];rocket_estimate+=seconds*3;details.append(dict(**case,model=cfg['cpu_model'],estimated_job_seconds=seconds))
    total=neural_estimate+rocket_estimate+600
    save(out/'PREFLIGHT.json',dict(status='PASS' if total<cfg['budget']['total_seconds'] else 'OVER_BUDGET',neural=records,rocket=mr,full_shape=full,estimates=details,estimated_neural_seconds=neural_estimate,estimated_rocket_seconds=rocket_estimate,estimated_total_seconds=total,code=implementation(cfg),new_Test_scores=0))
    assert total<cfg['budget']['total_seconds'];lock.close()

def seal(cfg):
    out=Path(cfg['output']);assert not (out/'SEAL.json').exists();code=implementation(cfg);evidence={};assert shutil.disk_usage(out).free/2**30>=cfg['budget']['minimum_free_disk_GiB']
    for n in ['RUNTIME_WITNESS.json','INDEPENDENT_RUNTIME_WITNESS.json']:
        r=read(out/n);assert r['status']=='PASS' and r['implementation_files']==code;evidence[n]=sha(out/n)
    r=read(out/'PREFLIGHT.json');assert r['status']=='PASS' and r['code']==code;evidence['PREFLIGHT.json']=sha(out/'PREFLIGHT.json')
    prepared=read(out/'PREPARED.json')
    for f,h in prepared['files'].items():assert sha(f)==h,f
    evidence['PREPARED.json']=sha(out/'PREPARED.json')
    save(out/'SEAL.json',dict(config=cfg,code=code,files=prepared['files'],evidence=evidence,unix=time.time(),new_Test_scores_before_seal=0))

def check(cfg,full=False):
    out=Path(cfg['output']);r=read(out/'SEAL.json');assert r['config']==cfg and r['code']==implementation(cfg)
    for f,h in r['evidence'].items():assert sha(out/f)==h
    if full:
        for f,h in r['files'].items():assert sha(f)==h,f

def neural(cfg,p,fold,seed,arm):
    check(cfg);lock=core.acquire_gpu();d=Path(cfg['output'])/'models'/p/fold/arm/f'seed{seed}';assert not (d/'STARTED.json').exists();save(d/'STARTED.json',dict(pid=os.getpid(),unix=time.time()))
    started=time.monotonic();data=formal.load_nn_data(read(cfg['parent_config']),p,fold);core.seed_all(seed);mcfg=read(cfg['model_config']);m=model_for(cfg,arm,data['prior'],'cuda');opt=base.optimizer_for(m,mcfg);ema=base.ParameterEMA(m,.998,200);history=[]
    for epoch in range(1,21):
        tick=time.monotonic();order=np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(data['train']);num=den=0.
        for offset in range(0,len(order),256):
            loss,mass=base.update(m,opt,ema,data,order[offset:offset+256]);num+=loss*mass;den+=mass
        history.append(dict(epoch=epoch,fit_ce=num/den,seconds=time.monotonic()-tick));save(d/'HISTORY.json',history)
        base.checkpoint(d/'latest.pt',m,opt,ema,epoch=epoch,protocol=p,fold=fold,seed=seed,model_name=arm)
        save(d/'PROGRESS.json',dict(epoch=epoch,total_epochs=20,phase='NEURAL_FIT',elapsed_seconds=time.monotonic()-started));print(history[-1],flush=True)
    payload=torch.load(d/'latest.pt',map_location='cpu',weights_only=False)
    for view,net,key in [('raw',m,'model'),('ema',ema.model,'ema')]:
        prob=base.infer(net,data,data['evaluation']);other=model_for(cfg,arm,data['prior'],'cuda').eval().requires_grad_(view=='raw');other.load_state_dict(payload[key])
        np.testing.assert_array_equal(prob,base.infer(other,data,data['evaluation']));np.save(d/f'test_{view}.npy',prob);del other
    check(cfg);save(d/'COMPLETE.json',dict(status='COMPLETE',protocol=p,fold=fold,seed=seed,model=arm,epoch=20,raw_EMA_replay_exact=True,Test_labels_parsed=False,Test_scores_computed=0,fit_targets=len(data['train']),test_targets=len(data['evaluation']),elapsed_seconds=time.monotonic()-started,files={str(d/n):sha(d/n) for n in ['latest.pt','HISTORY.json','test_raw.npy','test_ema.npy']}));lock.close()

def rocket(cfg,p,fold,seed):
    import joblib
    from sklearn.preprocessing import StandardScaler
    check(cfg);lock=core.acquire_gpu();d=Path(cfg['output'])/'models'/p/fold/cfg['cpu_model']/f'seed{seed}';assert not (d/'STARTED.json').exists();save(d/'STARTED.json',dict(pid=os.getpid(),unix=time.time()));started=time.monotonic()
    def progress(**values):save(d/'PROGRESS.json',dict(**values,elapsed_seconds=time.monotonic()-started))
    data=load_rocket(cfg,p,fold);module,mrfit,transform,setseed=rocket_functions();options=read(cfg['model_config'])['multirocket'];np.random.seed(seed);random.seed(seed);setseed(seed)
    trainparts=[];testparts=[];pars=[];stages={}
    for key in ['native','state']:
        tick=time.monotonic();x=np.ascontiguousarray(data['fit'][key].transpose(0,2,1));xt=np.ascontiguousarray(data['test'][key].transpose(0,2,1));progress(stage=key+'_kernel_fit')
        par=mrfit(module,x,options['requested_features_per_branch'],options);pars.append(par);progress(stage=key+'_Fit_transform');trainparts.append(transform(module,x,par,options));progress(stage=key+'_Test_unlabeled_transform');testparts.append(transform(module,xt,par,options));stages[key]=time.monotonic()-tick;del x,xt
    trainparts += [data['fit']['context'],data['fit']['current']];testparts += [data['test']['context'],data['test']['current']]
    x=np.concatenate(trainparts,-1);xt=np.concatenate(testparts,-1);del trainparts,testparts;assert x.shape[1]==49890
    scaler=StandardScaler(copy=False);x=scaler.fit_transform(x);xt=scaler.transform(xt);y=data['y'];weights=data['class_weights'][y].astype(np.float32).astype(np.float64)
    model=ridge.fit(x,y,weights,block=cfg['ridge']['feature_block'],progress=progress);pred=ridge.predict(xt,model)+1
    joblib.dump(dict(parameters=pars,scaler=scaler,classifier=model,model_config=options),d/'model.joblib');other=joblib.load(d/'model.joblib');np.testing.assert_array_equal(pred,ridge.predict(xt,other['classifier'])+1)
    replay=[transform(module,np.ascontiguousarray(data['test'][key][:256].transpose(0,2,1)),par,options) for key,par in zip(['native','state'],other['parameters'])]
    replay += [data['test']['context'][:256],data['test']['current'][:256]];replay=other['scaler'].transform(np.concatenate(replay,-1));np.testing.assert_array_equal(pred[:256],ridge.predict(replay,other['classifier'])+1)
    missing=~((data['test']['native'][:,:,6]>.5).any(1)|data['test']['state_valid'].any(1));pred[missing]=int(data['prior'].argmax())+1;np.save(d/'test_predictions.npy',pred)
    check(cfg);save(d/'COMPLETE.json',dict(status='COMPLETE',protocol=p,fold=fold,seed=seed,model=cfg['cpu_model'],features=49890,fit_targets=len(y),test_targets=len(pred),Test_labels_parsed=False,Test_scores_computed=0,serialized_predictions_exact=True,input_kernel_replay=256,relative_system_residual=model['relative_system_residual'],peak_cuda_bytes=model['peak_cuda_bytes'],stages=stages,elapsed_seconds=time.monotonic()-started,files={str(d/n):sha(d/n) for n in ['model.joblib','test_predictions.npy']}));lock.close()

def queue(cfg):
    import signal
    out=Path(cfg['output']);check(cfg,True);assert not (out/'LAUNCH.json').exists();jobs=read(out/'JOB_MANIFEST.json')['jobs'];start=time.monotonic();active=None
    save(out/'LAUNCH.json',dict(pid=os.getpid(),started_unix=time.time(),jobs=150,budget=cfg['budget']))
    def snapshot(phase):save(out/'PUBLIC_PROGRESS.json',dict(phase=phase,pid=os.getpid(),updated_unix=time.time(),elapsed_seconds=time.monotonic()-start,completed=sum(j['status']=='completed' for j in jobs),total=len(jobs),jobs=jobs))
    try:
        for j in jobs:
            if time.monotonic()-start>=cfg['budget']['total_seconds']:raise RuntimeError('Total budget exhausted')
            p,f,s,m=j['protocol'],j['fold'],j['seed'],j['model'];log=out/'logs'/f'{p}_{f}_{m}_{s}.log';log.parent.mkdir(parents=True,exist_ok=True);limit=cfg['budget']['rocket_job_seconds' if j['kind']=='rocket' else 'neural_job_seconds']
            with log.open('wb') as h:
                active=subprocess.Popen([sys.executable,'-s','-u','-m',MODULE,'--mode','worker','--protocol',p,'--fold',f,'--seed',str(s),'--model',m],stdin=subprocess.DEVNULL,stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
                j.update(status='running',pid=active.pid,started_unix=time.time());tick=time.monotonic()
                while active.poll() is None:
                    snapshot('FORMAL_FITS_AND_PREDICTIONS')
                    if time.monotonic()-tick>limit or time.monotonic()-start>cfg['budget']['total_seconds']:raise RuntimeError('Scientific budget exhausted')
                    time.sleep(5)
            marker=out/'models'/p/f/m/f'seed{s}'/'COMPLETE.json'
            if active.returncode!=0 or not marker.exists():j['status']='failed';raise RuntimeError('Job failed; stopped without retry')
            j.update(status='completed',finished_unix=time.time());snapshot('FORMAL_FITS_AND_PREDICTIONS')
        check(cfg,True);snapshot('FREEZING_AND_SCORING')
        from scripts.report_comparators_fivefold_20261025 import report
        save(out/'COMPLETE.json',report(cfg));snapshot('COMPLETE_STOP_FOR_REVIEW')
    except BaseException:
        if active is not None and active.poll() is None:
            os.killpg(active.pid,signal.SIGTERM)
            try:active.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
        snapshot('FAILED_STOP_FOR_INSPECTION');save(out/'QUEUE_FAILURE.json',dict(status='FAILED',traceback=traceback.format_exc(),unix=time.time()));raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',required=True,choices=['cpu_witness','witness','prepare','preflight','seal','launch','queue','worker']);ap.add_argument('--witness-output');ap.add_argument('--protocol');ap.add_argument('--fold');ap.add_argument('--seed',type=int);ap.add_argument('--model');a=ap.parse_args();cfg=configure(a.mode in ['witness','preflight','worker']);out=Path(cfg['output'])
    if a.mode in ['cpu_witness','witness']:witness(cfg,a.witness_output or str(out/'RUNTIME_WITNESS.json'),a.mode=='cpu_witness')
    elif a.mode=='prepare':prepare(cfg)
    elif a.mode=='preflight':preflight(cfg)
    elif a.mode=='seal':seal(cfg)
    elif a.mode=='worker':
        assert a.protocol in cfg['protocols'] and a.fold in cfg['folds'] and a.seed in cfg['seeds']
        if a.model==cfg['cpu_model']:rocket(cfg,a.protocol,a.fold,a.seed)
        else:assert a.model in cfg['neural_models'];neural(cfg,a.protocol,a.fold,a.seed,a.model)
    elif a.mode=='queue':queue(cfg)
    else:
        check(cfg,True);assert not (out/'DISPATCH.json').exists() and not (out/'LAUNCH.json').exists()
        with (out/'queue.log').open('wb') as h:child=subprocess.Popen([sys.executable,'-s','-u','-m',MODULE,'--mode','queue'],stdin=subprocess.DEVNULL,stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
        save(out/'DISPATCH.json',dict(pid=child.pid,unix=time.time()));print(child.pid)

if __name__=='__main__':main()
