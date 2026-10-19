"""Official sealed-source reuse admission and per-fold Grid no-op check; zero fits."""
import time
START,WALL=time.monotonic(),time.time()
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.sensor_motion_architecture_formal_common import CONFIG,read,write,sha,setup,get_data,load_base,original_path
from scripts.sensor_motion_pair_model import MotionPairRefiner


def main():
    setup();cfg=read(CONFIG);root=Path(cfg['preflight_output'])
    if root.exists():raise RuntimeError('Existing formal admission preserved')
    root.mkdir(parents=True);write(root/'BUDGET_START.json',dict(unix=WALL,limit=1800))
    def guard():
        if time.monotonic()-START>1800:raise RuntimeError('Admission budget exceeded')
    try:
        old=read('reports/sensor_motion_pair_formal_preflight_20261009/COMPLETE.json');assert old['status']=='PASS'
        identity=old['inputs'].copy()
        for p,h in identity.items():assert sha(p)==h;guard()
        for p in [CONFIG,Path(__file__),Path('scripts/sensor_motion_architecture_formal_common.py'),Path('scripts/sensor_motion_architecture_model.py'),Path('scripts/sensor_motion_architecture_common.py')]:identity[str(p)]=sha(p)
        inherited=read('reports/sensor_motion_pair_formal_20261009/EXPERIMENT_AUDIT.json');assert inherited['integrity_status']=='pass'
        for p in ['scripts/sensor_native_motion_features.py','scripts/sensor_motion_pair_model.py','q3_sensor/endpoint_models.py','q3_sensor/model.py','q3_sensor/protocol.py','scripts/sensor_motion_pair.py','scripts/sensor_local_tcn_residual.py','scripts/sensor_reliability_data.py']:
            h=inherited['audited_input_hashes'][p];assert sha(p)==h;identity[p]=h
        identity['reports/sensor_motion_pair_formal_20261009/EXPERIMENT_AUDIT.json']=sha('reports/sensor_motion_pair_formal_20261009/EXPERIMENT_AUDIT.json')
        records=[];grids={}
        for proto in cfg['protocols']:
            for fold in cfg['folds']:
                case=proto+'/'+fold;d=get_data(cfg,case);cache=Path(cfg['cache'])/case;receipt=read(cache/'COMPLETE.json')
                for key,h in receipt['identity']['bound_files'].items():assert sha(key)==h;identity[key]=h
                for name in ['targets.jsonl','normalizer.json','fit_statistics.json','COMPLETE.json']:identity[str(cache/name)]=sha(cache/name)
                olddir=Path(cfg['original_formal'][proto])/fold;op=pd.read_csv(olddir/'predictions.csv')
                original_rows=pd.read_json(olddir/'targets.jsonl',lines=True,convert_dates=False)
                original_indices=np.flatnonzero(original_rows.split.eq('train').to_numpy())
                pd.testing.assert_frame_equal(original_rows.iloc[original_indices].reset_index(drop=True),d['train'])
                identity[str(olddir/'targets.jsonl')]=sha(olddir/'targets.jsonl')
                assert sha(olddir/'predictions.csv')==receipt['identity']['legacy_predictions_sha256']
                pd.testing.assert_frame_equal(op[['cow_id','timestamp','label']],d['val'][['cow_id','timestamp','label']]);identity[str(olddir/'predictions.csv')]=sha(olddir/'predictions.csv')
                noop=all(np.allclose(n,g,atol=1e-6,rtol=1e-6) for n,g in zip(d['native'],d['grid']))
                grids[case]=dict(no_op=bool(noop),Fit_max_abs=float(np.abs(d['native'][0]-d['grid'][0]).max()),Test_max_abs=float(np.abs(d['native'][1]-d['grid'][1]).max()),Test_labels_used_for_no_op=False)
                for seed in cfg['seeds']:
                    ra=np.random.default_rng(seed);rb=np.random.default_rng(seed)
                    for epoch in range(20):
                        ia=ra.permutation(original_indices);ib=rb.permutation(len(d['train']))
                        np.testing.assert_array_equal(original_rows.iloc[ia][['cow_id','timestamp','label']].to_numpy(),d['train'].iloc[ib][['cow_id','timestamp','label']].to_numpy())
                    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                        from q3_sensor.endpoint_models import EndpointFusion
                        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);before=torch.cuda.get_rng_state().cpu().clone()
                        reconstructed=EndpointFusion('gru',d['stats']['prior']).cuda();after=torch.cuda.get_rng_state().cpu()
                        torch.testing.assert_close(before,after,atol=0,rtol=0)
                        torch.save(dict(state_dict={k:v.cpu() for k,v in reconstructed.state_dict().items()},CUDA_rng=before,CPU_rng_after_constructor=torch.get_rng_state(),historical_initial_hash_not_available=True),root/f'{proto}_{fold}_seed{seed}_initialization_witness.pt')
                        del reconstructed
                    base=load_base(cfg,case,seed).eval();ckpt=torch.load(original_path(cfg,case,seed),map_location='cpu',weights_only=False)
                    np.testing.assert_array_equal(ckpt['prior'],d['stats']['prior']);assert ckpt['normalizer']==read(cache/'normalizer.json')
                    basepred=[]
                    with torch.no_grad():
                        for at in range(0,len(d['val']),256):
                            idx=np.arange(at,min(at+256,len(d['val'])));basepred.append(base(d['vapool'].batch(idx)).argmax(1).cpu().numpy()+1)
                    np.testing.assert_array_equal(np.concatenate(basepred),op[f'NN_seed{seed}'].to_numpy());records.append(dict(case=case,seed=seed,arm='Base',exact=True))
                    for arm in ['BiasPair','ContextPair','NoMotionPair','MotionPair']:
                        folder=Path(cfg['reused_refiners'])/case/f'{arm}_seed{seed}';done=read(folder/'COMPLETE.json')
                        assert done['epochs']==20 and done['Base_frozen_exact'] and done['TEST_scoring'] is False
                        for name,h in done['files'].items():assert sha(folder/name)==h;identity[str(folder/name)]=h
                        identity[str(folder/'COMPLETE.json')]=sha(folder/'COMPLETE.json')
                        model=MotionPairRefiner(arm,d['stats']['prior'],ckpt['state_dict'],seed+600000).cuda()
                        initial=torch.load(folder/'initial_state.pt',map_location='cpu',weights_only=True)
                        for k,v in model.state_dict().items():torch.testing.assert_close(v.cpu(),initial[k],atol=0,rtol=0)
                        state=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
                        assert state['fit']=='author_TRAIN_plus_VAL';model.load_state_dict(state['state_dict']);model.eval();bb=[];aa=[];pp=[]
                        np.testing.assert_array_equal(read(folder/'motion_scaler.json')['mean'],d['native_scaler']['mean']);np.testing.assert_array_equal(read(folder/'motion_scaler.json')['std'],d['native_scaler']['std'])
                        with torch.no_grad():
                            for at in range(0,len(d['val']),256):
                                idx=np.arange(at,min(at+256,len(d['val'])));out=model(d['vapool'].batch(idx),torch.from_numpy(d['native'][1][idx]).cuda())
                                bb.append(out['base_logits'].cpu().numpy());aa.append(out['auxiliary_logits'].cpu().numpy());pp.append(out['prediction'].cpu().numpy())
                        prediction=pd.read_csv(folder/'predictions.csv');pd.testing.assert_frame_equal(prediction[['cow_id','timestamp','label']],d['val'][['cow_id','timestamp','label']])
                        np.testing.assert_array_equal(np.concatenate(bb),np.load(folder/'base_logits.npy'));np.testing.assert_array_equal(np.concatenate(aa),np.load(folder/'auxiliary_logits.npy'));np.testing.assert_array_equal(np.concatenate(pp),prediction.prediction.to_numpy())
                        records.append(dict(case=case,seed=seed,arm=arm,initial_exact=True,final_outputs_exact=True));del model;guard()
                    del base
        for p,h in identity.items():assert sha(p)==h;guard()
        write(root/'COMPLETE.json',dict(status='PASS',seconds=time.monotonic()-START,inputs=identity,records=records,reused_models=150,new_classifier_fits=0,grid=grids,Grid_skipped_protocols=[c for c,v in grids.items() if v['no_op']],new_Test_metrics_computed=False,existing_final_Test_prediction_replayed=True))
        print('FORMAL_ABLATION_ADMISSION_PASS',len(records),grids,flush=True)
    except Exception as e:write(root/'FAILURE.json',dict(error=repr(e),seconds=time.monotonic()-START,new_classifier_fits=0));raise


if __name__=='__main__':main()
