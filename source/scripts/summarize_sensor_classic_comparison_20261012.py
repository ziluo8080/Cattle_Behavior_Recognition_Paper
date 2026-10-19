"""Raw-GT saved-prediction rescoring; synchronized paired conditional intervals."""
from pathlib import Path
import numpy as np
import pandas as pd
from q3_sensor.evaluation import metrics
from q3_sensor.protocol import sha
from scripts.run_sensor_history_controls_20261011 import read_json, write_json
from scripts.run_sensor_classic_comparison_20261012 import setup, data_for, verify
from scripts.summarize_sensor_architecture_comparison_20261011 import f1, group_confusions


def main():
    cfg,root=setup()
    assert read_json(root/'FORMAL_COMPLETE.json')['expected_jobs']==120
    lock=read_json(root/'METHOD_LOCK.json')
    anchor=read_json(Path(cfg['reuse_architecture_root'])/'METHOD_LOCK.json')
    models=[cfg['reference_model']]+cfg['models']
    count=len(models)
    sources={str(root/'run_identity.json'):sha(root/'run_identity.json'),str(root/'METHOD_LOCK.json'):sha(root/'METHOD_LOCK.json')}
    gt={}
    for cow in range(1,11):
        path=Path(cfg['source'])/f'behavior_labels/individual/C{cow:02d}_0725.csv'
        values=pd.read_csv(path)
        assert not values.timestamp.duplicated().any()
        gt[cow]=values.set_index('timestamp').behavior
        sources[str(path)]=sha(path)
    output,fold_rows,seed_rows,class_rows,cm_rows,cost_rows={},[],[],[],[],[]
    for protocol in cfg['protocols']:
        loaded,all_times,unique=[],[],[]
        for fold in cfg['folds']:
            data=data_for(cfg,'formal',protocol,fold)
            expected=data['targets'].iloc[data['evaluation_ids']][['cow_id','timestamp','label']].reset_index(drop=True)
            for cow,records in expected.groupby('cow_id'):
                np.testing.assert_array_equal(records.label.to_numpy(),gt[int(cow)].loc[records.timestamp].to_numpy(dtype=int))
            predictions=[]
            for model in models:
                for seed in cfg['formal_seeds']:
                    if model==cfg['reference_model']:
                        directory=Path(cfg['reuse_architecture_root'])/'formal'/protocol/fold/'GRU'/f'seed{seed}'
                        complete=read_json(directory/'COMPLETE.json')
                        assert complete['identity_sha256']==cfg['reuse_architecture_identity_sha256']
                        assert complete['job_id']==f'formal/{protocol}/{fold}/GRU/seed{seed}'
                        for name,digest in complete['outputs'].items(): assert sha(directory/name)==digest
                        recipe=anchor['selected'][f'{protocol}/{fold}/GRU']
                    else:
                        job=f'formal/{protocol}/{fold}/{model}/seed{seed}'
                        directory=root/job
                        verify(directory,root,job,data['identity'])
                        recipe=lock['selected'][f'{protocol}/{fold}/{model}']
                    result=read_json(directory/'result.json')
                    assert result['seed']==seed and result['trained_epochs']==recipe['epochs']
                    if model in (cfg['reference_model'],'CurrentRepeat-GRU-VAL'):
                        assert result['learning_rate']==recipe['learning_rate']
                        history=read_json(directory/'history.json')
                        assert len(history)==recipe['epochs'] and all('val_macro_f1' not in row for row in history)
                    else:
                        assert result['candidate_index']==recipe['candidate_index']
                    table=pd.read_csv(directory/'predictions.csv')
                    pd.testing.assert_frame_equal(table[['cow_id','timestamp','label']],expected)
                    pred=table.prediction.to_numpy(dtype=int)
                    assert np.isin(pred,range(1,8)).all()
                    scores=metrics(expected.label.to_numpy(dtype=int),pred)
                    for key,value in scores.items(): np.testing.assert_allclose(value,result['metrics'][key],rtol=0,atol=2e-15)
                    predictions.append(pred)
                    sources[str(directory/'COMPLETE.json')]=sha(directory/'COMPLETE.json')
                    cost_rows.append(dict(protocol=protocol,fold=fold,model=model,seed=seed,
                        seconds=result['seconds'],feature_seconds=result.get('feature_seconds'),
                        classifier_fit_seconds=result.get('classifier_fit_seconds'),
                        cost_scope='new_job_full_fit_and_evaluation' if model!=cfg['reference_model'] else 'reused_prior_job',
                        inference_latency_benchmark=False))
            labels=expected.label.to_numpy(dtype=int)
            times=expected.timestamp.to_numpy()//cfg['bootstrap']['block_seconds']
            all_times.append(times)
            unique.extend(zip(expected.cow_id,expected.timestamp))
            loaded.append(dict(fold=fold,table=expected,labels=labels,predictions=np.stack(predictions),times=times))
        assert len(unique)==len(set(unique))
        blocks=np.unique(np.concatenate(all_times))
        reps=cfg['bootstrap']['replicates']
        rng=np.random.default_rng(cfg['bootstrap']['seed'])
        draws=rng.multinomial(len(blocks),np.full(len(blocks),1/len(blocks)),size=reps)
        cowrng=np.random.default_rng(cfg['bootstrap']['seed'])
        bt=np.zeros((reps,count)); bc=np.zeros((reps,count)) if protocol=='OS' else None
        fold_scores,fold_classes=[],[]
        for row in loaded:
            matrices=group_confusions(row['labels'],row['predictions'],np.searchsorted(blocks,row['times']),len(blocks))
            totals=matrices.sum(1)
            classes=f1(totals).reshape(count,3,7); macro=classes.mean(2)
            fold_scores.append(macro); fold_classes.append(classes)
            sampled=np.einsum('rb,mbij->rmij',draws,matrices)
            assert (sampled.sum((-1,-2))>0).all()
            bt+=f1(sampled).mean(2).reshape(reps,count,3).mean(2)/5
            if protocol=='OS':
                cows,ids=np.unique(row['table'].cow_id,return_inverse=True)
                assert len(cows)==2
                cow_cm=group_confusions(row['labels'],row['predictions'],ids,2)
                cow_draws=cowrng.multinomial(2,np.full(2,.5),size=reps)
                bc+=f1(np.einsum('rb,mbij->rmij',cow_draws,cow_cm)).mean(2).reshape(reps,count,3).mean(2)/5
            for i,model in enumerate(models):
                fold_rows.append(dict(protocol=protocol,fold=row['fold'],model=model,macro_f1=float(macro[i].mean()),targets=len(row['labels'])))
                for j,seed in enumerate(cfg['formal_seeds']):
                    seed_rows.append(dict(protocol=protocol,fold=row['fold'],model=model,seed=seed,macro_f1=float(macro[i,j])))
                    cm=totals[i*3+j]
                    for truth in range(7):
                        for prediction in range(7):
                            cm_rows.append(dict(protocol=protocol,fold=row['fold'],model=model,seed=seed,
                                true_class=truth+1,predicted_class=prediction+1,count=int(cm[truth,prediction])))
        fold_scores,fold_classes=np.stack(fold_scores),np.stack(fold_classes)
        means=fold_scores.mean((0,2)); seed_means=fold_scores.mean(0)
        comparisons={}
        for left,right in cfg['primary_comparisons']:
            li,ri=models.index(left),models.index(right)
            comparisons[f'{left}_minus_{right}']=dict(delta=float(means[li]-means[ri]),
                conditional_time_CI95=np.quantile(bt[:,li]-bt[:,ri],[.025,.975]).tolist(),
                conditional_cow_CI95=None if bc is None else np.quantile(bc[:,li]-bc[:,ri],[.025,.975]).tolist(),
                per_fold_delta=(fold_scores[:,li].mean(1)-fold_scores[:,ri].mean(1)).tolist(),
                per_seed_delta=(seed_means[li]-seed_means[ri]).tolist())
        for i,model in enumerate(models):
            for k in range(7):class_rows.append(dict(protocol=protocol,model=model,class_id=k+1,f1=float(fold_classes[:,i,:,k].mean())))
        output[protocol]=dict(scores={model:dict(mean=float(means[i]),per_seed_fivefold_mean=seed_means[i].tolist(),
            seed_population_std=float(seed_means[i].std()),per_class_mean=fold_classes[:,i].mean((0,1)).tolist()) for i,model in enumerate(models)},
            comparisons=comparisons,unique_Test_targets=len(unique),shared_time_blocks=len(blocks))
    for p,digest in sources.items():assert sha(p)==digest
    write_json(root/'verified_summary.json',dict(protocols=output,sources=sources,original_GT_verified=True,
        saved_metrics_reconstructed=True,independent_audit='pending',historical_Test_exposure=True,
        new_blind_evaluation=False,multiple_comparison_correction=False,statistical_significance_claim=False,
        interval_scope='fixed_fitted_models_selected_recipes_existing_data_not_selection_refit_new_environments',
        mainline_automatically_replaced=False))
    for name,rows in [('fold_results',fold_rows),('seed_results',seed_rows),('class_results',class_rows),
                      ('confusion_matrices',cm_rows),('fit_costs',cost_rows)]:
        pd.DataFrame(rows).to_csv(root/f'{name}.csv',index=False)
    lines=['# 经典模型有限比较','', '官方TS/OS五折、每折三seed；先seed均值再五折均值。', '',
        '| 方法 | TS宏F1% | OS宏F1% |','| --- | ---: | ---: |']
    for model in models:lines.append(f"| {model} | {output['TS']['scores'][model]['mean']*100:.6f} | {output['OS']['scores'][model]['mean']*100:.6f} |")
    lines+=['','所有预定模型及负结果保留。原20轮主线与本VAL-refit配方分开。',
        '条件区间未校正多重比较，不包含重新训练、选型或新牧场。独立全批审计待完成。',
        '成本CSV为单次完整job拟合与评价成本，不能称端到端部署时延。','']
    (root/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    write_json(root/'AUTHOR_PIPELINE_COMPLETE.json',dict(status='fits_and_rescore_complete',
        independent_audit='pending',identity_sha256=sha(root/'run_identity.json'),mainline_automatically_replaced=False))
    print('CLASSIC_RESULTS_RECONSTRUCTED',flush=True)


if __name__=='__main__':main()
