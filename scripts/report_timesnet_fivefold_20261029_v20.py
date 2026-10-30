"""Freeze all 30 TimesNet predictions before opening any Test target bytes."""
import csv
import json
from fractions import Fraction
from pathlib import Path
import numpy as np
from scripts import timesnet_fivefold_common_20261029_v20 as c
from scripts.report_shared_fusion_fivefold_20261024 import evaluate,exact_decimal
from scripts.report_convtran_context_20261022 import CLASSES


def csv_write(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def aggregate(cfg,models,metrics,fractions):
    summary={};fold_rows=[]
    for model in models:
        summary[model]={}
        for p in cfg['protocols']:
            folds=[];classes=[]
            for fold in cfg['folds']:
                scores=[fractions[p,fold,model,s] for s in cfg['seeds']]
                mean=sum(scores,Fraction())/len(scores);folds.append(mean)
                classes.append(np.mean([metrics[p,fold,model,s]['class_f1_percent'] for s in cfg['seeds']],axis=0))
                fold_rows.append(dict(protocol=p,fold=fold,model=model,macro_F1_percent=float(mean),seed_SD_pp=float(np.std([float(x) for x in scores],ddof=1))))
            mean=sum(folds,Fraction())/len(folds)
            summary[model][p]=dict(macro_F1_percent=float(mean),unrounded_decimal=exact_decimal(mean),
                per_fold_F1_percent=[float(x) for x in folds],fold_SD_pp=float(np.std([float(x) for x in folds],ddof=1)),
                class_F1_percent=dict(zip(CLASSES,np.mean(classes,axis=0).tolist())))
    return summary,fold_rows


def report(cfg):
    from scripts.run_timesnet_fivefold_20261029_v20 import receipt
    c.check(cfg);lock=c.check_training_lock(cfg);out=Path(cfg['output']);models=lock['eligible_models']
    assert models==['TimesNet_DS'] and not (out/'PREDICTIONS_FROZEN.json').exists()
    files={};gold={};jobs=[]
    for p in cfg['protocols']:
        for fold in cfg['folds']:
            source=c.read(Path(cfg['formal_parent'])/'data'/p/fold/'DATA_RECEIPT.json')
            target=Path(cfg['formal_parent'])/'data'/p/fold/'TEST_TARGETS.jsonl'
            gold[str(target)]=source['outputs']['TEST_TARGETS.jsonl']
            for model in models:
                for seed in cfg['seeds']:
                    job=dict(stage='prediction',protocol=p,fold=fold,model=model,seed=seed)
                    r=receipt(cfg,job);d=c.case_dir(cfg,'prediction',p,fold,model,seed)
                    assert r['status']=='COMPLETE_PREDICTION' and not r['Test_labels_parsed']
                    train=c.case_dir(cfg,'refit',p,fold,model,seed)/'COMPLETE.json'
                    assert c.sha(train)==lock['files'][str(train)]
                    for name,digest in r['files'].items():
                        file=Path(name);file=d/file if not file.is_absolute() and len(file.parts)==1 else file
                        assert c.sha(file)==digest;files[str(file)]=digest
                    files[str(d/'COMPLETE.json')]=c.sha(d/'COMPLETE.json');jobs.append(job)
    assert len(jobs)==30
    c.save(out/'PREDICTIONS_FROZEN.json',dict(status='PASS',files=files,expected_Test_hashes=gold,
        models=models,predictions=len(jobs),training_lock_sha256=c.sha(out/'TRAINING_LOCK.json'),Test_targets_opened=False))
    # Test target bytes remain inaccessible until every prediction is frozen above.
    c.save(out/'SCORING_STARTED.json',dict(prediction_lock_sha256=c.sha(out/'PREDICTIONS_FROZEN.json')))
    rows=[];class_rows=[];metrics={};fractions={}
    for p in cfg['protocols']:
        for fold in cfg['folds']:
            target=Path(cfg['formal_parent'])/'data'/p/fold/'TEST_TARGETS.jsonl'
            assert c.sha(target)==gold[str(target)]
            truth=np.array([json.loads(line)['label'] for line in target.read_text().splitlines()])
            assert np.isin(truth,np.arange(1,8)).all()
            for model in models:
                for seed in cfg['seeds']:
                    directory=c.case_dir(cfg,'prediction',p,fold,model,seed)
                    prob=np.load(directory/'test_ema.npy',allow_pickle=False)
                    value,exact=evaluate(truth,prob)
                    key=(p,fold,model,seed);metrics[key]=value;fractions[key]=exact
                    rows.append(dict(protocol=p,fold=fold,model=model,seed=seed,**value))
                    for k,name in enumerate(CLASSES):class_rows.append(dict(protocol=p,fold=fold,model=model,seed=seed,behavior=name,F1_percent=value['class_f1_percent'][k]))
    summary,fold_rows=aggregate(cfg,models,metrics,fractions)
    c.save(out/'PER_SEED_METRICS.json',rows);csv_write(out/'PER_FOLD.csv',fold_rows);csv_write(out/'PER_CLASS.csv',class_rows)
    result=dict(status='COMPLETE_FIVEFOLD',summary=summary,completed_models=models,held_models=lock['held_models'],
        final_fits=len(jobs),protocols=2,folds=5,seeds=3,
        aggregation='Mean3seeds within fold then equal mean5folds; folds, not15seeds, are reporting units',
        Test_used_for_selection=False,historical_Test_exposure=True,new_blind_evaluation=False,
        neural_endpoint='fresh TRAIN+VAL refit at calibration best CE epoch with frozen actual LR prefix and causal3EMA',
        refit_terminal_stationarity_claimed=False,no_overfitting_claimed=False,
        retained_dropout=.1,Historical_GRU_included=False,TSLANet_included=False)
    c.save(out/'RESULTS.json',result)
    return result
