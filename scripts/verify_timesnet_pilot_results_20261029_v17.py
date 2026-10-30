"""Read-only result verification: source hashes, original gates and saved metrics."""
import json
from pathlib import Path
import time
from scripts import run_timesnet_pilot_20261029_v17 as run


def main():
    cfg=run.setup(False);seal=run.check(cfg);root=Path(cfg['output'])
    records=[]
    for protocol in cfg['protocols']:
        folder=run.directory(cfg,protocol)
        receipt=run.read(folder/'COMPLETE.json');run.verify(receipt['files'])
        history=run.read(folder/'history.json')
        assert receipt['stopping']==run.bounded_stop(history,cfg)
        assert receipt['selected_checkpoint']==run.selected(history)
        assert receipt['terminal_checkpoint']==history[-1]
        assert receipt['scientific_training_stability_pass']
        d=Path(cfg['data_parent'])/'data'/protocol/'inner0'
        with run.np.load(d/'FIT.npz',allow_pickle=False) as z:fit_truth=z['labels'].copy()
        dev_truth=run.np.asarray([json.loads(line)['label']-1 for line in (d/'DEV_TARGETS.jsonl').read_text().splitlines()])
        counts=run.np.bincount(fit_truth,minlength=7)
        weights=run.np.sqrt(counts.sum()/(7*counts)).astype(run.np.float32).astype(run.np.float64)
        for name,truth,expected in [('selected_DEV_EMA.npy',dev_truth,receipt['selected_checkpoint']['dev_ema']),
                                    ('selected_FIT_EMA.npy',fit_truth,receipt['selected_checkpoint']['fit_ema']),
                                    ('terminal_DEV_EMA.npy',dev_truth,history[-1]['dev_ema'])]:
            actual,_=run.metric(run.np.load(folder/name,allow_pickle=False),truth,weights)
            assert actual==expected,(protocol,name)
        chosen=receipt['selected_checkpoint'];last=history[-1]
        records.append(dict(protocol=protocol,epochs=len(history),selected_epoch=chosen['epoch'],
            selected_DEV=chosen['dev_ema'],selected_FIT=chosen['fit_ema'],terminal_DEV=last['dev_ema'],
            training_stability_pass=True,validation_plateau=True,
            selected_F1_gap_pp=receipt['best_vs_terminal']['selected_FIT_DEV_gap_pp'],
            terminal_F1_gap_pp=receipt['best_vs_terminal']['terminal_FIT_DEV_gap_pp'],
            terminal_CE_growth_relative=last['dev_ema']['weighted_CE']/chosen['dev_ema']['weighted_CE']-1,
            legacy_terminal_generalization_pass=receipt['legacy_terminal_generalization']['passed'],
            elapsed_seconds=receipt['elapsed_seconds'],files_verified=True,saved_metrics_exact=True,
            stopping_recomputed_exact=True,selected_checkpoint_replay_exact_from_training_receipt=receipt['selected_checkpoint_replay_exact']))
    run.check(cfg)
    out=root/'VERIFIED_PILOT_SUMMARY.json'
    value=dict(status='VERIFIED_PILOT_RESULTS',verified_unix=time.time(),records=records,
               scientific_training_updates=0,Test_rows_read=0,new_GPU_inference=0,
               boundary='Two internal development pilots, not full fivefold; empirical training stability passed. Terminal generalization guards retained separately; selected early points are not certified stationary.')
    if out.exists():assert run.read(out)['records']==records
    else:run.save(out,value)
    print(json.dumps(value,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
