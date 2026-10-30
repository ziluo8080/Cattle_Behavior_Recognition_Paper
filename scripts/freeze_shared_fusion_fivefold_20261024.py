"""Seal all ten formal caches and forty fixed training jobs before any fit."""
from pathlib import Path
import time

from scripts.shared_fusion_fivefold_common_20261024 import CONFIG,configure,read,save,sha,check_method_lock,check_data,code_files


def jobs(cfg):
    out=Path(cfg['output']);result=[]
    for fold in cfg['folds']:
        for protocol in cfg['protocols']:
            result.append(dict(id=f'XGB_{protocol}_{fold}',resource='cpu',module='scripts.run_shared_fusion_fivefold_20261024',
                args=['--mode','tree','--protocol',protocol,'--fold',fold],depends_on=[],
                expected=str(out/'tree'/protocol/fold/'COMPLETE.json'),timeout_seconds=cfg['budget']['tree_job_seconds']))
            for seed in cfg['seeds']:
                result.append(dict(id=f'B_{protocol}_{fold}_{seed}',resource='gpu',module='scripts.run_shared_fusion_fivefold_20261024',
                    args=['--mode','neural','--protocol',protocol,'--fold',fold,'--seed',str(seed)],depends_on=[],
                    expected=str(out/'B'/protocol/fold/f'seed{seed}'/'COMPLETE.json'),timeout_seconds=cfg['budget']['neural_job_seconds']))
    assert len(result)==len({j['id'] for j in result})==40
    assert sum(j['resource']=='gpu' for j in result)==30
    return result


def check_training_seal(cfg):
    check_method_lock(cfg);out=Path(cfg['output']);r=read(out/'SEAL.json')
    assert r['config']==cfg and sha(out/'METHOD_LOCK.json')==r['method_lock_sha256']
    assert sha(out/'JOB_MANIFEST.json')==r['job_manifest_sha256']
    for path,digest in r['receipts'].items():assert sha(path)==digest,path
    return r


def main():
    cfg=configure();out=Path(cfg['output']);assert not (out/'SEAL.json').exists()
    check_method_lock(cfg,full_sources=True)
    for name in ('CODE_REVIEW_GATE.json','RUNTIME_WITNESS.json','INDEPENDENT_RUNTIME_WITNESS.json',
                 'DEVELOPMENT_INPUT_REPLAY.json','FIVEFOLD_COVERAGE.json','PREFLIGHT.json'):
        assert read(out/name)['status']=='PASS',name
    for name in ('RUNTIME_WITNESS.json','INDEPENDENT_RUNTIME_WITNESS.json'):
        for path,digest in read(out/name)['implementation_files'].items():assert sha(path)==digest,(name,path)
    assert read(out/'RUNTIME_WITNESS.json')['scope']=='synthetic only'
    pre=read(out/'PREFLIGHT.json');assert len(pre['records'])==10
    assert {(r['protocol'],r['fold']) for r in pre['records']}=={(p,f) for p in cfg['protocols'] for f in cfg['folds']}
    assert pre['conservative_serial_seconds']<=cfg['budget']['total_seconds']
    for path,digest in pre['implementation_files'].items():assert sha(path)==digest,path
    receipts={};data_files={}
    for p in cfg['protocols']:
        for f in cfg['folds']:
            directory=out/'data'/p/f;r=check_data(cfg,p,f)
            receipts[str(directory/'DATA_RECEIPT.json')]=sha(directory/'DATA_RECEIPT.json')
            data_files.update({str(directory/k):v for k,v in r['outputs'].items()})
    save(out/'JOB_MANIFEST.json',dict(jobs=jobs(cfg),total_seconds=cfg['budget']['total_seconds'],
        max_parallel_cpu=cfg['budget']['cpu_parallel'],max_parallel_gpu=cfg['budget']['gpu_parallel'],max_attempts=1))
    for name in ('PREFLIGHT.json','FIVEFOLD_COVERAGE.json','DEVELOPMENT_INPUT_REPLAY.json'):
        receipts[str(out/name)]=sha(out/name)
    save(out/'SEAL.json',dict(status='PASS',config=cfg,method_lock_sha256=sha(out/'METHOD_LOCK.json'),
        job_manifest_sha256=sha(out/'JOB_MANIFEST.json'),receipts=receipts,data_files=data_files,created_unix=time.time(),
        scientific_Test_scores_before_seal=0,scientific_final_models_before_seal=0))
    print('FIVEFOLD_TRAINING_SEAL_PASS',flush=True)


if __name__=='__main__':main()
