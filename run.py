"""Public entry point: data, model interface and result reconstruction."""
from pathlib import Path
import argparse,json,sys,hashlib
ROOT=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser();p.add_argument('--task',required=True,choices=['inspect','data','verify-results','model-check']);a,rest=p.parse_known_args()
 if a.task=='data':
  from tools.download_data import main as run
  return run(rest)
 if a.task=='verify-results':
  from tools.verify_results import main as run
  return run()
 if rest:p.error('unrecognized arguments: '+' '.join(rest))
 if a.task=='inspect':
  records=json.loads((ROOT/'data/CODE_MANIFEST.json').read_text('utf8'))
  for r in records:assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']
  print(json.dumps(dict(status='PASS_FROZEN_CODE_HASHES',code_files=len(records),model=json.loads((ROOT/'configs/model.json').read_text('utf8')))));return
 import torch
 from models.ttcsf import EndpointInteractionConvTran
 cfg=json.loads((ROOT/'configs/model.json').read_text('utf8'));torch.manual_seed(17);torch.set_num_threads(2)
 model=EndpointInteractionConvTran(cfg,[1/7]*7).eval()
 batch=dict(dynamic=torch.zeros(2,100,8),state=torch.zeros(2,60,26),state_valid=torch.ones(2,60,dtype=torch.bool),context=torch.zeros(2,136),current=torch.zeros(2,26));batch['dynamic'][:,:,6]=1
 with torch.no_grad():y=model(batch)
 assert tuple(y.shape)==(2,7) and torch.isfinite(y).all()
 assert sum(x.numel() for x in model.parameters())==213119
 print(json.dumps(dict(status='PASS_UNTRAINED_MODEL_INTERFACE',logits_shape=list(y.shape),parameters=213119,scientific_scores_computed=False,training_started=False)))
if __name__=='__main__':main()
