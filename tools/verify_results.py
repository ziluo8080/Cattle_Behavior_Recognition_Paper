"""Recompute7-class and macroF1 from all240 preserved confusion matrices."""
from pathlib import Path
from collections import defaultdict
from statistics import mean,stdev
import json
ROOT=Path(__file__).resolve().parents[1]
def main():
 cmdata=json.loads((ROOT/'benchmarks/confusion_matrices.json').read_text('utf8'))
 expected=json.loads((ROOT/'benchmarks/expected_results.json').read_text('utf8'))['protocols']
 groups=defaultdict(dict)
 for row in cmdata['records']:
  cm=row['confusion_matrix'];assert len(cm)==7 and all(len(x)==7 for x in cm)
  f=[]
  for k in range(7):
   denom=sum(cm[k])+sum(r[k] for r in cm);f.append(200*cm[k][k]/denom if denom else 0.)
  f.append(mean(f));key=(row['protocol'],row['display_model']);case=(row['fold'],row['seed']);assert case not in groups[key];groups[key][case]=f
 output={};cells=0
 for (protocol,model),cases in groups.items():
  assert set(cases)=={(f'fold_{f}',s) for f in range(1,6) for s in [17,23,29]}
  folds=[[mean(cases[f'fold_{f}',s][c] for s in [17,23,29]) for c in range(8)] for f in range(1,6)]
  avg=[mean(r[c] for r in folds) for c in range(8)];sd=[stdev(r[c] for r in folds) for c in range(8)]
  ref=expected[protocol][model]
  assert max(abs(a-b) for a,b in zip(avg,ref['mean']))<1e-9,(protocol,model,'mean')
  assert max(abs(a-b) for a,b in zip(sd,ref['sd']))<1e-9,(protocol,model,'sd')
  output.setdefault(protocol,{})[model]=dict(mean=avg,sample_sd=sd);cells+=8
 assert cells==128 and len(cmdata['records'])==240
 out=ROOT/'_verification_output';out.mkdir(exist_ok=True);(out/'results.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),'utf8')
 print(json.dumps(dict(status='PASS_240_CM_128_F1_MEAN_SD_PAIRS',cases=240,F1_cells=128,training_started=False)))
if __name__=='__main__':main()
