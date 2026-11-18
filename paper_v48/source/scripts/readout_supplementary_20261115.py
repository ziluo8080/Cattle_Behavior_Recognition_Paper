"""Cloud-only independent CM recount and paired fivefold descriptive readout."""
from pathlib import Path
from fractions import Fraction
import json,math,statistics,time
ROOT=Path('/q3/mmcows-sensor-behavior-20261007');OUT=ROOT/'reports/sensor_supplementary_readout_20261115'
PATHS={'full':'reports/sensor_fair_selection_20261113','no_eri':'reports/sensor_recovery_20261114/runs/no_eri','short_10_10':'reports/sensor_recovery_20261114/runs/short_10_10','long_60_60':'reports/sensor_long60_scale1_corrected_20261114','dynamic_only':'reports/sensor_recovery_20261114/runs/dynamic_only','state_only':'reports/sensor_recovery_20261114/runs/state_only','no_uwb':'reports/sensor_recovery_20261114/runs/no_uwb','no_ankle':'reports/sensor_recovery_20261114/runs/no_ankle'}
def load(p):return json.loads(Path(p).read_text())
def f1(cm):
 assert len(cm)==7 and all(len(r)==7 and all(isinstance(v,int) and v>=0 for v in r) for r in cm)
 out=[]
 for i in range(7):
  d=sum(cm[i])+sum(r[i] for r in cm);out.append(Fraction(2*cm[i][i],d)*100 if d else Fraction())
 return float(sum(out)/7),[float(v) for v in out]
def stats(v):return {'mean':statistics.mean(v),'fold_sd':statistics.stdev(v),'fold_means':v}
def main():
 assert Path.cwd()==ROOT;OUT.mkdir(exist_ok=True);raw={k:load(ROOT/p/'RESULTS.json')['records'] for k,p in PATHS.items()};truth={};checked=0;summary={};classes={};pairs={}
 for arm,rows in raw.items():
  assert len(rows)==30
  summary[arm]={};classes[arm]={}
  assert len({(r['protocol'],r['fold'],r['seed']) for r in rows})==30
  for row in rows:
   key=row['protocol'],row['fold'],row['seed']
   for expert in ['tree','neural','fusion']:
    m=row[expert];v,per=f1(m['confusion_matrix']);assert abs(v-m['macro_f1_percent'])<1e-10;assert all(abs(a-b)<1e-10 for a,b in zip(per,m['class_f1_percent']));checked+=1
    support=[sum(r) for r in m['confusion_matrix']]
    if key in truth:assert support==truth[key]
    else:truth[key]=support
  for p in ['TS','OS']:
   summary[arm][p]={};classes[arm][p]={}
   for expert in ['tree','neural','fusion']:
    fold=[statistics.mean(r[expert]['macro_f1_percent'] for r in rows if r['protocol']==p and r['fold']==f'fold_{i}') for i in range(1,6)];summary[arm][p][expert]=stats(fold)
    classes[arm][p][expert]=[statistics.mean(r[expert]['class_f1_percent'][i] for r in rows if r['protocol']==p) for i in range(7)]
 for arm in PATHS:
  pairs[arm]={}
  for p in ['TS','OS']:
   pairs[arm][p]={}
   for expert in ['tree','neural','fusion']:
    diff=[a-b for a,b in zip(summary[arm][p][expert]['fold_means'],summary['full'][p][expert]['fold_means'])]
    pairs[arm][p][expert]={'variant_minus_full_pp':statistics.mean(diff),'fold_deltas':diff,'higher_folds':sum(v>0 for v in diff),'lower_folds':sum(v<0 for v in diff),'same_folds':sum(v==0 for v in diff)}
 fixed=load(ROOT/'reports/sensor_recovery_20261114/fixed_fusion_new_reference/RESULTS.json');rules=['equal_0.5','fixed_0.7','validation_fixed','margin_0.7'];fusion={}
 assert len(fixed['records'])==120
 for row in fixed['records']:
  v,per=f1(row['confusion_matrix']);assert abs(v-row['macro_f1'])<1e-10;assert [sum(z) for z in row['confusion_matrix']]==truth[(row['protocol'],row['fold'],row['seed'])];checked+=1
 for p in ['TS','OS']:
  fusion[p]={}
  for rule in rules:
   fold=[statistics.mean(r['macro_f1'] for r in fixed['records'] if r['protocol']==p and r['rule']==rule and r['fold']==f'fold_{i}') for i in range(1,6)];fusion[p][rule]=stats(fold)
  assert abs(fusion[p]['margin_0.7']['mean']-summary['full'][p]['fusion']['mean'])<1e-10
 for p in ['TS','OS']:
  for seed in [17,23,29]:
   support=[sum(truth[p,f'fold_{i}',seed][j] for i in range(1,6)) for j in range(7)];assert sum(support)==46821
 analysis={'status':'PASS_CM_AND_AGGREGATION_RECOUNT','confusion_matrices_checked':checked,'all_variants_same_class_support_per_fold_seed':True,'aggregation':'average3seed metrics withinfold, equalmean5folds, sampleSD betweenfolds','summary':summary,'variant_minus_full':pairs,'class_means':classes,'fixed_fusion':fusion,'alpha_weights':fixed['weights'],'outputs':PATHS,'scope':'descriptive internalCV, no p-values or independent external validation','unix':time.time()}
 (OUT/'ANALYSIS.json').write_text(json.dumps(analysis,indent=2))
 print(json.dumps({'checked':checked,'fusion':fusion,'summary':{a:{p:summary[a][p]['fusion'] for p in ['TS','OS']} for a in PATHS},'neural_only':{a:{p:summary[a][p]['neural']['mean'] for p in ['TS','OS']} for a in PATHS},'delta':{a:{p:pairs[a][p]['fusion'] for p in ['TS','OS']} for a in PATHS}},ensure_ascii=False))
if __name__=='__main__':main()
