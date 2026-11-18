"""Offline frozen prediction recount; only NumPy is required."""
from pathlib import Path
import argparse,csv,hashlib,json,statistics,tarfile
import numpy as np
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metric(prob,y):
 assert prob.shape==(len(y),7) and np.isfinite(prob).all()
 assert (prob>=-1e-7).all() and np.allclose(prob.sum(1),1,atol=2e-5)
 cm=np.bincount(y*7+prob.argmax(1),minlength=49).reshape(7,7)
 den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(7),where=den>0)*100
 return cm,float(f.mean()),f
def margin(x,b):
 mx=np.diff(np.sort(x,axis=1)[:,-2:],axis=1)[:,0]*.7;mb=np.diff(np.sort(b,axis=1)[:,-2:],axis=1)[:,0]*.3
 w=np.divide(mx,mx+mb,out=np.full(len(mx),.7),where=mx+mb>1e-15)
 return w[:,None]*x+(1-w[:,None])*b
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--bundle',type=Path,default=Path(__file__).parent);ap.add_argument('--archive',type=Path);ap.add_argument('--extracted',type=Path);ap.add_argument('--output',type=Path,default=Path('recount_output'));a=ap.parse_args();base=a.bundle
 index=read(base/'CLOUD_INDEX.json')
 if a.archive:
  manifest=read(base/'ARCHIVE_MANIFEST.json');assert sha(a.archive)==manifest['archive_sha256']
  root=a.output/'restored';root.mkdir(parents=True,exist_ok=True)
  with tarfile.open(a.archive) as tar:
   for member in tar.getmembers():
    assert (member.isfile() or member.islnk()) and (root/member.name).resolve().is_relative_to(root.resolve())
    if member.islnk():assert (root/member.linkname).resolve().is_relative_to(root.resolve())
   tar.extractall(root)
 else:root=a.extracted;assert root is not None,'Pass --archive or --extracted'
 for rel,v in index['files'].items():assert sha(root/rel)==v['sha256'],rel
 rows=[];cms={};checked=0;summary={}
 for arm,m in index['arms'].items():
  result=read(base/'results'/arm/'RESULTS.json');records=result['records'];assert len(records)==30
  for r in records:
   p,f,seed=r['protocol'],r['fold'],r['seed'];y=np.array([json.loads(s)['label']-1 for s in (root/m['targets'][p+'/'+f]).read_text().splitlines()]);assert ((y>=0)&(y<7)).all()
   for expert in ['tree','neural','fusion']:
    suffix=f'/fusion_predictions/{p}/{f}/'+('XGB.npy' if expert=='tree' else f'fusion_seed{seed}.npy')
    candidates=[z for z in m['predictions'] if z.endswith(suffix)] if expert!='neural' else [z for z in m['predictions'] if f'/prediction/{p}/{f}/' in z and z.endswith(f'/seed{seed}/test_ema.npy')]
    assert len(candidates)==1,(arm,expert,candidates)
    cm,v,per=metric(np.load(root/candidates[0],allow_pickle=False),y);assert cm.tolist()==r[expert]['confusion_matrix'];assert abs(v-r[expert]['macro_f1_percent'])<1e-10;assert np.max(np.abs(per-r[expert]['class_f1_percent']))<1e-10
    key=f'{arm}/{p}/{f}/{seed}/{expert}';cms[key]=cm.tolist();rows.append(dict(arm=arm,protocol=p,fold=f,seed=seed,expert=expert,macro_f1_percent=v));checked+=1
  summary[arm]={}
  for p in ['TS','OS']:
   summary[arm][p]={}
   for expert in ['tree','neural','fusion']:
    folds=[statistics.mean(z['macro_f1_percent'] for z in rows if z['arm']==arm and z['protocol']==p and z['expert']==expert and z['fold']==f'fold_{i}') for i in range(1,6)]
    target=read(base/'ANALYSIS.json')['summary'][arm][p][expert];assert abs(statistics.mean(folds)-target['mean'])<1e-10;assert abs(statistics.stdev(folds)-target['fold_sd'])<1e-10
    summary[arm][p][expert]={'mean':statistics.mean(folds),'fold_sd':statistics.stdev(folds),'fold_means':folds}
 fixed=read(base/'results/fixed_fusion_new_reference/RESULTS.json');weights={(r['protocol'],r['fold']):r['alpha'] for r in fixed['weights']};m=index['arms']['full']
 for r in fixed['records']:
  p,f,seed=r['protocol'],r['fold'],r['seed'];y=np.array([json.loads(s)['label']-1 for s in (root/m['targets'][p+'/'+f]).read_text().splitlines()]);x=np.load(root/next(z for z in m['predictions'] if z.endswith(f'/fusion_predictions/{p}/{f}/XGB.npy')));b=np.load(root/next(z for z in m['predictions'] if f'/prediction/{p}/{f}/' in z and z.endswith(f'/seed{seed}/test_ema.npy')));alpha=weights[p,f]
  prob={'equal_0.5':lambda:.5*x+.5*b,'fixed_0.7':lambda:.7*x+.3*b,'validation_fixed':lambda:alpha*x+(1-alpha)*b,'margin_0.7':lambda:margin(x,b)}[r['rule']]();cm,v,per=metric(prob,y);assert cm.tolist()==r['confusion_matrix'];assert abs(v-r['macro_f1'])<1e-10;checked+=1
  rows.append(dict(arm='fixed_fusion_new_reference',protocol=p,fold=f,seed=seed,expert=r['rule'],macro_f1_percent=v));cms[f'fixed/{p}/{f}/{seed}/{r["rule"]}']=cm.tolist()
 a.output.mkdir(exist_ok=True,parents=True)
 with (a.output/'fold_seed_metrics.csv').open('w',newline='',encoding='utf-8') as h:
  w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 for name,obj in [('confusion_matrices.json',cms),('summary.json',summary),('VERIFICATION.json',{'status':'PASS_FROZEN_PREDICTIONS_TO_CM_AND_AGGREGATION','source_files_sha256_checked':len(index['files']),'frozen_prediction_files':index['prediction_files'],'confusion_matrices_recounted':checked,'scope':'file identity and frozen predictions against packaged dataset target labels; no retraining, source annotation audit, or independent external validation'})]:
  (a.output/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(read(a.output/'VERIFICATION.json')))
if __name__=='__main__':main()
