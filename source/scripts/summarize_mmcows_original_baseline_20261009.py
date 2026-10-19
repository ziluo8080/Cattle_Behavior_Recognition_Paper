"""Retabulate existing audited folds; no model loading, training, or new scoring."""
import json
from pathlib import Path

root=Path('/q3/mmcows-sensor-behavior-20261007')
assert Path.cwd()==root
motion=json.loads(Path('reports/sensor_motion_pair_formal_20261009/verified_summary.json').read_text())
result={'source':'original_Table3_UWB_HD_Akl_and_existing_audited_formal_outputs','new_training':False,'new_inference':False,'new_Test_scoring':False,'protocols':{}}
for proto,paper_value,path in [('TS',.737,'reports/sensor_formal_ts_20261007/verified_summary.json'),('OS',.714,'reports/sensor_formal_os_20261007/verified_summary.json')]:
    old=json.loads(Path(path).read_text());folds=old['folds'];assert len(folds)==5
    rf=sum(f['RF_macro_f1'] for f in folds)/5
    rf_f1=[sum(f['RF_per_class_f1'][c] for f in folds)/5 for c in range(7)]
    base=motion['protocols'][proto]['scores']['Base'];full=motion['protocols'][proto]['scores']['MotionPair']
    legacy=sum(f['NN_macro_f1_mean'] for f in folds)/5
    assert abs(legacy-base['mean'])<1e-15
    result['protocols'][proto]={'paper_RF_mean':paper_value,'reproduced_RF_mean':rf,'Base_mean':base['mean'],'full_mean':full['mean'],
        'full_minus_reproduced_RF_pp':100*(full['mean']-rf),'full_minus_paper_RF_pp':100*(full['mean']-paper_value),
        'Base_minus_RF_pp':100*(base['mean']-rf),'Motion_increment_over_Base_pp':100*(full['mean']-base['mean']),
        'RF_f1':rf_f1,'Base_f1':base['f1'],'full_f1':full['f1'],'full_minus_RF_class_pp':[100*(v-r) for v,r in zip(full['f1'],rf_f1)],
        'RF_source':path,'full_source':'reports/sensor_motion_pair_formal_20261009/verified_summary.json',
        'same_author_targets_and_GT':True,'RF_fold_count':5,'network_seeds_per_fold':3,'input_history_same':False}
out=Path('reports/SENSOR_ORIGINAL_PAPER_BASELINE_COMPARISON_20261009.json')
out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
for proto,p in result['protocols'].items():
    print(proto,{k:v for k,v in p.items() if k.endswith('mean') or k.endswith('_pp') and 'class' not in k})
