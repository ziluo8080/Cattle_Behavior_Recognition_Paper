"""Synthetic-only scientific invariants, no real Test data or local model fit."""
from fractions import Fraction
import json
from pathlib import Path
import unittest

import numpy as np

from scripts.prepare_shared_fusion_fivefold_20261024 import native_windows,normalize_context,validate_targets,fit_roles_and_all_roles
from scripts.sensor_native_immu_data_20261021 import fit_scaler,windows as old_windows
from scripts.report_shared_fusion_fivefold_20261024 import evaluate,exact_decimal
from scripts.freeze_shared_fusion_fivefold_20261024 import jobs
from scripts.sensor_xgb_b_fusion_20261022 import temper,fuse


class FormalChecks(unittest.TestCase):
    def test_native_role_window_and_future_boundaries(self):
        ticks=np.arange(900,1301);values=np.repeat(ticks[:,None].astype(float),6,axis=1)
        sources={1:(ticks,values)};scaler=dict(mean=[0.]*6,std=[1.]*6)
        rows=[dict(cow_id=1,timestamp=105,start=100,end=110,split='train'),dict(cow_id=1,timestamp=115,start=110,end=120,split='test')]
        result,src=native_windows(rows,sources,scaler)
        for i,row in enumerate(rows):
            valid=src[i]>=0;grid=row['timestamp']*10+np.arange(-99,1)
            self.assertTrue((src[i,valid]<=grid[valid]).all());self.assertTrue((src[i,valid]>=row['start']*10).all())
            self.assertTrue((src[i,valid]>row['timestamp']*10-100).all())
        changed=values.copy();changed[ticks>1150]=1e8
        replay,_=native_windows(rows,{1:(ticks,changed)},scaler);np.testing.assert_array_equal(result,replay)

    def test_native_numerically_identical_to_old_development_rule(self):
        rng=np.random.default_rng(17);ticks=np.arange(1000,1400);values=rng.normal(size=(400,6));values[::7]=np.nan
        rows=[dict(cow_id=1,timestamp=115,start=100,end=120,split='train'),dict(cow_id=1,timestamp=135,start=120,end=140,split='val')]
        sources={1:(ticks,values)};normal=dict(mean=[.1]*6,std=[2.]*6)
        old,old_prov=old_windows(rows,sources,normal)
        new,prov=native_windows([dict(r,split='test' if r['split']=='val' else 'train') for r in rows],sources,normal)
        np.testing.assert_array_equal(old,new);np.testing.assert_array_equal(old_prov,prov)

    def test_native_fit_scaler_never_uses_test_cow_or_time(self):
        ticks=np.arange(1000,1200);values=np.repeat(np.arange(200)[:,None],6,axis=1).astype(float)
        fit={1:[dict(start=100,end=110,split='train')]}
        before=fit_scaler({1:(ticks,values),2:(ticks,values)},fit)
        mutated=values.copy();mutated[ticks>=1100]=1e12
        after=fit_scaler({1:(ticks,mutated),2:(ticks,values*1e9)},fit)
        self.assertEqual(before,after)

    def test_context_fit_statistics_ignore_test_values(self):
        x=np.array([[1,2,4],[3,2,4],[20,100,100]],np.float32)
        a,stats=normalize_context(x,[0,1]);x[2]=[-100,9999,-9999]
        b,other=normalize_context(x,[0,1]);self.assertEqual(stats,other)
        np.testing.assert_array_equal(a[:2],b[:2]);self.assertEqual(stats['std'],[1.,1.,1.])

    def test_official_os_all_five_test_pairs_partition_cows(self):
        held=[]
        for fold in range(1,6):
            fit,roles=fit_roles_and_all_roles({},'OS',f'fold_{fold}')
            test=[c for c,r in roles.items() if r[0]['split']=='test']
            self.assertEqual(len(fit),8);self.assertEqual(len(test),2);self.assertFalse(set(fit)&set(test));held+=test
        self.assertEqual(sorted(held),list(range(1,11)))

    def test_target_roles_reject_overlap_and_wrong_identity(self):
        roles={1:[dict(start=100,end=110,split='train'),dict(start=110,end=120,split='test')]}
        rows=[dict(cow_id=1,timestamp=105,start=100,end=110,split='train',label=1),dict(cow_id=1,timestamp=115,start=110,end=120,split='test',label=2)]
        validate_targets(rows,roles)
        with self.assertRaises(AssertionError):validate_targets(rows+[rows[0]],roles)
        with self.assertRaises(AssertionError):validate_targets([rows[0],dict(rows[1],start=100)],roles)

    def test_exact_macro_aggregation_not_pooled_counts(self):
        y=np.arange(1,8);metric,perfect=evaluate(y,np.eye(7));self.assertEqual(perfect,Fraction(100))
        one=np.zeros((7,7));one[:,0]=1
        _,bad=evaluate(y,one);self.assertEqual(bad,Fraction(25,7))
        mean=(perfect+bad)/2;self.assertEqual(exact_decimal(Fraction(1,4)),'0.25')
        self.assertGreater(mean,bad);self.assertEqual(metric['targets'],7)

    def test_frozen_fusion_tie_and_temperature(self):
        px=np.array([[.5,.3,.04,.04,.04,.04,.04]]);pb=np.array([[.1,.6,.06,.06,.06,.06,.06]])
        x=temper(px,2.);b=temper(pb,1.);out=fuse(x,b,'margin_select',.7)
        sx,sb=np.sort(x,axis=1),np.sort(b,axis=1)
        expected=x if .7*(sx[0,-1]-sx[0,-2])>=.3*(sb[0,-1]-sb[0,-2]) else b
        np.testing.assert_array_equal(out,expected)

    def test_manifest_is_exact_fivefold_30_neural_10_tree(self):
        cfg=json.loads(Path('configs/sensor_shared_fusion_fivefold_20261024_v1.json').read_text())
        manifest=jobs(cfg);self.assertEqual(len(manifest),40)
        self.assertEqual(sum(j['resource']=='gpu' for j in manifest),30)
        self.assertEqual(sum(j['resource']=='cpu' for j in manifest),10)


if __name__=='__main__':unittest.main()
