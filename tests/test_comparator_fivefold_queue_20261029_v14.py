import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch,Mock
from scripts.run_comparator_fivefold_retrain_20261029_v14 import manifest,identity,eligible
from scripts import run_comparator_fivefold_retrain_20261029_v14 as runner

class QueuePolicyTest(unittest.TestCase):
    def setUp(self):
        self.cfg=dict(models=['GRU_Fusion','Transformer_Fusion','ModernTCN_Fusion'],rocket_model='MultiRocket_Fusion',
            protocols=['TS','OS'],folds=['fold_'+str(i) for i in range(1,6)],seeds=[17,23,29])
        self.jobs=manifest(self.cfg);self.lookup={identity(j):j for j in self.jobs}
    def test_complete_grid_no_tsla_or_historical(self):
        self.assertEqual(len(self.jobs),210);self.assertEqual(len(self.lookup),210)
        self.assertEqual(sum(j['stage']=='calibration' for j in self.jobs),90)
        self.assertEqual(sum(j['model']=='MultiRocket_Fusion' for j in self.jobs),30)
        self.assertFalse(any('TSLANet' in j['model'] or 'Historical' in j['model'] for j in self.jobs))
    def test_refit_waits_for_own_calibration(self):
        j=next(j for j in self.jobs if j['stage']=='refit' and j['model']=='GRU_Fusion')
        dependency=self.lookup[identity(dict(j,stage='calibration'))]
        for status in ['pending','running','hold','failed','blocked']:
            dependency['status']=status;self.assertFalse(eligible(j,self.lookup))
        dependency['status']='completed';self.assertTrue(eligible(j,self.lookup))
    def test_rocket_does_not_depend_on_neural(self):
        for j in self.jobs:
            if j['model']=='MultiRocket_Fusion':self.assertTrue(eligible(j,self.lookup))
    def test_completed_job_never_reeligible(self):
        for j in self.jobs:
            j['status']='completed';self.assertFalse(eligible(j,self.lookup))
    def test_dispatch_error_drains_existing_child_before_raising(self):
        first=dict(self.jobs[0]);second=next(dict(j) for j in self.jobs if j['stage']=='calibration' and j['model']=='Transformer_Fusion')
        proc=Mock();proc.poll.side_effect=[None,0];stream=Mock()
        def launch(cfg,job,slot,phase):
            if slot==0:
                job['status']='running'
                return dict(process=proc,stream=stream,job=job,tick=0,limit=100)
            raise OSError('synthetic admission/query/log failure')
        with tempfile.TemporaryDirectory() as d:
            cfg=dict(self.cfg,output=d,budget=dict(queue_seconds=1000))
            with patch.object(runner,'dispatch',side_effect=launch),patch.object(runner.time,'monotonic',return_value=1),patch.object(runner.time,'sleep'),patch.object(runner,'receipt',return_value=dict(status='COMPLETE_CALIBRATION')):
                with self.assertRaisesRegex(RuntimeError,'synthetic admission'):
                    runner.run_jobs(cfg,[first,second],'training',0)
            self.assertEqual(first['status'],'completed');self.assertEqual(second['status'],'failed')
            self.assertEqual(proc.poll.call_count,2);stream.close.assert_called_once()

if __name__=='__main__':unittest.main()
