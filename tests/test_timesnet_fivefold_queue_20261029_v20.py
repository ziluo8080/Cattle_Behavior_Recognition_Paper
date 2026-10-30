import tempfile
import unittest
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from scripts import run_timesnet_fivefold_20261029_v20 as runner
from scripts import report_timesnet_fivefold_20261029_v20 as reporter


class ControllerAdmissionTest(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(Torch_memory_fraction_per_process=.15, GPU_processes=3, minimum_free_GPU_MiB=6144)
        self.handle = Mock()
        self.fcntl = SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=Mock())

    def test_existing_occupancy_allowed_with_full_caps_and_reserve(self):
        with patch.dict(sys.modules, {'fcntl': self.fcntl}), patch.object(Path, 'open', return_value=self.handle), patch('subprocess.check_output', return_value='24576, 21000\n'):
            self.assertIs(runner.c.controller_gpu_lock(self.cfg), self.handle)
        self.handle.close.assert_not_called()
        self.fcntl.flock.assert_called_once_with(self.handle, 6)

    def test_insufficient_free_memory_releases_global_lock(self):
        with patch.dict(sys.modules, {'fcntl': self.fcntl}), patch.object(Path, 'open', return_value=self.handle), patch('subprocess.check_output', return_value='24576, 17203\n'):
            with self.assertRaisesRegex(AssertionError, 'free 17203 MiB < required 17204 MiB'):
                runner.c.controller_gpu_lock(self.cfg)
        self.handle.close.assert_called_once()

    def test_busy_global_lock_rejected_without_gpu_query(self):
        self.fcntl.flock.side_effect = BlockingIOError('global lock occupied')
        with patch.dict(sys.modules, {'fcntl': self.fcntl}), patch.object(Path, 'open', return_value=self.handle), patch('subprocess.check_output') as query:
            with self.assertRaises(BlockingIOError):
                runner.c.controller_gpu_lock(self.cfg)
        query.assert_not_called()
        self.handle.close.assert_called_once()


class QueuePolicyTest(unittest.TestCase):
    def setUp(self):
        self.cfg=dict(models=['TimesNet_DS'],protocols=['TS','OS'],folds=['fold_'+str(i) for i in range(1,6)],seeds=[17,23,29])
        self.jobs=runner.manifest(self.cfg);self.lookup={runner.identity(j):j for j in self.jobs}

    def test_complete_grid(self):
        self.assertEqual(len(self.lookup),60)
        self.assertEqual(sum(j['stage']=='calibration' for j in self.jobs),30)
        self.assertEqual(sum(j['stage']=='refit' for j in self.jobs),30)
        self.assertEqual(len(runner.prediction_manifest(self.cfg,self.cfg['models'])),30)

    def test_refit_waits_for_own_calibration(self):
        j=next(j for j in self.jobs if j['stage']=='refit');dep=self.lookup[runner.identity(dict(j,stage='calibration'))]
        for status in ['pending','running','hold','failed','blocked']:
            dep['status']=status;self.assertFalse(runner.eligible(j,self.lookup))
        dep['status']='completed';self.assertTrue(runner.eligible(j,self.lookup))

    def test_completed_job_never_reeligible(self):
        for j in self.jobs:
            j['status']='completed';self.assertFalse(runner.eligible(j,self.lookup))

    def test_dispatch_error_drains_existing_child(self):
        first=dict(self.jobs[0]);second=next(dict(j) for j in self.jobs if j['stage']=='calibration' and j['protocol']=='OS')
        proc=Mock();proc.poll.side_effect=[None,0];stream=Mock()
        def launch(cfg,job,slot,phase):
            if slot==0:
                job['status']='running';return dict(process=proc,stream=stream,job=job,tick=0,limit=100)
            raise OSError('synthetic dispatch failure')
        with tempfile.TemporaryDirectory() as d:
            cfg=dict(self.cfg,output=d,budget=dict(queue_seconds=1000))
            with patch.object(runner,'dispatch',side_effect=launch),patch.object(runner.time,'monotonic',return_value=1),patch.object(runner.time,'sleep'),patch.object(runner,'receipt',return_value=dict(status='COMPLETE_CALIBRATION')):
                with self.assertRaisesRegex(RuntimeError,'synthetic dispatch failure'):runner.run_jobs(cfg,[first,second],'training',0)
            self.assertEqual(first['status'],'completed');self.assertEqual(second['status'],'failed')
            stream.close.assert_called_once()

    def test_unexpected_queue_write_failure_cleans_own_child(self):
        proc=Mock();stream=Mock();job=dict(self.jobs[0])
        def launch(cfg,j,slot,phase):
            j['status']='running';return dict(process=proc,stream=stream,job=j,tick=0,limit=100)
        with tempfile.TemporaryDirectory() as d,patch.object(runner,'dispatch',side_effect=launch),patch.object(runner.time,'monotonic',return_value=1),patch.object(runner.c,'save',side_effect=OSError('write failed')),patch.object(runner,'kill_own') as kill:
            with self.assertRaisesRegex(OSError,'write failed'):runner.run_jobs(dict(self.cfg,output=d,budget=dict(queue_seconds=1000)),[job],'training',0)
            kill.assert_called_once_with(proc);stream.close.assert_called_once()

    def test_all_calibrations_and_refits_frozen(self):
        for j in self.jobs:j['status']='completed'
        with tempfile.TemporaryDirectory() as d,patch.object(runner,'receipt',return_value=dict(files={})),patch.object(runner.c,'sha',return_value='sha'):
            cfg=dict(self.cfg,output=d)
            self.assertEqual(runner.training_lock(cfg,self.jobs),['TimesNet_DS'])
            lock=runner.c.read(Path(d)/'TRAINING_LOCK.json');self.assertEqual(len(lock['files']),60)
            self.jobs[-1]['status']='blocked'
            self.assertEqual(runner.training_lock(cfg,self.jobs),[])
            self.assertEqual(runner.c.read(Path(d)/'TRAINING_LOCK.json')['held_models'],['TimesNet_DS'])

    def test_receipt_rejects_escaping_path(self):
        job=self.jobs[0]
        with tempfile.TemporaryDirectory() as d:
            cfg=dict(self.cfg,output=d);folder=runner.folder(cfg,job)
            runner.c.save(folder/'COMPLETE.json',dict(status='COMPLETE_CALIBRATION',Test_labels_parsed=False,
                **{k:job[k] for k in ('protocol','fold','model','seed')},files={str(Path(d).resolve()/'outside.txt'):'fake'}))
            with self.assertRaises(AssertionError):runner.receipt(cfg,job)

    def test_scoring_lock_failure_opens_no_test(self):
        with patch.object(reporter.c,'check'),patch.object(reporter.c,'check_training_lock',side_effect=AssertionError('missing all60')),patch.object(reporter.np,'load') as load:
            with self.assertRaisesRegex(AssertionError,'missing all60'):reporter.report(dict(self.cfg,output='missing'))
            load.assert_not_called()

    def test_equal_fold_and_seed_aggregation(self):
        values={};exact={};m='TimesNet_DS'
        for p in self.cfg['protocols']:
            for index,f in enumerate(self.cfg['folds']):
                for si,s in enumerate(self.cfg['seeds']):
                    key=p,f,m,s;score=Fraction(10*index+si)
                    values[key]=dict(class_f1_percent=[float(score)]*7);exact[key]=score
        summary,rows=reporter.aggregate(self.cfg,[m],values,exact)
        self.assertEqual(summary[m]['TS']['macro_F1_percent'],21.)
        self.assertEqual(summary[m]['TS']['per_fold_F1_percent'],[1.,11.,21.,31.,41.])
        self.assertEqual(len(rows),10)


if __name__=='__main__':unittest.main()
