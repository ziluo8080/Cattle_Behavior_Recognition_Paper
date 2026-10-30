"""Small CPU checks; no real FIT, Test, GPU or cloud execution."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from scripts import comparator_fivefold_rocket_20261029_v14 as worker


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value), encoding='utf-8')


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = dict(output=str(self.root), model_config='unused.json', rocket_model=worker.MODEL,
            alpha_by_protocol=dict(worker.ALPHA), ridge=dict(feature_block=512, max_relative_residual=1e-6),
            budget=dict(rocket_worker_seconds=7200), folds=['fold_1'], protocols=['TS', 'OS'], seeds=[17, 23, 29])
        options = dict(requested_features_per_branch=25000, features_per_kernel=4,
            max_dilations_per_kernel=32, transform_batch_size=256, internal_cross_validation=False)
        def read(path):
            if str(path) == 'unused.json': return dict(multirocket=options)
            return json.loads(Path(path).read_text())
        def case_dir(cfg, stage, p, f, model, seed):
            return self.root / stage / p / f / model / f'seed{seed}'
        rng = np.random.default_rng(17); n = 14
        dynamic = rng.normal(size=(n, 100, 8)).astype(np.float32)
        dynamic[:, :, 6] = 1.; dynamic[0, :, 6] = 0.
        labels = np.arange(n) % 7; labels[-1] = 3
        valid = np.ones((n, 60), bool); valid[0] = False
        counts = np.bincount(labels, minlength=7)
        self.raw = dict(dynamic=dynamic, state=rng.normal(size=(n, 60, 26)).astype(np.float32),
            state_valid=valid, context=rng.normal(size=(n, 136)).astype(np.float32),
            current=rng.normal(size=(n, 26)).astype(np.float32), labels=labels,
            prior=counts / n, class_weights=np.sqrt(n / (7 * counts)))
        self.common = SimpleNamespace(check=Mock(), read=read, save=save,
            sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(), case_dir=case_dir,
            gpu_lock=Mock(side_effect=lambda *a, **kw: SimpleNamespace(close=Mock())),
            check_training_lock=Mock(), raw_refit=Mock(side_effect=lambda *a: self.raw),
            raw_test=Mock(side_effect=lambda *a: {k: v for k, v in self.raw.items() if k != 'labels'}))
        self.fit_calls = []
        def fit(module, x, requested, options):
            self.fit_calls.append(dict(rows=len(x), requested=requested, key=x.shape[1]))
            return np.random.uniform(size=(3,))
        def transform(module, x, parameters, options):
            # Wide enough to exercise the true concatenation/scaler/ridge path.
            # Distinct columns avoid all-zero normalization in the replay check.
            base = x[:, 0, :3] @ parameters
            return (base[:, None] + np.arange(24864)[None, :] % 13).astype(np.float64)
        self.patches = [patch.object(worker, '_common', return_value=self.common),
            patch.object(worker, '_configure_cuda'),
            patch.object(worker, 'rocket_functions', return_value=(None, fit, transform, lambda seed: None))]
        original_fit = worker.fit_ridge
        def cpu_fit(*args, **kwargs):
            kwargs['device'] = 'cpu'
            return original_fit(*args, **kwargs)
        self.patches.append(patch.object(worker, 'fit_ridge', side_effect=cpu_fit))
        for item in self.patches:
            item.start(); self.addCleanup(item.stop)

    def folder(self, stage='refit', seed=17):
        return self.common.case_dir(self.cfg, stage, 'TS', 'fold_1', worker.MODEL, seed)

    def run_refit(self, seed=17):
        return worker.refit(self.cfg, 'TS', 'fold_1', seed, 2)

    def test_refit_fresh_fit_only_and_exact_serialization(self):
        import joblib
        row = self.run_refit()
        self.assertEqual(row['status'], 'COMPLETE_REFIT')
        self.assertEqual([r['rows'] for r in self.fit_calls], [14, 14])
        self.assertEqual([r['key'] for r in self.fit_calls], [8, 26])
        self.assertEqual(row['features'], 49890)
        self.assertEqual(row['alpha'], 20000)
        self.assertTrue(row['serialized_features_exact'] and row['serialized_scores_exact'])
        self.assertFalse(row['Test_inputs_opened'])
        self.common.raw_test.assert_not_called()
        self.common.check_training_lock.assert_not_called()
        self.assertEqual(set(row['files']), {'model.joblib', 'fit_replay_inputs.npz', 'fit_replay_scores.npy'})
        self.assertFalse((self.folder() / 'fit.npy').exists())
        model = joblib.load(self.folder() / 'model.joblib')
        np.testing.assert_array_equal(model['weights'], self.raw['class_weights'].astype(np.float32).astype(np.float64))

    def test_each_seed_really_refits_both_author_branches(self):
        import joblib
        self.run_refit(17); self.run_refit(23)
        self.assertEqual(len(self.fit_calls), 4)
        a = joblib.load(self.folder(seed=17) / 'model.joblib')
        b = joblib.load(self.folder(seed=23) / 'model.joblib')
        self.assertFalse(np.array_equal(a['parameters'][0], b['parameters'][0]))

    def test_predict_unlock_and_raw_replay_and_missing_prior(self):
        self.run_refit(); fit_count = len(self.fit_calls)
        row = worker.predict(self.cfg, 'TS', 'fold_1', 17, 2)
        pred = np.load(self.folder('prediction') / 'test_predictions.npy')
        self.assertEqual(row['status'], 'COMPLETE_PREDICTION')
        self.assertEqual(len(self.fit_calls), fit_count)
        self.assertEqual(pred[0], int(self.raw['prior'].argmax()) + 1)
        self.assertTrue(np.isin(pred, np.arange(1, 8)).all())
        self.assertFalse(row['Test_labels_parsed'])
        self.assertFalse(row['new_blind_evaluation'])
        self.assertTrue(row['serialized_scores_exact'])
        self.common.check_training_lock.assert_called_once_with(self.cfg)

    def test_training_lock_blocks_before_test_input(self):
        self.common.check_training_lock.side_effect = RuntimeError('not frozen')
        with self.assertRaisesRegex(RuntimeError, 'not frozen'):
            worker.predict(self.cfg, 'TS', 'fold_1', 17, 2)
        self.common.raw_test.assert_not_called()
        self.assertFalse(self.folder('prediction').exists())

    def test_test_labels_rejected_without_scoring(self):
        self.run_refit(); self.common.raw_test.side_effect = lambda *args: self.raw
        with self.assertRaises(AssertionError): worker.predict(self.cfg, 'TS', 'fold_1', 17, 2)
        self.assertTrue((self.folder('prediction') / 'FAILED.json').exists())
        self.assertFalse((self.folder('prediction') / 'test_predictions.npy').exists())

    def test_changed_checkpoint_rejected_before_test_open(self):
        self.run_refit()
        with (self.folder() / 'model.joblib').open('ab') as stream: stream.write(b'tamper')
        with self.assertRaisesRegex(AssertionError, 'hash mismatch'):
            worker.predict(self.cfg, 'TS', 'fold_1', 17, 2)
        self.common.raw_test.assert_not_called()

    def test_failure_is_preserved_and_no_scientific_retry(self):
        with patch.object(worker, 'fit_ridge', side_effect=RuntimeError('deliberate failure')):
            with self.assertRaisesRegex(RuntimeError, 'deliberate failure'): self.run_refit()
        self.assertTrue((self.folder() / 'FAILED.json').exists())
        with self.assertRaises(FileExistsError): self.run_refit()
        self.assertFalse((self.folder() / 'COMPLETE.json').exists())

    def test_frozen_alpha_and_private_gpu_slot(self):
        self.cfg['alpha_by_protocol']['TS'] = 1
        with self.assertRaisesRegex(AssertionError, 'Frozen alphas'): self.run_refit()
        self.cfg['alpha_by_protocol']['TS'] = 20000
        with self.assertRaises(AssertionError): worker.refit(self.cfg, 'TS', 'fold_1', 17, 0)
        self.common.raw_refit.assert_not_called()

    def test_original_masking_and_missingness(self):
        result = worker._data(self.raw, True)
        np.testing.assert_array_equal(result['native'][0, :, :6], 0.)
        np.testing.assert_array_equal(result['native'][0, :, 7], .2)
        self.assertTrue(result['missing'][0]); self.assertFalse(result['missing'][1])
        self.assertFalse(np.array_equal(self.raw['dynamic'][0, :, :6], result['native'][0, :, :6]))

    def test_unsealed_witness_runs_without_scientific_seal_check(self):
        self.common.check.side_effect = FileNotFoundError('SEAL not created yet')
        def fit(module, values, requested, options):
            return ((np.random.normal(size=3),), (np.random.normal(size=3),))
        def whole(values, difference, original, differenced, pooling):
            return np.column_stack((values[:, 0, :3] @ original[0],
                                    difference[:, 0, :3] @ differenced[0]))
        module = SimpleNamespace(transform=whole)
        def transform(module, values, parameters, options):
            return whole(values, np.diff(values, axis=2), *parameters, 4)
        with patch.object(worker, 'rocket_functions', return_value=(module, fit, transform, lambda seed: None)), \
             patch('threadpoolctl.threadpool_limits'):
            result = worker.witness(self.cfg, 'cpu', self.root / 'unsealed_witness')
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['real_data_rows_read'], 0)
        self.common.check.assert_not_called()
        self.common.raw_refit.assert_not_called(); self.common.raw_test.assert_not_called()

    def test_unsealed_preflight_does_not_require_training_seal(self):
        self.common.check.side_effect = FileNotFoundError('SEAL not created yet')
        full = dict(n=14, columns=512, gram_seconds=.001, solve_seconds=.001,
                    residual=1e-12, peak_cuda_bytes=1024)
        with patch('torch.cuda.mem_get_info', return_value=(24 * 2**30, 24 * 2**30)), \
             patch('scripts.sensor_comparator_ridge_20261025.full_shape_witness', return_value=full):
            result = worker.preflight(self.cfg)
        self.assertEqual(result['status'], 'PASS')
        self.assertFalse(result['scientific_fit'])
        self.common.check.assert_not_called(); self.common.raw_test.assert_not_called()

    def test_scientific_workers_still_require_seal_before_data_or_started(self):
        self.common.check.side_effect = FileNotFoundError('SEAL missing')
        for action in (worker.refit, worker.predict):
            with self.assertRaisesRegex(FileNotFoundError, 'SEAL missing'):
                action(self.cfg, 'TS', 'fold_1', 17, 2)
        self.common.raw_refit.assert_not_called(); self.common.raw_test.assert_not_called()
        self.assertFalse(self.folder('refit').exists())
        self.assertFalse(self.folder('prediction').exists())

    def test_resource_admission_failure_leaves_no_scientific_receipt_or_data_read(self):
        self.run_refit()
        self.common.raw_refit.reset_mock(); self.common.raw_test.reset_mock()
        self.common.gpu_lock.side_effect = RuntimeError('resource admission denied')
        for action, stage, seed in ((worker.refit, 'refit', 23), (worker.predict, 'prediction', 17)):
            with self.subTest(stage=stage):
                with self.assertRaisesRegex(RuntimeError, 'resource admission denied'):
                    action(self.cfg, 'TS', 'fold_1', seed, 2)
                self.assertFalse(self.folder(stage, seed).exists())
        self.common.raw_refit.assert_not_called(); self.common.raw_test.assert_not_called()

    def test_configuration_failure_releases_lock_before_scientific_start(self):
        self.run_refit()
        self.common.raw_refit.reset_mock(); self.common.raw_test.reset_mock()
        lock = SimpleNamespace(close=Mock())
        self.common.gpu_lock.side_effect = lambda *args, **kwargs: lock
        with patch.object(worker, '_configure_cuda', side_effect=RuntimeError('configuration rejected')):
            for action, stage, seed in ((worker.refit, 'refit', 23), (worker.predict, 'prediction', 17)):
                with self.subTest(stage=stage):
                    with self.assertRaisesRegex(RuntimeError, 'configuration rejected'):
                        action(self.cfg, 'TS', 'fold_1', seed, 2)
                    self.assertFalse(self.folder(stage, seed).exists())
        self.assertEqual(lock.close.call_count, 2)
        self.common.raw_refit.assert_not_called(); self.common.raw_test.assert_not_called()


class RidgeObjectiveTests(unittest.TestCase):
    def test_frozen_alphas_match_independent_numpy_primal_both_shapes(self):
        rng = np.random.default_rng(81)
        for n, d in ((21, 9), (14, 27)):
            x = rng.normal(size=(n, d)); y = np.arange(n) % 7; w = rng.uniform(.2, 3, n)
            target = 2 * (y[:, None] == np.arange(7)).astype(float) - 1
            xm = np.average(x, axis=0, weights=w); ym = np.average(target, axis=0, weights=w)
            xc = (x - xm) * np.sqrt(w)[:, None]; yc = (target - ym) * np.sqrt(w)[:, None]
            for alpha in worker.ALPHA.values():
                model = worker.fit_ridge(x, y, w, alpha, block=5, device='cpu')
                coef = np.linalg.solve(xc.T @ xc + alpha * np.eye(d), xc.T @ yc)
                expected = (x - xm) @ coef + ym
                np.testing.assert_allclose(worker.predict_scores(x, model), expected, atol=1e-10, rtol=1e-10)
                self.assertLess(model['relative_system_residual'], 1e-6)


if __name__ == '__main__':
    unittest.main()
