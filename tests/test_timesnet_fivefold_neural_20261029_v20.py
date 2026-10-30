"""Policy regressions and stage boundaries; no real data or CUDA required."""
import copy
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import timesnet_fivefold_neural_20261029_v20 as neural


def configuration():
    path = Path(__file__).resolve().parents[1] / 'configs/sensor_comparator_repair_20261028_v7.json'
    cfg = json.loads(path.read_text(encoding='utf-8-sig'))
    pilot = json.loads((path.parent / 'sensor_timesnet_pilot_20261029_v17.json').read_text(encoding='utf-8-sig'))
    cfg.update(timesnet=pilot['timesnet'], eval_batch_size=256)
    cfg.update(calibration_stopping=dict(minimum_epochs=40, maximum_epochs=160,
               patience_epochs=20, relative_improvement=.001), models=list(neural.MODELS), folds=[1, 2, 3, 4, 5])
    return cfg


def history(count=40):
    return [dict(epoch=e, stage='source', stage_epoch=e, learning_rate=3e-6,
                 next_learning_rate=3e-6, train_ce=.3, fit_ema=dict(weighted_CE=.3),
                 dev_ema=dict(weighted_CE=.4, macro_f1_percent=80.),
                 raw_relative_parameter_change=.001, EMA_relative_parameter_change=.001)
            for e in range(1, count + 1)]


class CalibrationPolicy(unittest.TestCase):
    def test_requires_training_stability_even_with_VAL_plateau(self):
        rows = history()
        rows[-1]['raw_relative_parameter_change'] = .003
        decision = neural.calibration_stop(rows, configuration())
        self.assertTrue(decision['validation_plateau'])
        self.assertFalse(decision['accepted'])
        self.assertFalse(decision['stop'])

    def test_requires_VAL_plateau_even_with_training_stability(self):
        rows = history()
        rows[-1]['dev_ema']['weighted_CE'] = .3
        decision = neural.calibration_stop(rows, configuration())
        self.assertTrue(decision['training_components']['passed'])
        self.assertFalse(decision['validation_plateau'])
        self.assertFalse(decision['stop'])

    def test_old_mixed_gate_is_recorded_without_becoming_training_gate(self):
        rows = history(60)
        rows[0]['dev_ema']['weighted_CE'] = .2
        decision = neural.calibration_stop(rows, configuration())
        self.assertFalse(decision['legacy_mixed_gate']['passed'])
        self.assertTrue(decision['training_components']['passed'])
        self.assertTrue(decision['accepted'])
        self.assertTrue(decision['stop'])

    def test_maximum_returns_hold_without_weakening_threshold(self):
        rows = history(160)
        rows[-1]['EMA_relative_parameter_change'] = .002001
        decision = neural.calibration_stop(rows, configuration())
        self.assertTrue(decision['stop'])
        self.assertFalse(decision['accepted'])
        self.assertEqual(decision['reason'], 'BOUNDED_BUDGET_HOLD')

    def test_minimum_40_epochs_and_exact_parameter_boundary(self):
        rows = history()
        for row in rows:
            row['raw_relative_parameter_change'] = .002
        self.assertFalse(neural.calibration_stop(rows[:39], configuration())['accepted'])
        self.assertTrue(neural.calibration_stop(rows, configuration())['accepted'])

    def test_actual_minimum_can_follow_material_improvement_anchor(self):
        rows = history()
        rows[29]['dev_ema']['weighted_CE'] = .3999
        plan = neural.refit_plan(rows, configuration())
        self.assertEqual(plan['selected_epoch'], 30)
        self.assertEqual(neural.calibration_stop(rows, configuration())['last_material_improvement_epoch'], 1)

    def test_lr_prefix_uses_rate_that_trained_selected_epoch_and_ties_earlier(self):
        rows = history(4)
        for row, rate, next_rate, ce in zip(rows, [.0003, .0003, .00006, .000012],
                                          [.0003, .00006, .000012, .000003], [.5, .4, .3, .3]):
            row.update(learning_rate=rate, next_learning_rate=next_rate)
            row['dev_ema']['weighted_CE'] = ce
        original = copy.deepcopy(rows)
        plan = neural.refit_plan(rows, configuration())
        self.assertEqual(plan['selected_epoch'], 3)
        self.assertEqual(plan['base_learning_rates'], [.0003, .0003, .00006])
        self.assertFalse(plan['refit_terminal_stationarity_claimed'])
        self.assertEqual(rows, original)


class WorkerBoundaries(unittest.TestCase):
    def test_exclusive_started_record_survives_duplicate_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            common = SimpleNamespace(case_dir=lambda *args: Path(directory))
            with patch.object(neural, 'common', common, create=True):
                cfg = configuration()
                neural._start(cfg, 'calibration', 'TS', 1, neural.MODEL, 17)
                original = (Path(directory) / 'STARTED.json').read_bytes()
                with self.assertRaises(FileExistsError):
                    neural._start(cfg, 'calibration', 'TS', 1, neural.MODEL, 17)
                self.assertEqual((Path(directory) / 'STARTED.json').read_bytes(), original)

    def test_hold_cannot_load_refit_data_or_acquire_gpu(self):
        common = SimpleNamespace(check=Mock(), case_dir=Mock(return_value=Path('unused')),
                                 read=Mock(return_value=dict(status='COMPLETE_HOLD')),
                                 load_refit=Mock(), gpu_lock=Mock())
        with patch.object(neural, 'runtime'), patch.object(neural, 'common', common, create=True):
            with self.assertRaises(AssertionError):
                neural.refit(configuration(), 'TS', 1, neural.MODEL, 17, 0)
        common.load_refit.assert_not_called()
        common.gpu_lock.assert_not_called()

    def test_test_inputs_cannot_be_loaded_before_training_lock(self):
        common = SimpleNamespace(check=Mock(), check_training_lock=Mock(side_effect=RuntimeError('unsealed')),
                                 load_test=Mock(), gpu_lock=Mock())
        with patch.object(neural, 'runtime'), patch.object(neural, 'common', common, create=True):
            with self.assertRaisesRegex(RuntimeError, 'unsealed'):
                neural.predict(configuration(), 'TS', 1, neural.MODEL, 17, 0)
        common.load_test.assert_not_called()
        common.gpu_lock.assert_not_called()


class ActualTimesNetCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Only synthetic CPU paths run on Windows; no Linux lock is called.
        if os.name == 'nt' and 'fcntl' not in sys.modules:
            sys.modules['fcntl'] = types.SimpleNamespace()
        neural.runtime()
        neural.torch.set_num_threads(4)
        neural.torch.use_deterministic_algorithms(True)
        neural.torch.backends.mha.set_fastpath_enabled(False)
        neural.torch.backends.cuda.enable_flash_sdp(False)
        neural.torch.backends.cuda.enable_mem_efficient_sdp(False)

    def test_actual_TimesNet_full_state_next_update_and_frozen_replay(self):
        # This runs the 403047-parameter official TimesNet, not a tiny proxy.
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(neural.parent, 'new', side_effect=AssertionError('legacy factory called')), \
             patch.object(neural.parent, 'model', side_effect=AssertionError('legacy factory called')), \
             patch.object(neural.parent, 'restore', side_effect=AssertionError('legacy factory called')):
            records = neural.witness(configuration(), 'cpu', Path(tmp) / 'witness')
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['model'], neural.MODEL)
        self.assertEqual(records[0]['parameters'], 403047)
        self.assertTrue(records[0]['complete_next_update_exact'])
        self.assertTrue(records[0]['causal_three_view_and_snapshot_reload_exact'])
        self.assertTrue(records[0]['actual_LR_prefix_verified'])
        self.assertTrue(records[0]['requires_grad_false_fresh_replay_exact'])
        self.assertTrue(records[0]['live_EMA_and_RNG_preserved'])
        self.assertEqual(records[0]['real_data_rows_read'], 0)

    def test_official_FFT_batch_receipt_records_actual_tail_and_order(self):
        np = neural.np
        data = dict(train=np.arange(513), evaluation=np.arange(513, 1030))
        contract = neural.batch_contract(configuration(), data, 23)
        self.assertEqual(contract['TRAIN_or_FIT']['tail_size'], 1)
        self.assertEqual(contract['evaluation']['tail_size'], 5)
        self.assertTrue(contract['FFT_periods_depend_on_batch_membership'])
        self.assertFalse(contract['drop_last'])
        changed = neural.batch_contract(configuration(), dict(data, evaluation=data['evaluation'][::-1]), 23)
        self.assertNotEqual(contract['evaluation']['ordered_ids_sha256'],
                            changed['evaluation']['ordered_ids_sha256'])
        wrong = dict(configuration(), eval_batch_size=128)
        with self.assertRaises(AssertionError):
            neural.batch_contract(wrong, data, 23)
        empty = neural.batch_contract(configuration(), dict(data, evaluation=np.array([], dtype=int)), 23)
        self.assertEqual(empty['evaluation']['tail_size'], 0)

    def test_actual_TimesNet_inference_keeps_batch256_tail_and_loader_order(self):
        np, torch = neural.np, neural.torch
        neural.core.seed_all(23)
        batch = neural.parent.old.synthetic_batch('cpu', 257)
        batch['context'][:, 0] = torch.arange(257).float()
        data = dict(tensors=dict(**batch, labels=torch.arange(257) % 7))
        evaluation = neural.model_for(configuration(), neural.MODEL, np.ones(7) / 7, 'cpu')
        evaluation.eval().requires_grad_(False)
        ids = np.arange(257)[::-1].copy()
        calls = []
        handle = evaluation.register_forward_pre_hook(
            lambda model, args: calls.append(args[0]['context'][:, 0].long().tolist()))
        try:
            probabilities = neural.base.infer(evaluation, data, ids)
        finally:
            handle.remove()
        self.assertEqual([len(values) for values in calls], [256, 1])
        self.assertEqual(sum(calls, []), ids.tolist())
        self.assertEqual(probabilities.shape, (257, 7))
        fresh = neural.model_for(configuration(), neural.MODEL, np.ones(7) / 7, 'cpu')
        fresh.load_state_dict(evaluation.state_dict())
        neural.prepare_evaluation_replay(fresh, evaluation.state_dict())
        np.testing.assert_array_equal(probabilities, neural.base.infer(fresh, data, ids))

    def test_model_factory_rejects_regularized_backbone(self):
        cfg = configuration()
        cfg['timesnet']['dropout'] = .2
        with self.assertRaises(AssertionError):
            neural.model_for(cfg, neural.MODEL, neural.np.ones(7) / 7, 'cpu')


if __name__ == '__main__':
    unittest.main()
