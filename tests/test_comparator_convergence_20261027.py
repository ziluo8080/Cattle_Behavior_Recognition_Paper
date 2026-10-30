import copy
import json
import unittest
from pathlib import Path
from scripts.comparator_convergence_criteria_20261027 import assess

CFG = json.loads(Path('configs/sensor_comparator_convergence_20261027_v1.json').read_text())['convergence']


def stable(n=40):
    return [dict(epoch=i+1, stage='source', learning_rate=3e-6, train_ce=0.2,
                 fit_ema={'weighted_CE': 0.2},
                 dev_ema={'weighted_CE': 0.3, 'macro_f1_percent': 83.0},
                 raw_relative_parameter_change=0.0001, EMA_relative_parameter_change=0.0001) for i in range(n)]


class ConvergenceTests(unittest.TestCase):
    def test_stable_passes(self): self.assertTrue(assess(stable(), CFG)['passed'])

    def test_still_learning_fails(self):
        h=stable()
        for i,r in enumerate(h): r['train_ce']=0.4-0.004*i
        self.assertFalse(assess(h, CFG)['passed'])

    def test_stable_but_overfit_fails(self):
        h=stable(80); h[0]['dev_ema']['weighted_CE']=0.1
        self.assertFalse(assess(h, CFG)['passed'])

    def test_zero_lr_is_not_convergence(self):
        h=stable()
        for r in h: r['learning_rate']=0.0
        self.assertFalse(assess(h, CFG)['passed'])

    def test_recovery_keeps_source_min(self):
        h=stable(80)
        for r in h[40:]: r['stage']='recovery'
        h[0]['dev_ema']['weighted_CE']=0.1
        self.assertFalse(assess(h, CFG)['passed'])

    def test_recovery_cannot_use_source_window(self):
        h=stable(50)
        for r in h[35:]: r['stage']='recovery'
        self.assertFalse(assess(h, CFG)['passed'])

    def test_F1_oscillation_fails(self):
        h=stable()
        for i,r in enumerate(h): r['dev_ema']['macro_f1_percent']+=2*(i%2)
        self.assertFalse(assess(h, CFG)['passed'])


if __name__ == '__main__': unittest.main()
