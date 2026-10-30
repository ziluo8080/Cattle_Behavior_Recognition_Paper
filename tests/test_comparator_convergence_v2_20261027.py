import json
import unittest
from pathlib import Path
from tests.test_comparator_convergence_20261027 import stable
from scripts.comparator_convergence_criteria_20261027_v2 import assess
CFG=json.loads(Path('configs/sensor_comparator_convergence_20261027_v2.json').read_text())['convergence']


class StricterTests(unittest.TestCase):
    def test_stable_passes(self):self.assertTrue(assess(stable(),CFG)['passed'])

    def test_F1_step_between_windows_fails(self):
        h=stable()
        for r in h[-10:]:r['dev_ema']['macro_f1_percent']+=10
        self.assertFalse(assess(h,CFG)['passed'])

    def test_CE_step_between_windows_fails(self):
        h=stable()
        for r in h[-10:]:r['dev_ema']['weighted_CE']*=1.04
        self.assertFalse(assess(h,CFG)['passed'])

    def test_bad_terminal_hidden_by_mean_fails(self):
        h=stable()
        for i,r in enumerate(h):r['dev_ema']['weighted_CE']=1.25 if i%5==4 else 1.0
        self.assertFalse(assess(h,CFG)['passed'])


if __name__=='__main__':unittest.main()
