import json
import unittest

import numpy as np

from scripts.train_acoustic_deletion import FAMILIES, assess, source_weights


class AcousticReportingTests(unittest.TestCase):
    def rows(self):
        return [{'pair_id': str(i), 'phone_collision_group': 'word' + str(i // 2),
                 'family': family, 'label': int(family.startswith('P')), 'diagnostic_retained_ms': 50.}
                for i in range(24) for family in FAMILIES]

    def test_numpy_predictions_produce_strict_json_for_all_gates_and_strata(self):
        rows = self.rows()
        logits = np.array([2. if r['label'] else -2. for r in rows])
        report = assess(rows, rows, logits, 0.)
        parsed = json.loads(json.dumps(report, allow_nan=False))
        self.assertTrue(parsed['passed'])
        self.assertTrue(parsed['gates']['short_vowel_40_to_60ms'])

    def test_short_retained_vowel_has_half_of_negative_mass(self):
        rows = self.rows()
        values = source_weights(rows)
        for identity in {r['pair_id'] for r in rows}:
            negative = sum(w for r, w in zip(rows, values) if r['pair_id'] == identity and r['label'] == 0)
            short = sum(w for r, w in zip(rows, values) if r['pair_id'] == identity and r['family'] == 'S2')
            self.assertAlmostEqual(float(short / negative), .5, places=6)
