from __future__ import annotations

import unittest

import numpy as np

from hindi_detector.context_calibrated import decisions


class ContextNoiseSafety(unittest.TestCase):
    def inputs(self, quality=False):
        x = np.zeros((1, 114), np.float32)
        for start in (0, 57):
            x[0, start + 7] = -.1
            x[0, start + 9] = .01
            x[0, start + 13] = -.1
            x[0, start + 31] = -.1
            x[0, start + 30] = 1
            x[0, start + 44] = -.5
            x[0, start + 16] = not quality
            x[0, start + 17] = quality
            x[0, start + 19] = -.1
            x[0, start + 2] = .1
        state = {'heads': {kind: {'operating_points': {'high': .5, 'low': .2}} for kind in ('vowel_deletion', 'vowel_quality')},
                 'null_effect_floor': {'vowel_deletion': .035}}
        return state, x

    def test_high_model_score_cannot_override_measured_no_change_noise(self):
        state, x = self.inputs()
        self.assertEqual(decisions(state, x, [.999]), ['uncertain'])
        x[0, 9] = x[0, 66] = .1
        self.assertEqual(decisions(state, x, [.999]), ['strong_vowel_change'])

    def test_quality_head_is_not_recalibrated_by_deletion_noise(self):
        state, x = self.inputs(quality=True)
        self.assertEqual(decisions(state, x, [.999]), ['strong_vowel_change'])

    def test_negative_noise_bound_is_rejected(self):
        state, x = self.inputs()
        state['null_effect_floor']['vowel_deletion'] = -.1
        with self.assertRaises(ValueError):
            decisions(state, x, [.999])
