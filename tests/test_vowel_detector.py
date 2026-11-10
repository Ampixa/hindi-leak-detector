import unittest

import numpy as np

from hindi_detector.vowel_detector import direction_supported
from scripts.build_vowel_controls import splice, split_for


class VowelControlTests(unittest.TestCase):
    def test_controls_only_change_target_region_and_join_fades(self):
        samples = np.ones(1000, np.float32)
        result = splice(samples, 200, 600, np.zeros(100, np.float32))
        self.assertEqual(len(result), 700)
        np.testing.assert_array_equal(result[:168], samples[:168])
        np.testing.assert_array_equal(result[-368:], samples[-368:])

    def test_user_regression_voice_is_held_out(self):
        self.assertEqual(split_for('yt01', 'कामदारहरुलाई'), 'test')
        self.assertEqual(split_for('yt10', 'काम'), 'test')

    def test_score_cannot_override_a_reference_already_missing_the_vowel(self):
        x = np.zeros((1, 18), np.float32)
        x[0, [1, 8, 14]] = [.3, .5, 1]
        x[0, 5] = .2
        self.assertFalse(direction_supported(x)[0])
        x[0, 5] = -.2; x[0, 7] = -.2
        self.assertTrue(direction_supported(x)[0])

    def test_boundary_sensitive_margin_does_not_become_a_strong_flag(self):
        x = np.zeros((1, 18), np.float32)
        x[0, [1, 8, 14, 5]] = [.3, .5, 2/3, -.2]
        self.assertFalse(direction_supported(x)[0])
