import unittest

import numpy as np

from hindi_detector.boundary_audit import WINDOWS, audit_boundaries
from hindi_detector.g2p_evidence import word_path_evidence


class BoundarySensitivityTests(unittest.TestCase):
    hypothesis = {'status': 'supported_competing_g2p_paths', 'nepali_ids': [1, 2], 'hindi_ids': [1],
                  'probe_special_ids': [0]}

    def probabilities(self, extra_phone):
        values = np.tile([.998, .001, .001], (100, 1))
        values[31] = [.001, .998, .001]
        if extra_phone:
            values[42] = [.001, .001, .998]
        return np.log(values)

    def test_neighboring_phone_can_reverse_path_preference(self):
        result = audit_boundaries(self.probabilities(True), self.hypothesis, start=.3, end=.4, seconds=1.)
        self.assertTrue(result['direction_changes'])
        self.assertEqual(result['status'], 'boundary_sensitive_g2p_evidence')
        self.assertEqual(len(result['windows']), len(WINDOWS))
        self.assertIsNone(result['hindi_error_probability'])

    def test_agreement_does_not_become_error_probability(self):
        result = audit_boundaries(self.probabilities(False), self.hypothesis, start=.3, end=.4, seconds=1.)
        self.assertFalse(result['direction_changes'])
        self.assertTrue(result['all_windows_supported'])
        self.assertEqual(result['status'], 'consistent_sign_in_tested_windows')
        self.assertIsNone(result['hindi_error_probability'])

    def test_baseline_matches_existing_floor_ceil_crop(self):
        values = self.probabilities(True)
        result = audit_boundaries(values, self.hypothesis, start=.3, end=.4, seconds=1.)
        window = result['windows'][0]
        original = word_path_evidence(values[window['first_frame']:window['end_frame_exclusive']], self.hypothesis)
        self.assertEqual(window['hindi_minus_nepali_log_likelihood'], original['hindi_minus_nepali_log_likelihood'])

    def test_unsupported_phone_remains_unavailable(self):
        result = audit_boundaries(self.probabilities(False), {'status': 'unsupported_g2p_projection', 'reason': 'dental phone absent'}, start=.3, end=.4, seconds=1.)
        self.assertEqual(result['windows'], [])
        self.assertIsNone(result['direction_changes'])

    def test_invalid_word_times_cannot_be_repaired_silently(self):
        with self.assertRaises(ValueError):
            audit_boundaries(self.probabilities(False), self.hypothesis, start=.4, end=.3, seconds=1.)


if __name__ == '__main__':
    unittest.main()
