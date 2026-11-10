import unittest

import numpy as np
import torch
from hindi_detector.dual_paired_detector import DualPairedNetwork

from hindi_detector.dual_paired_vowels import supported
from hindi_detector.task_vowel_detector import decisions, score


class ReleaseVowelSafetyTests(unittest.TestCase):
    def evidence(self, quality=False):
        x = np.zeros((1, 114), np.float32)
        for offset in (0, 57):
            x[0, offset + 31] = 0  # Reference expected-vowel emission is plausible.
            x[0, offset + 7] = -.2  # Reference favors expected path.
            x[0, offset + 9] = .2  # Query/reference shift survives crop perturbations.
            x[0, offset + 13] = 0
            x[0, offset + (17 if quality else 16)] = 1
            x[0, offset + 30] = .1  # Query alternative-vowel support.
            x[0, offset + 44] = -.1  # Local expected nucleus decreased.
            x[0, offset + 2] = .2
        return x

    def test_reduced_reference_blocks_even_a_high_model_score(self):
        x = self.evidence()
        self.assertTrue(supported(x)[0])
        state = {'heads': {kind: {'operating_points': {'high': .5, 'low': .2}}
                          for kind in ('vowel_deletion', 'vowel_quality')}}
        self.assertEqual(decisions(state, x, [.99]), ['strong_vowel_change'])
        for probe in (0, 57):
            bad = x.copy()
            bad[0, probe + 7] = .1
            self.assertEqual(decisions(state, bad, [.99]), ['uncertain'])

    def test_both_probes_must_support_crop_stable_relative_change(self):
        for quality in (False, True):
            x = self.evidence(quality)
            self.assertTrue(supported(x)[0])
            for probe in (0, 57):
                bad = x.copy()
                bad[0, probe + 9] = -.01
                self.assertFalse(supported(bad)[0])

    def test_task_heads_receive_the_intended_probe_and_preserve_row_order(self):
        networks = {kind: DualPairedNetwork('linear', mode) for kind, mode in [('vowel_deletion', 'secondary57'), ('vowel_quality', 'primary57')]}
        for net in networks.values():
            with torch.no_grad():
                net.head.weight.zero_()
                net.head.bias.zero_()
                net.head.weight[0, 0] = 1
        state = {'heads': {kind: {'feature_mode': mode, 'mean': np.zeros(57), 'scale': np.ones(57)}
                           for kind, mode in [('vowel_deletion', 'secondary57'), ('vowel_quality', 'primary57')]}}
        x = np.concatenate([self.evidence(True), self.evidence(False), self.evidence(True)])
        x[:, 0] = [2, -6, -2]
        x[:, 57] = [-6, 3, 6]
        np.testing.assert_allclose(score(networks, state, x), torch.sigmoid(torch.tensor([2., 3., -2.])).numpy(), rtol=1e-6)

    def test_invalid_or_ambiguous_feature_rows_fail_explicitly(self):
        for x in (np.zeros((1, 57)), np.full((1, 114), np.nan), np.zeros((1, 114))):
            with self.assertRaises(ValueError):
                score({}, {}, x)


if __name__ == '__main__':
    unittest.main()
