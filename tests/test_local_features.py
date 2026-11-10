from __future__ import annotations

import unittest

import numpy as np

from hindi_detector.local_features import word_prosody


class LocalProsodyInvariance(unittest.TestCase):
    def inputs(self):
        t = np.arange(0, 2, .005)
        f0 = 150 * np.exp2(.15 * np.sin(2 * np.pi * t))
        f0[(t > .85) & (t < 1)] = 0
        rms = .1 + .03 * np.cos(3 * t)
        spans = [{'start': .1, 'end': .8}, {'start': 1., 'end': 1.8}]
        return t, f0, rms, spans

    def test_pitch_register_and_loudness_offsets_do_not_change_word_features(self):
        t, f0, rms, spans = self.inputs()
        original = word_prosody(t, f0, rms, spans)
        changed = word_prosody(t, f0 * np.exp2(5 / 12), rms * .3, spans)
        np.testing.assert_allclose(changed, original, atol=1e-6)

    def test_global_tempo_does_not_fake_relative_stress(self):
        t, f0, rms, spans = self.inputs()
        scaled = [{'start': s['start'] * 1.4, 'end': s['end'] * 1.4} for s in spans]
        np.testing.assert_allclose(word_prosody(t, f0, rms, spans), word_prosody(t * 1.4, f0, rms, scaled), atol=1e-6)

    def test_missing_pitch_is_masked_instead_of_becoming_native_evidence(self):
        t, f0, rms, spans = self.inputs()
        result = word_prosody(t, np.zeros_like(f0), rms, spans)
        np.testing.assert_array_equal(result[:, 4], 0)

    def test_overlapping_word_times_are_rejected(self):
        t, f0, rms, _ = self.inputs()
        with self.assertRaises(ValueError):
            word_prosody(t, f0, rms, [{'start': .1, 'end': 1.1}, {'start': 1., 'end': 1.8}])
