from __future__ import annotations

import unittest

import numpy as np

from hindi_detector.prosody_transfer import pair_prosody, transfer_pitch


class PitchInterventionSafety(unittest.TestCase):
    def inputs(self):
        times = np.arange(0, 1.005, .005)
        f0 = np.full(len(times), 150.)
        f0[(times > .4) & (times < .46)] = 0
        tracks = {'times': times.tolist(), 'pitch_relative_st': (4 * np.sin(2 * np.pi * times)).tolist()}
        return times, f0, tracks, {'start': .2, 'end': .8}, {'start': .1, 'end': .9}

    def test_intervention_preserves_unvoiced_frames_and_other_words(self):
        times, f0, tracks, carrier, donor = self.inputs()
        changed, receipt = transfer_pitch(times, f0, tracks, carrier, donor)
        np.testing.assert_array_equal(changed > 0, f0 > 0)
        outside = (times <= carrier['start']) | (times >= carrier['end'])
        np.testing.assert_allclose(changed[outside], f0[outside], atol=1e-12)
        self.assertGreater(receipt['intended_median_absolute_change_st'], .5)

    def test_carrier_pitch_register_does_not_change_relative_intervention(self):
        times, f0, tracks, carrier, donor = self.inputs()
        a, _ = transfer_pitch(times, f0, tracks, carrier, donor)
        b, _ = transfer_pitch(times, f0 * 1.25, tracks, carrier, donor)
        np.testing.assert_allclose(a * 1.25, b, atol=1e-10)

    def test_missing_or_impossible_donor_is_rejected(self):
        times, f0, tracks, carrier, donor = self.inputs()
        for values in ([None] * len(times), [100.] * len(times)):
            with self.assertRaises(ValueError):
                transfer_pitch(times, f0, {**tracks, 'pitch_relative_st': values}, carrier, donor)

    def test_paired_input_does_not_use_absolute_pitch_or_ids(self):
        reference = np.zeros((4, 29), np.float32)
        query = reference.copy()
        query[1, 9] = 3
        result = pair_prosody(query, reference, 1)
        self.assertEqual(result.shape, (300,))
        np.testing.assert_array_equal(result[150:], np.abs(result[:150]))
        with self.assertRaises(ValueError):
            pair_prosody(query, reference[:2], 1)
