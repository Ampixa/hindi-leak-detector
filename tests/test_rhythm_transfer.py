from __future__ import annotations

import unittest

import numpy as np

from hindi_detector.rhythm_transfer import duration_map, retime_parameters


class DurationInterventionSafety(unittest.TestCase):
    def words(self):
        return [{'word': str(i), 'start': .1 + i * .3, 'end': .3 + i * .3} for i in range(3)]

    def test_duration_and_pauses_preserved_while_word_lengths_change(self):
        carrier = self.words()
        donor = self.words()
        donor[0]['end'] += .07
        source, target, spans, ratios = duration_map(carrier, donor, 1.)
        self.assertEqual(target[-1], 1.)
        self.assertAlmostEqual(spans[-1, 1], carrier[-1]['end'])
        np.testing.assert_allclose(spans[1:, 0] - spans[:-1, 1], [.1, .1], atol=1e-9)
        self.assertGreater(ratios[0], 1.)
        self.assertTrue(np.all(np.diff(source) > 0))

    def test_identity_map_preserves_world_parameters(self):
        source, target, _, _ = duration_map(self.words(), self.words(), 1.)
        t = np.arange(0, 1.005, .005)
        f0 = np.where(t > .4, 160., 0.)
        spectrum = np.stack([t, t ** 2], axis=1)
        pitch, sp, ap = retime_parameters(t, f0, spectrum, spectrum * .5, source, target)
        np.testing.assert_array_equal(pitch, f0)
        np.testing.assert_allclose(sp, spectrum, atol=1e-12)
        np.testing.assert_allclose(ap, spectrum * .5, atol=1e-12)

    def test_unbounded_stretch_and_word_mismatch_rejected(self):
        donor = self.words()
        donor[0]['end'] = donor[0]['start'] + .001
        with self.assertRaises(ValueError):
            duration_map(self.words(), donor, 1.)

    def test_last_word_can_end_at_the_last_sample(self):
        words = self.words()
        source, target, _, _ = duration_map(words, words, words[-1]['end'])
        self.assertEqual(source[-1], words[-1]['end'])
        self.assertTrue(np.all(np.diff(target) > 0))

    def test_short_function_word_requires_explicit_revised_protocol(self):
        words = self.words()
        words[0]['end'] = words[0]['start'] + .05
        with self.assertRaises(ValueError):
            duration_map(words, words, 1.)
        duration_map(words, words, 1., minimum_word_seconds=.03)
        donor = self.words()
        donor[1]['word'] = 'different'
        with self.assertRaises(ValueError):
            duration_map(self.words(), donor, 1.)
