from __future__ import annotations

import unittest

import numpy as np

from hindi_detector.anchored_vowels import FEATURE_NAMES, measure


class AnchoredVowelEvidence(unittest.TestCase):
    def evidence(self, tokens):
        p = np.full((len(tokens), 9), .0001)
        for i, token in enumerate(tokens):
            p[i, token] = 1 - .0001 * 8
        values = np.log(p)
        compiled = {'any_vowel': frozenset({7, 8})}
        variant = {'phones': ['k', 'ax', 'p', 'i', 't'],
                   'units': [{4}, {7}, {5}, {8}, {6}]}
        contrast = {'kind': 'vowel_deletion', 'index': 1}
        return measure(compiled, variant, contrast, [(pad, 0, values) for pad in (0., .03, .06)])

    def test_missing_vowel_at_equal_duration_changes_local_evidence(self):
        intact, detail = self.evidence([0, 4, 0, 7, 7, 0, 5, 0, 8, 0, 6, 0])
        missing, _ = self.evidence([0, 4, 0, 0, 0, 0, 5, 0, 8, 0, 6, 0])
        self.assertLess(intact[0], 0)
        self.assertGreater(missing[0], 0)
        self.assertGreater(intact[3], missing[3])
        self.assertGreater(intact[6], .9)
        self.assertTrue(detail['no_reference_audio_used'])

    def test_neighbor_vowel_does_not_fill_the_focal_gap(self):
        a, _ = self.evidence([0, 4, 0, 0, 0, 0, 5, 0, 8, 0, 6, 0])
        b, _ = self.evidence([0, 4, 0, 0, 0, 0, 5, 8, 8, 8, 6, 0])
        np.testing.assert_allclose(a[:5], b[:5], atol=1e-6)

    def test_schema_has_no_word_duration_or_lexical_identity(self):
        self.assertEqual(len(FEATURE_NAMES), 12)
        self.assertFalse(any(any(part in name for part in ('word_frames', 'unit_count', 'site_position', 'speaker', 'reference')) for name in FEATURE_NAMES))

    def test_quality_is_rejected_explicitly(self):
        with self.assertRaises(ValueError):
            measure({}, {}, {'kind': 'vowel_quality'}, [])
