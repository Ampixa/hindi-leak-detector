import unittest

import numpy as np

from hindi_detector.pronunciation_sites import align_units, paired_site_features


class PronunciationSiteTests(unittest.TestCase):
    def test_internal_vowel_deletion_is_not_a_consonant_substitution(self):
        native = [{4}, {5}, {6}, {7}, {4}, {5}]
        hindi = [{4}, {5}, {6}, {4}, {5}]
        self.assertIn((3, None), align_units(native, hindi))

    def test_quality_and_final_reduction_are_separate_edits(self):
        aligned = align_units([{4}, {5}, {6}, {7}], [{4}, {8}, {6}])
        self.assertEqual(aligned, [(0, 0), (1, 1), (2, 2), (3, None)])

    def test_score_features_use_neither_word_identity_nor_voice(self):
        def evidence(margin):
            return {'windows': [{'alternative_minus_target_per_frame': margin,
                'target_score_per_frame': -1., 'alternative_score_per_frame': -1.+margin} for _ in range(3)]}
        v = paired_site_features(evidence(.3), evidence(-.2), {'kind': 'vowel_deletion'})
        self.assertEqual(v.shape, (18,))
        self.assertAlmostEqual(v[8], .5)
        self.assertEqual(v[-2], 1)
        self.assertTrue(np.isfinite(v).all())
