from __future__ import annotations

import unittest

import numpy as np

from hindi_detector.g2p_evidence import compile_word_hypotheses, word_path_evidence


def row(native, hindi):
    return {'word': 'कर्म', 'nepali_g2p': {'ipa_candidate_units': native}, 'hindi_g2p': {'units_with_stress_preserved': hindi}}


class CompetingG2PIntegrity(unittest.TestCase):
    def test_stress_annotations_are_recorded_and_do_not_change_segments(self):
        result = compile_word_hypotheses([row(['ʌ'], ['ˈʌ'])], {'_': 0, 'ʌ': 4}, {0})['कर्म']
        self.assertEqual(result['status'], 'no_resolved_segmental_contrast')
        self.assertEqual(result['nepali_ids'], result['hindi_ids'])
        self.assertTrue(result['explicit_annotation_projections'])

    def test_dental_detail_is_not_silently_erased(self):
        result = compile_word_hypotheses([row(['t̪', 'ʌ'], ['t'])], {'_': 0, 't': 4, 'ʌ': 5}, {0})['कर्म']
        self.assertEqual(result['status'], 'unsupported_g2p_projection')
        scored = word_path_evidence(np.log(np.array([[.2, .8]])), result)
        self.assertIsNone(scored['hindi_minus_nepali_log_likelihood'])

    def test_equivalent_tie_bar_notation_is_explicit(self):
        result = compile_word_hypotheses([row(['t͡s', 'ʌ'], ['ts'])], {'_': 0, 'ts': 4, 'ʌ': 5}, {0})['कर्म']
        self.assertEqual(result['status'], 'supported_competing_g2p_paths')
        self.assertEqual(result['nepali_ids'], [4, 5])

    def test_missing_ipa_mapping_cannot_become_native_support(self):
        result = compile_word_hypotheses([row([None], ['ʌ'])], {'_': 0, 'ʌ': 4}, {0})['कर्म']
        self.assertEqual(result['status'], 'unsupported_g2p_projection')

    def test_acoustic_preference_direction_and_nonexhaustive_paths(self):
        hypothesis = compile_word_hypotheses([row(['a', 'b'], ['a'])], {'_': 0, 'a': 1, 'b': 2}, {0})['कर्म']
        scores = word_path_evidence(np.log(np.array([[.01, .98, .01], [.01, .01, .98]])), hypothesis)
        self.assertLess(scores['hindi_minus_nepali_log_likelihood'], 0)
        self.assertFalse(scores['hypotheses_exhaustive'])
        self.assertEqual(scores['probe_greedy_sequence_ids'], [1, 2])
