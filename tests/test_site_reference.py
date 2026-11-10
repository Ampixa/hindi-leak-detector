from __future__ import annotations

import unittest

from hindi_detector.site_reference import rank_site


class NativeReferenceIntegrity(unittest.TestCase):
    def test_exact_audio_and_query_speaker_are_both_removed(self):
        bank = {'references': [{'speaker_id': str(i), 'score': float(i), 'audio_sha256s': [str(i)]} for i in range(6)]}
        result = rank_site(2., bank, '0', '1')
        self.assertEqual(result['reference_speaker_ids'], ['2', '3', '4', '5'])
        self.assertEqual(result['n'], 4)

    def test_missing_references_do_not_become_low_risk(self):
        result = rank_site(2., {'references': []}, 'query', 'sha')
        self.assertIsNone(result['native_reference_rank'])
        self.assertEqual(result['status'], 'insufficient_independent_native_references')

    def test_duplicate_speakers_cannot_inflate_reference_size(self):
        row = {'speaker_id': 'a', 'score': 1., 'audio_sha256s': ['a']}
        with self.assertRaises(ValueError):
            rank_site(2., {'references': [row, row]}, 'query', 'sha')
