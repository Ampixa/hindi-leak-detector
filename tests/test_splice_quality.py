import unittest

from hindi_detector.splice_quality import boundary_issues


class SpliceBoundaryTests(unittest.TestCase):
    def test_sentence_edge_is_quarantined_before_writing_either_control(self):
        words = [{'word': 'नष्ट', 'start': .5, 'end': 1.}]
        result = boundary_issues([16000, 16000], words, words, [0])
        self.assertEqual(len(result), 2)
        self.assertTrue(all(r['fade_samples_available'] == 0 for r in result))

    def test_supported_word_keeps_existing_fade_recipe(self):
        words = [{'word': 'काम', 'start': .2, 'end': .8}]
        self.assertEqual(boundary_issues([16000, 16000], words, words, [0]), [])

    def test_different_word_sequences_are_not_silently_quarantined(self):
        with self.assertRaises(ValueError):
            boundary_issues([16000, 16000], [{'word': 'एक'}], [{'word': 'दुई'}], [0])
