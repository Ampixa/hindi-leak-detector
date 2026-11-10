import unittest

from hindi_detector.content_quality import normalize, transcript_metrics


class ContentMetrics(unittest.TestCase):
    def test_nepali_marks_are_preserved(self):
        self.assertEqual(normalize('तपाईं, आउनुभयो?'), 'तपाईं आउनुभयो')

    def test_content_deletion_is_not_a_pronunciation_label(self):
        result = transcript_metrics('म घर जान्छु', 'म जान्छु')
        self.assertAlmostEqual(result['wer'], 1 / 3)
        self.assertIsNone(result['actual_hindi_error_label'])

    def test_hallucination_can_exceed_one(self):
        self.assertGreater(transcript_metrics('घर', 'धेरै फरक शब्द छन्')['wer'], 1.)

    def test_empty_reference_is_an_error(self):
        with self.assertRaises(ValueError):
            transcript_metrics('।', '')


if __name__ == '__main__':
    unittest.main()
