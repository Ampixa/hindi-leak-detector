import unittest

from scripts.summarize_real_word_transfer import describe, matched_style


class TransferSummaryGuards(unittest.TestCase):
    def test_empty_distribution_stays_unavailable(self):
        self.assertIsNone(describe([])['median'])

    def test_nonfinite_distribution_fails(self):
        with self.assertRaises(ValueError):
            describe([float('nan')])

    def test_context_mismatch_cannot_become_style_evidence(self):
        query = {'text': 'one context', 'word': {'word': 'same', 'word_index': 1}}
        reference = {'text': 'different context', 'system': 'pocket_nepali_v1c', 'word': query['word']}
        with self.assertRaises(ValueError):
            matched_style(query, reference)


if __name__ == '__main__':
    unittest.main()
