import unittest

import numpy as np

from hindi_detector.paired_detector import paired_features


class PairFeatureTests(unittest.TestCase):
    def feature(self, word='राम'):
        return {'detail': {'words': [{'word': word}]}, 'phonetics': np.ones((1, 4), np.float32),
                'prosody': np.ones((1, 3), np.float32), 'hidden_word_12': np.ones((1, 2), np.float32),
                'hidden_word_24': np.ones((1, 2), np.float32)}

    def test_identical_reference_zero_and_signed_difference(self):
        native = self.feature(); changed = self.feature(); changed['prosody'][0, 0] = 3
        projection = np.ones((4, 2), np.float32)
        p, s = paired_features(native, native, projection)
        np.testing.assert_array_equal(p, 0); np.testing.assert_array_equal(s, 0)
        _, positive = paired_features(changed, native, projection)
        _, negative = paired_features(native, changed, projection)
        np.testing.assert_array_equal(positive[:, :3], -negative[:, :3])
        np.testing.assert_array_equal(positive[:, 3:], negative[:, 3:])

    def test_mismatched_text_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'same aligned word'):
            paired_features(self.feature(), self.feature('सीता'), np.ones((4, 2)))
