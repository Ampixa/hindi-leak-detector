import unittest

import numpy as np

from hindi_detector.local_features import PROSODY_NAMES
from hindi_detector.native_reference import PHONE_NAMES, PROSODY_COMPARISON_NAMES, metadata_partitions, pair_with_native, phone_comparison, prosody_comparison


def prosody(**changes):
    values = dict.fromkeys(PROSODY_NAMES, 0.)
    values.update(duration_relative=1., pitch_available=1., voiced_fraction=.9)
    values.update(changes)
    return np.array([values[k] for k in PROSODY_NAMES])


class NativeReferenceGuards(unittest.TestCase):
    def test_transitive_target_and_speaker_overlap_stays_in_one_partition(self):
        rows = [{'id': str(i), 'native_speaker_id': 'person' + str(i), 'target_words': ['word' + str(i // 2)]} for i in range(24)]
        rows[2]['native_speaker_id'] = rows[0]['native_speaker_id']
        result = metadata_partitions(rows)['context_partitions']
        self.assertEqual(result['0'], result['1'])
        self.assertEqual(result['0'], result['2'])
        self.assertEqual(result['0'], result['3'])
        self.assertEqual(result, metadata_partitions(rows)['context_partitions'])

    def test_unavailable_posterior_is_not_zero_risk(self):
        with self.assertRaises(ValueError):
            phone_comparison(np.zeros(6), np.zeros(6), {'ə': 0, 'ɐ': 1, 'ʌ': 2})

    def test_identical_posterior_has_zero_distance(self):
        vector = np.array([.1, .2, .3, .4, .01, .02, .03, .04])
        result = phone_comparison(vector, vector, {'ə': 0, 'ɐ': 1, 'ʌ': 2, 'k': 3})
        self.assertEqual(result.shape, (len(PHONE_NAMES),))
        np.testing.assert_allclose(result, 0., atol=1e-8)

    def test_word_pitch_offset_is_separate_from_contour_shape(self):
        q = prosody(pitch_mean_st=3., **{f'pitch_bin_{i}': i + 3. for i in range(8)})
        r = prosody(**{f'pitch_bin_{i}': float(i) for i in range(8)})
        result = dict(zip(PROSODY_COMPARISON_NAMES, prosody_comparison(q, r)))
        self.assertEqual(result['pitch_shape_rmse'], 0.)
        self.assertEqual(result['pitch_mean_delta'], 3.)

    def test_missing_pitch_has_an_explicit_availability_feature(self):
        q = prosody(voiced_fraction=.1)
        result = dict(zip(PROSODY_COMPARISON_NAMES, prosody_comparison(q, prosody())))
        self.assertEqual(result['pitch_pair_available'], 0.)
        self.assertEqual(result['pitch_shape_rmse'], 0.)

    def test_prosody_requires_full_text_including_punctuation(self):
        q = {'clip_id': 'q', 'audio_sha256': 'q', 'speaker_id': 'query', 'system': 'indicf5_stock', 'text': 'कर्म?', 'words': [{'word': 'कर्म'}]}
        r = dict(q, clip_id='r', audio_sha256='r', speaker_id='reference', system='human_nepali_unlabeled', text='कर्म।')
        with self.assertRaises(ValueError):
            pair_with_native(q, r, {'ə': 0, 'ɐ': 1, 'ʌ': 2}, 'train')

    def test_reference_cannot_be_the_same_audio_in_disguise(self):
        q = {'clip_id': 'q', 'audio_sha256': 'same', 'speaker_id': 'query'}
        r = {'clip_id': 'r', 'audio_sha256': 'same', 'speaker_id': 'reference'}
        with self.assertRaises(ValueError):
            pair_with_native(q, r, {}, 'train')


if __name__ == '__main__':
    unittest.main()
