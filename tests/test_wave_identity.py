import unittest

import numpy as np

from hindi_detector.wave_identity import comparison


class WaveCorrespondenceTests(unittest.TestCase):
    def test_gain_and_pcm_quantization_preserve_correspondence(self):
        wave = .4 * np.sin(np.arange(3000) * .13)
        staged = np.round(wave * 2 * 32768) / 32768
        result = comparison(wave, 24000, staged, 24000)
        self.assertTrue(result['same_rate_length_and_gain_adjusted_samples_within_1e_minus4'])
        self.assertAlmostEqual(result['least_squares_gain'], 2, places=4)

    def test_added_gap_cannot_count_as_sample_equivalence(self):
        wave = .4 * np.sin(np.arange(3000) * .13)
        staged = np.r_[wave[:1500], np.zeros(6000), wave[1500:]]
        result = comparison(wave, 24000, staged, 24000)
        self.assertFalse(result['same_rate_length_and_gain_adjusted_samples_within_1e_minus4'])

    def test_length_difference_cannot_be_hidden_by_common_prefix(self):
        wave = .4 * np.sin(np.arange(3000) * .13)
        result = comparison(wave, 24000, wave[:-1], 24000)
        self.assertFalse(result['same_rate_length_and_gain_adjusted_samples_within_1e_minus4'])
        self.assertEqual(result['resampled_frame_count_difference'], -1)

    def test_invalid_or_silent_audio_is_explicit(self):
        for bad in (np.zeros(50), np.array([1., np.nan]), np.ones((10, 2))):
            with self.assertRaises(ValueError):
                comparison(bad, 24000, np.ones(50), 24000)


if __name__ == '__main__':
    unittest.main()
