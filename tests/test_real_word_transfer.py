import unittest

from hindi_detector.real_word_transfer import crop_windows, human_reference_rank, prosody_difference, summarize_component_windows


class RealWordInferenceGuards(unittest.TestCase):
    def test_short_crops_are_unavailable_instead_of_padded(self):
        windows = crop_windows(24000, 24000, .3, .34)
        self.assertFalse(windows[0]['within_component_duration_contract'])
        self.assertEqual(windows[0]['source_first_frame'], 6480)

    def test_missing_baseline_does_not_select_another_window(self):
        windows = [{'request_scores': None}, {'request_scores': {'ctc_site': .95}}]
        result = summarize_component_windows(windows, 'ctc_site')
        self.assertIsNone(result['baseline_request_score'])
        self.assertFalse(result['all_windows_supported'])

    def test_request_class_change_is_not_a_hindi_label(self):
        windows = [{'request_scores': {'ctc_site': value}} for value in (.2, .7, .2, .3, .4, .4, .3)]
        result = summarize_component_windows(windows, 'ctc_site')
        self.assertTrue(result['request_class_changes'])
        self.assertIsNone(result['actual_hindi_error_probability'])

    def test_repeated_words_do_not_inflate_human_source_groups(self):
        q = {'all_windows_supported': True, 'baseline_request_score': .9, 'minimum_request_score': .8, 'maximum_request_score': .95}
        refs = [dict(q, speaker_id='one')] * 20
        result = human_reference_rank(q, refs, exclude_speaker_id=None)
        self.assertEqual(result['source_groups'], 1)
        self.assertEqual(result['status'], 'insufficient_human_or_boundary_coverage')

    def test_rank_does_not_become_native_acceptability(self):
        q = {'all_windows_supported': True, 'baseline_request_score': .9, 'minimum_request_score': .8, 'maximum_request_score': .95}
        refs = [dict(q, speaker_id=str(i), baseline_request_score=.2, minimum_request_score=.1, maximum_request_score=.3) for i in range(6)]
        result = human_reference_rank(q, refs, exclude_speaker_id=None)
        self.assertTrue(result['query_all_window_scores_above_observed_human_envelope'])
        self.assertIsNone(result['native_acceptability'])
        self.assertIsNone(result['hindi_error_probability'])

    def test_query_source_group_cannot_supply_its_own_support(self):
        q = {'all_windows_supported': True, 'baseline_request_score': .9, 'minimum_request_score': .8, 'maximum_request_score': .95}
        refs = [dict(q, speaker_id=str(i)) for i in range(6)]
        result = human_reference_rank(q, refs, exclude_speaker_id='0')
        self.assertEqual(result['source_groups'], 5)
        self.assertEqual(result['status'], 'insufficient_human_or_boundary_coverage')

    def test_pitch_shape_excludes_constant_word_register_offset(self):
        q = {'duration_relative': 1., 'pause_after_relative': 0., 'pitch_available': 1., 'voiced_fraction': .9, 'pitch_mean_st': 3., 'pitch_change_st': 2., **{f'pitch_bin_{i}': float(i + 3) for i in range(8)}}
        r = dict(q, pitch_mean_st=0., **{f'pitch_bin_{i}': float(i) for i in range(8)})
        result = prosody_difference(q, r)
        self.assertEqual(result['pitch_shape_rmse_st'], 0.)
        self.assertEqual(result['relative_pitch_mean_delta_st'], 3.)
        self.assertIsNone(result['actual_hindi_intonation_error'])

    def test_sparse_pitch_cannot_supply_a_full_word_contour(self):
        q = {'duration_relative': 1., 'pause_after_relative': 0., 'pitch_available': 1., 'voiced_fraction': .1}
        result = prosody_difference(q, dict(q, voiced_fraction=.9))
        self.assertIsNone(result['pitch_shape_rmse_st'])
        self.assertEqual(result['pitch_comparison_status'], 'insufficient_pitch_or_voiced_coverage')

    def test_nonfinite_voicing_coverage_is_an_error(self):
        q = {'voiced_fraction': float('nan')}
        with self.assertRaises(ValueError):
            prosody_difference(q, q)


if __name__ == '__main__':
    unittest.main()
