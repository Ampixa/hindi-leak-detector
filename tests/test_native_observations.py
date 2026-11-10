import unittest

from hindi_detector.native_observations import matching_observations


class ObservedVariationTests(unittest.TestCase):
    def setUp(self):
        self.site = {'kind': 'vowel_quality', 'index': 1, 'alternative': 'ɛː'}
        self.phones = ['h', 'axj', 'n', 'ax']
        self.bank = {'entries': [{'word': 'हैन', 'site': self.site, 'native_phones': self.phones,
            'recordings': [{'id': 'one', 'voice': 'group_a'}, {'id': 'two', 'voice': 'group_b'}]}]}

    def test_context_matches_exact_contrast_and_excludes_current_recording(self):
        result = matching_observations(self.bank, 'हैन', self.site, self.phones, exclude_case='one')
        self.assertEqual([row['id'] for row in result['cases']], ['two'])
        self.assertEqual(result['status'], 'observed_in_other_real_recordings')
        self.assertIsNone(result['human_error_label'])
        self.assertFalse(result['used_for_prediction'])

    def test_different_phonetic_site_does_not_inherit_word_level_variant(self):
        for site, phones in [({**self.site, 'index': 3}, self.phones),
                             ({**self.site, 'alternative': 'o'}, self.phones),
                             (self.site, ['h', 'e', 'n'])]:
            result = matching_observations(self.bank, 'हैन', site, phones)
            self.assertEqual(result['status'], 'not_observed_in_limited_bank')
            self.assertEqual(result['cases'], [])

    def test_no_bank_is_explicit_and_not_a_native_acceptability_claim(self):
        self.assertEqual(matching_observations(None, 'हैन', self.site, self.phones),
                         {'status': 'bank_unavailable', 'used_for_prediction': False})


if __name__ == '__main__':
    unittest.main()
