from __future__ import annotations

import unittest

from hindi_detector.schwa_probe import site_paths


class SiteCounterfactualIntegrity(unittest.TestCase):
    def test_only_the_selected_schwa_gap_changes(self):
        candidate = dict(target_phones=['k', 'ax', 'r', 'm', 'ax'], hindi_schwa_hypothesis=['k', 'ax', 'r', 'm'],
                         sites=[{'gap_index': 3, 'target_count': 1, 'hindi_count': 0}])
        paths = site_paths(candidate, {'k': 'k', 'ax': 'ʌ', 'r': 'r', 'm': 'm'},
                           {'k': 4, 'r': 5, 'm': 6, 'ʌ': 7, 'ə': 8, 'ɐ': 9})
        self.assertEqual(paths[0]['alternative_project_phones'], ['k', 'ax', 'r', 'm'])
        self.assertEqual(paths[0]['vowel_quality_nuisance_paths'][1]['alternative_ids'], [4, 8, 5, 6])

    def test_unsupported_dental_is_not_silently_replaced(self):
        candidate = dict(target_phones=['t', 'ax'], hindi_schwa_hypothesis=['t'],
                         sites=[{'gap_index': 1, 'target_count': 1, 'hindi_count': 0}])
        with self.assertRaisesRegex(ValueError, 'cannot represent'):
            site_paths(candidate, {'t': 't̪', 'ax': 'ʌ'}, {'t': 4, 'ʌ': 5, 'ə': 6, 'ɐ': 7})

    def test_inconsistent_site_is_rejected(self):
        candidate = dict(target_phones=['k', 'ax'], hindi_schwa_hypothesis=['k'], sites=[])
        with self.assertRaisesRegex(ValueError, 'metadata'):
            site_paths(candidate, {'k': 'k', 'ax': 'ʌ'}, {'k': 4, 'ʌ': 5, 'ə': 6, 'ɐ': 7})
