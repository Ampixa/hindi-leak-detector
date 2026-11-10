from __future__ import annotations

from pathlib import Path
import unittest

from hindi_detector.consonant_context import NonSyllabicTable, compile_context_word
from hindi_detector.pronunciation_sites import compile_word

ROOT = Path(__file__).resolve().parents[1]
VOCABULARY = {'<pad>': 0, '<s>': 1, '</s>': 2, '<unk>': 3, 'n': 4, 's': 5, 't': 6, 'ə': 7,
              'a': 9, 'i': 10, 'd': 12, 'e': 16, 'o': 19, 'ʌ': 33, 'ɡ': 35, 'aɪ': 37,
              'h': 39, 'n̩': 169, 't̪': 189, 'ĩ': 314, 'dzː': 382}


class UnknownConsonantSafety(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = NonSyllabicTable(ROOT / 'data/panphon_context_v58')

    def test_vowels_syllabic_consonants_and_unparsed_labels_never_enter_pool(self):
        ids, _ = self.table.pool({**VOCABULARY, '??': 85, 'ɑːɹ': 68})
        self.assertFalse(ids & {7, 9, 10, 16, 19, 33, 37, 68, 85, 169, 314})
        self.assertIn(12, ids)
        self.assertEqual(self.table.classify('d̪')['status'], 'pure_non_syllabic')

    def test_unknown_dental_is_labeled_unknown_rather_than_asserted_alveolar(self):
        unit, strict, projection = self.table.representation('d̪', VOCABULARY)
        self.assertIsNone(strict)
        self.assertEqual(projection['ipa'], 'd̪')
        self.assertFalse(projection['consonant_identity_tested'])
        self.assertGreater(len(unit), 1)

    def test_unicode_nasal_equivalence_is_exact_and_missing_nasal_is_not_oral(self):
        unit, _, projection = self.table.representation('ĩ', VOCABULARY)
        self.assertEqual(unit, {314})
        self.assertEqual(projection['representation_mode'], 'exact_Unicode_canonical_representation')
        with self.assertRaises(ValueError):
            self.table.representation('ã', VOCABULARY)

    def test_empty_g2p_separators_are_explicitly_recorded(self):
        row = {'word': 'fixture', 'nepali_project_variants': [['dental', 'ax', 'd']], 'hindi_units': ['d̪', '', 'd']}
        result = compile_context_word(row, {'dental': 'd̪', 'ax': 'ʌ', 'd': 'd'}, VOCABULARY, self.table)
        self.assertEqual(result['input_normalization']['removed_empty_separator_positions'], [1])
        self.assertEqual([(c['kind'], c['index']) for c in result['variants'][0]['contrasts']], [('vowel_deletion', 1)])
        with self.assertRaises(ValueError):
            compile_context_word({**row, 'hindi_units': ['']}, {'dental': 'd̪', 'ax': 'ʌ', 'd': 'd'}, VOCABULARY, self.table)

    def test_masking_acoustics_does_not_change_supported_lexical_contrasts(self):
        row = {'word': 'fixture', 'nepali_project_variants': [['h', 'axj', 'n', 'ax']], 'hindi_units': ['h', 'e', 'n']}
        ipa = {'h': 'h', 'axj': 'ʌi̯', 'n': 'n', 'ax': 'ʌ'}
        expected = compile_word(row, ipa, VOCABULARY)
        actual = compile_context_word(row, ipa, VOCABULARY, self.table, mask_all=True)
        self.assertEqual([(c['kind'], c['index']) for c in expected['variants'][0]['contrasts']],
                         [(c['kind'], c['index']) for c in actual['variants'][0]['contrasts']])
