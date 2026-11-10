import unittest

from hindi_detector.interventions import LABEL_SOURCE, require_disjoint, require_partition


def pair(partition='train', word='word', speaker=1):
    return [{'id': f'{word}-{speaker}-{i}', 'sha256': f'{word}-{speaker}-{i}',
             'word': word, 'candidate_id': word, 'speaker_id': speaker, 'length_scale': .85,
             'condition': name, 'intervention_class_id': i, 'label_source': LABEL_SOURCE,
             'partition': partition, 'word_split': partition, 'voice_split': partition,
             'actual_hindi_leakage_label': None, 'human_pronunciation_gold': False}
            for i, name in enumerate(('lexicon_path', 'site_count_alternative'))]


class InterventionSupervisionGuards(unittest.TestCase):
    def test_missing_pair_is_rejected(self):
        with self.assertRaises(ValueError):
            require_partition(pair()[:1], 'train')

    def test_synthetic_labels_cannot_claim_human_gold(self):
        rows = pair(); rows[0]['human_pronunciation_gold'] = True
        with self.assertRaises(ValueError):
            require_partition(rows, 'train')

    def test_voice_and_word_both_must_be_held_out(self):
        train = pair(); dev = pair('development', 'other', 1)
        with self.assertRaises(ValueError):
            require_disjoint(train, dev)
        dev = pair('development', 'word', 2)
        with self.assertRaises(ValueError):
            require_disjoint(train, dev)
        require_disjoint(train, pair('development', 'other', 2))

    def test_crossed_diagnostic_cannot_enter_core_training(self):
        rows = pair(); rows[0]['voice_split'] = 'reserved_evaluation'
        with self.assertRaises(ValueError):
            require_partition(rows, 'train')


if __name__ == '__main__':
    unittest.main()
