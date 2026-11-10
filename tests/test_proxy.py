from __future__ import annotations

import unittest

from hindi_detector.proxy import LABEL_SOURCE, check_disjoint, check_proxy_pairs


class ProxyIntegrity(unittest.TestCase):
    def pair(self):
        base = {'speaker_id': 'voice', 'sentence_id': 'text', 'text_sha256': 'hash',
                'split': 'train', 'label_source': LABEL_SOURCE}
        return [{**base, 'system': 'indicf5_stock', 'class_id': 1},
                {**base, 'system': 'audio8_r3', 'class_id': 0}]

    def test_unmatched_source_cannot_enter_paired_training(self):
        with self.assertRaises(ValueError):
            check_proxy_pairs(self.pair()[:1], 'train')
        check_proxy_pairs(self.pair(), 'train')

    def test_proxy_cannot_masquerade_as_human_gold(self):
        rows = self.pair()
        rows[0]['label_source'] = 'human_blind_review'
        with self.assertRaises(ValueError):
            check_proxy_pairs(rows, 'train')

    def test_same_audio_cannot_cross_proxy_split(self):
        keys = ['speaker_id', 'sentence_id', 'text_sha256', 'audio_sha256']
        left, right = {k: 'left' for k in keys}, {k: 'right' for k in keys}
        right['audio_sha256'] = left['audio_sha256']
        with self.assertRaises(ValueError):
            check_disjoint([left], [right])
