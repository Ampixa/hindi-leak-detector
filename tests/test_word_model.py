from __future__ import annotations

import unittest

import numpy as np

from hindi_detector.word_model import pool_logits, predict_bag


class WeakWordSemantics(unittest.TestCase):
    def test_positive_bag_can_localize_without_labeling_all_words_positive(self):
        self.assertGreater(pool_logits(np.array([-5., -5., 5.])), 0)
        self.assertLess(pool_logits(np.array([-5., -5., -5.])), 0)

    def test_word_order_does_not_change_bag_label(self):
        self.assertAlmostEqual(pool_logits(np.array([-2., 1., .5])), pool_logits(np.array([.5, -2., 1.])))

    def test_unusable_phone_words_do_not_become_zero_risk(self):
        bag = {'phonetics': np.ones((2, 1)), 'phone_mask': np.array([False, False]), 'words': [{}, {}]}
        model = {'branch': 'phonetics', 'feature_mean': [0], 'feature_scale': [1], 'weight': [1], 'bias': 0}
        with self.assertRaises(ValueError):
            predict_bag(model, bag)
