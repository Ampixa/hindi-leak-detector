from __future__ import annotations

import unittest

import numpy as np

from hindi_detector.anchored_proxy import full_pairs, logistic_objective, source_weights


class AnchoredIntegrity(unittest.TestCase):
    def test_pairwise_regularized_gradient_matches_finite_differences(self):
        z = np.array([[1., 2.], [-1., 0.], [0., -.5]])
        theta = np.array([.3, -.2, .1]); y = np.array([1, 0, 0]); weights = np.array([.5, .25, .25])
        objective = lambda t: logistic_objective(t, z, y, weights, [(0, 1), (0, 2)], .7)
        _, actual = objective(theta)
        eps = 1e-6
        expected = [(objective(theta + np.eye(3)[i] * eps)[0] - objective(theta - np.eye(3)[i] * eps)[0]) / (2 * eps) for i in range(3)]
        np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_crops_cannot_replace_matched_full_utterances(self):
        with self.assertRaises(ValueError):
            full_pairs([{'variant': 'short_1p5'}])

    def test_source_priors_do_not_change_when_a_parent_has_extra_views(self):
        rows = [{'system': s} for s in ['indicf5_stock', 'indicf5_stock', 'audio8_r3', 'native_recording']]
        weights = source_weights(rows)
        self.assertAlmostEqual(float(weights[:2].sum()), .5)
        self.assertAlmostEqual(float(weights[2:].sum()), .5)

    def test_missing_natural_anchor_is_an_error(self):
        with self.assertRaises(ValueError):
            source_weights([{'system': 'indicf5_stock'}, {'system': 'audio8_r3'}])
