from __future__ import annotations

import unittest

import numpy as np

from hindi_detector.anchored_detector import cells, fit, score, select, weights


class ReviewedLocalFitting(unittest.TestCase):
    def rows(self):
        return [{'pair_id': f'{word}:{voice}', 'phone_collision_group': f'phonetic-{word}', 'voice': voice,
                 'family': family, 'label': label}
                for word in range(18) for voice in ('a', 'b', 'c', 'd')
                for family, label in [('S0', 0), ('P1', 1), ('P1w', 1), ('S2', 0)]]

    def test_every_source_family_is_disjoint_by_phone_group_and_voice(self):
        rows = self.rows()
        counts = np.zeros(len(rows), int)
        for training, validation in cells(rows):
            counts += validation
            self.assertFalse({r['voice'] for r, m in zip(rows, training) if m} & {r['voice'] for r, m in zip(rows, validation) if m})
            self.assertFalse({r['phone_collision_group'] for r, m in zip(rows, training) if m} & {r['phone_collision_group'] for r, m in zip(rows, validation) if m})
        np.testing.assert_array_equal(counts, np.ones(len(rows)))

    def test_row_shuffle_preserves_source_weights_and_fitted_predictions(self):
        rows = self.rows()
        rng = np.random.default_rng(71)
        x = rng.normal(size=(len(rows), 24))
        x[:, [5, 17]] = -.5
        y = np.array([r['label'] for r in rows])
        permutation = rng.permutation(len(rows))
        moved = [rows[i] for i in permutation]
        np.testing.assert_allclose(weights(moved), weights(rows)[permutation])
        a, sa = fit(x, y, rows, 'local14', .1)
        b, sb = fit(x[permutation], y[permutation], moved, 'local14', .1)
        np.testing.assert_allclose(score(a, sa, x), score(b, sb, x), atol=1e-6)

    def test_anchor_damage_and_censor_fields_are_not_decision_inputs(self):
        a = np.zeros((2, 24))
        b = a.copy()
        b[:, [7, 8, 9, 10, 11, 19, 20, 21, 22, 23]] = 100
        np.testing.assert_array_equal(select(a, 'local14'), select(b, 'local14'))
