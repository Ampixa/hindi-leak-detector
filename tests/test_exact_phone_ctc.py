from __future__ import annotations

from itertools import product
import unittest

import numpy as np

from hindi_detector.ctc import log_likelihood
from hindi_detector.exact_phone_ctc import path_score


class ExactAllowedPhoneCTC(unittest.TestCase):
    def test_marginal_equals_enumeration_of_concrete_ctc_sequences(self):
        rng = np.random.default_rng(69)
        for units in [[{1, 2}, {2, 3}], [{1, 2}, {1, 2}], [{1}, {2, 3}, {3}], []]:
            for frames in (2, 3, 5):
                probabilities = rng.dirichlet(np.ones(4), size=frames)
                values = np.log(probabilities)
                terms = [log_likelihood(values, list(sequence)) for sequence in product(*units)]
                expected = float(np.logaddexp.reduce(terms))
                self.assertAlmostEqual(path_score(values, units), expected, places=11)

    def test_shared_token_requires_blank_but_different_tokens_can_skip(self):
        same = np.log(np.array([[.001, .001, .997, .001], [.001, .001, .997, .001]]))
        self.assertLess(path_score(same, [{1, 2}, {2, 3}]), -5)
        distinct = np.log(np.array([[.001, .997, .001, .001], [.001, .001, .001, .997]]))
        self.assertGreater(path_score(distinct, [{1, 2}, {2, 3}]), -.02)

    def test_one_phone_cannot_change_token_mid_state(self):
        values = np.log(np.array([[.001, .998, .001], [.001, .001, .998]]))
        self.assertLess(path_score(values, [{1, 2}]), -5)

    def test_trace_matches_exhaustive_viterbi(self):
        rng = np.random.default_rng(70)
        values = np.log(rng.dirichlet(np.ones(4), size=5))
        units = [{1, 2}, {2, 3}]
        candidates = []
        for route in product(range(4), repeat=5):
            decoded = [x for i, x in enumerate(route) if x and (i == 0 or x != route[i - 1])]
            if len(decoded) == len(units) and all(x in u for x, u in zip(decoded, units, strict=True)):
                candidates.append(sum(values[i, token] for i, token in enumerate(route)))
        trace = path_score(values, units, trace=True)
        self.assertAlmostEqual(trace['viterbi_score'], max(candidates), places=11)
        self.assertEqual(len(trace['spans']), 2)
        self.assertLessEqual(trace['spans'][0]['end_frame'], trace['spans'][1]['start_frame'])

    def test_impossible_trace_explicit_and_blank_only_supported(self):
        values = np.log(np.array([[.1, .9]]))
        self.assertTrue(np.isneginf(path_score(values, [{1}, {1}])))
        with self.assertRaises(ValueError):
            path_score(values, [{1}, {1}], trace=True)
        self.assertAlmostEqual(path_score(values, []), np.log(.1))

    def test_sum_and_max_against_independent_frame_route_enumeration(self):
        for seed in (71, 72, 73):
            values = np.log(np.random.default_rng(seed).dirichlet(np.ones(4), size=5))
            for units in [[{1, 2}, {1, 2}], [{2}, {2, 3}], [{1}, {1, 2}, {3}]]:
                scores = []
                for route in product(range(4), repeat=5):
                    decoded = [x for i, x in enumerate(route) if x and (i == 0 or x != route[i - 1])]
                    if len(decoded) == len(units) and all(x in u for x, u in zip(decoded, units, strict=True)):
                        scores.append(sum(values[i, token] for i, token in enumerate(route)))
                result = path_score(values, units, trace=True)
                self.assertAlmostEqual(result['log_score'], float(np.logaddexp.reduce(scores)), places=11)
                self.assertAlmostEqual(result['viterbi_score'], max(scores), places=11)
