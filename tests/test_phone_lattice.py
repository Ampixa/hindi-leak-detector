import unittest

import numpy as np

from hindi_detector.ctc import log_likelihood
from hindi_detector.phone_lattice import path_score, represent


class PhoneLatticeTests(unittest.TestCase):
    def test_singleton_sets_equal_existing_exact_ctc_including_repeats(self):
        rng = np.random.default_rng(8)
        x = rng.random((17, 5)); x /= x.sum(1, keepdims=True); x = np.log(x)
        for labels in ([1, 2, 3], [2, 2, 1], [3]):
            self.assertAlmostEqual(path_score(x, [frozenset([i]) for i in labels]), log_likelihood(x, list(labels)), places=10)

    def test_missing_internal_nucleus_prefers_deletion(self):
        x = np.full((9, 5), .001)
        x[np.arange(9), [0, 1, 0, 2, 0, 3, 0, 1, 0]] = .996
        x /= x.sum(1, keepdims=True); x = np.log(x)
        self.assertGreater(path_score(x, [{1}, {2}, {3}, {1}]), path_score(x, [{1}, {2}, {4}, {3}, {1}]))

    def test_trace_cannot_skip_any_required_phone(self):
        x = np.full((12, 4), .01)
        x[:4, 1] = .97; x[4:8, 2] = .97; x[8:, 3] = .97
        x /= x.sum(1, keepdims=True)
        spans = path_score(np.log(x), [{1}, {2}, {3}], trace=True)['spans']
        self.assertEqual(len(spans), 3)
        self.assertTrue(all(s['frames'] > 0 for s in spans))
        self.assertTrue(all(a['end_frame'] <= b['start_frame'] for a, b in zip(spans, spans[1:])))

    def test_unsupported_projection_remains_explicit(self):
        with self.assertRaises(ValueError):
            represent('d͡zʱ', {'<pad>': 0, 'a': 4})
