from __future__ import annotations

import unittest

from hindi_detector.control_measurements import attempt


class PostEditDenominator(unittest.TestCase):
    def test_positive_failure_stays_in_denominator_with_label(self):
        def failure():
            raise ValueError('Neither pronunciation fits edited frames')
        records = [attempt({'pair_id': 'a', 'family': 'P1', 'label': 1}, failure),
                   attempt({'pair_id': 'a', 'family': 'S0', 'label': 0}, lambda: {'features': [1.]})]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]['label'], 1)
        self.assertEqual(records[0]['status'], 'abstain')
        self.assertIsNone(records[0]['features'])
        self.assertEqual(records[1]['status'], 'measured')

    def test_unexpected_system_failure_propagates(self):
        def failure():
            raise RuntimeError('Encoder allocation failure')
        with self.assertRaises(RuntimeError):
            attempt({'pair_id': 'a', 'family': 'P1', 'label': 1}, failure)
