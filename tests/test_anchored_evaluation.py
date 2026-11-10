import unittest

from scripts.evaluate_anchored_detector import family_summary


class AttemptedDenominators(unittest.TestCase):
    def test_failed_positive_is_a_miss(self):
        rows = [{'label': 1, 'status': 'measured', 'candidate': True},
                {'label': 1, 'status': 'abstain', 'candidate': None}]
        result = family_summary(rows)
        self.assertEqual(result['recall'], .5)
        self.assertEqual(result['abstention_rate'], .5)

    def test_failed_negative_is_not_counted_correct(self):
        rows = [{'label': 0, 'status': 'measured', 'candidate': False},
                {'label': 0, 'status': 'abstain', 'candidate': None}]
        result = family_summary(rows)
        self.assertEqual(result['correct_decision_rate'], .5)
        self.assertEqual(result['false_flag_rate'], 0)

    def test_mixed_family_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            family_summary([{'label': 0, 'status': 'abstain'}, {'label': 1, 'status': 'abstain'}])
