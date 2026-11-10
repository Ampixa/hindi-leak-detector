from __future__ import annotations

import unittest

import numpy as np

from hindi_detector.evaluation import ENCODER_REVISION, agreement_decisions, binary_metrics, grouped_intervals, predict_linear, require_feature_recipe, require_held_out, selective_metrics


class HonestEvaluation(unittest.TestCase):
    def test_abstentions_are_never_counted_as_correct(self):
        decisions = agreement_decisions(np.array([.1, .9, .9, .1]), np.array([.1, .9, .1, .9]))
        result = selective_metrics(np.array([0, 1, 1, 0]), decisions)
        self.assertEqual(result['covered_accuracy'], 1)
        self.assertEqual(result['coverage'], .5)
        self.assertEqual(result['correct_fraction_of_all_items'], .5)
        self.assertEqual(result['positive_recall_including_abstentions'], .5)

    def test_all_abstained_has_no_accuracy_estimate(self):
        result = selective_metrics(np.array([0, 1]), np.array([-1, -1]))
        self.assertIsNone(result['covered_accuracy'])

    def test_same_width_different_encoder_cannot_be_substituted(self):
        manifest = {"model_revision": "other", "feature_dimension": 4096,
                    "layers": [12, 24], "pooling": ["mean", "std"], "items": []}
        with self.assertRaises(ValueError):
            require_feature_recipe(manifest)
        manifest["model_revision"] = ENCODER_REVISION
        require_feature_recipe(manifest)

    def test_no_positive_predictions_does_not_mean_perfect_precision(self):
        result = binary_metrics(np.array([0, 1]), np.array([0.1, 0.2]))
        self.assertIsNone(result["precision"])
        self.assertEqual(result["recall"], 0)

    def test_all_negative_test_set_has_no_recall_estimate(self):
        result = binary_metrics(np.array([0, 0]), np.array([0.1, 0.2]))
        self.assertIsNone(result["recall"])
        self.assertIsNone(result["balanced_accuracy"])

    def test_extreme_linear_logits_are_stable(self):
        model = {"weight": [1000], "bias": 0, "feature_mean": [0], "feature_scale": [1]}
        scores = predict_linear(model, np.array([[-1], [1]]))
        np.testing.assert_allclose(scores, [0, 1])

    def test_one_speaker_cannot_supply_a_cluster_interval(self):
        result = grouped_intervals(np.array([0, 1]), np.array([0.1, 0.9]), ["same", "same"], draws=100)
        self.assertIsNone(result["intervals"]["precision"]["ci95"])

    def test_reused_text_under_new_id_is_still_leakage(self):
        model = {"training_speaker_ids": [], "development_speaker_ids": [],
                 "training_text_sha256": ["same_text"], "development_text_sha256": []}
        with self.assertRaises(ValueError):
            require_held_out(model, [{"speaker_id": "new_speaker", "text_sha256": "same_text", "split": "reserved_evaluation"}])


if __name__ == "__main__":
    unittest.main()
