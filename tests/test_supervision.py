from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hindi_detector.supervision import approved_labels, require_training_labels


class SupervisionGuards(unittest.TestCase):
    def event(self, reviewer="A", judgment="native_acceptable", split="reserved_evaluation"):
        return {"source": "human_blind_review", "native_nepali": True, "reviewer": reviewer,
                "judgment": judgment, "clip_id": "clip1", "audio_sha256": "a" * 64,
                "speaker_id": "speaker1", "sentence_id": "sentence1", "variety": "Kathmandu",
                "split": split}

    def consume(self, rows, split="reserved_evaluation"):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            return approved_labels(path, variety="Kathmandu", split=split)

    def test_requires_two_independent_reviewers(self):
        rows, counts = self.consume([self.event(), self.event()])
        self.assertEqual(rows, [])
        self.assertEqual(counts["awaiting_additional_raters"], 1)

    def test_disagreement_is_not_forced_to_either_class(self):
        rows, counts = self.consume([self.event(), self.event("B", "hindi_like")])
        self.assertEqual(rows, [])
        self.assertEqual(counts["excluded_uncertain_other_or_disputed"], 1)

    def test_evaluation_annotations_are_excluded_from_training(self):
        rows, counts = self.consume([self.event(), self.event("B")], split="train")
        self.assertEqual(rows, [])
        self.assertEqual(counts["excluded_wrong_split"], 1)
        with self.assertRaises(ValueError):
            require_training_labels(rows)

    def test_model_identity_cannot_supply_labels(self):
        row = self.event()
        row["source"] = "system_name_pseudo_label"
        with self.assertRaises(ValueError):
            self.consume([row])

    def test_revisions_use_latest_review_not_duplicate_votes(self):
        rows, _ = self.consume([self.event(), self.event("A", "hindi_like"), self.event("B", "hindi_like")])
        self.assertEqual(rows[0]["class_id"], 1)
        self.assertEqual(rows[0]["rater_count"], 2)


if __name__ == "__main__":
    unittest.main()
