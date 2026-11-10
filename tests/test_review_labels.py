from __future__ import annotations

import unittest

from hindi_detector.review_server import validate_label


class HumanLabelValidation(unittest.TestCase):
    def setUp(self):
        self.item = {"duration": 3.0}
        self.payload = {"reviewer": "AB", "variety": "Kathmandu Nepali", "native_nepali": True,
                        "judgment": "uncertain", "prosody": "native_acceptable", "spans": []}

    def test_uncertainty_survives_without_becoming_hindi(self):
        self.assertEqual(validate_label(self.payload, self.item)["judgment"], "uncertain")

    def test_model_identity_is_not_a_valid_label(self):
        self.payload["judgment"] = "indicf5"
        with self.assertRaises(ValueError):
            validate_label(self.payload, self.item)

    def test_nonfinite_or_outside_word_spans_fail(self):
        for end in [float("nan"), 4.0]:
            self.payload["spans"] = [{"word": "आज", "start": 0.1, "end": end,
                                      "realization": "deleted", "judgment": "uncertain"}]
            with self.assertRaises(ValueError):
                validate_label(self.payload, self.item)

    def test_native_competence_is_explicit(self):
        self.payload["native_nepali"] = False
        with self.assertRaises(ValueError):
            validate_label(self.payload, self.item)


if __name__ == "__main__":
    unittest.main()
