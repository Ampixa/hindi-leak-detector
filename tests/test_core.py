from __future__ import annotations

import itertools
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from hindi_detector.candidates import schwa_slots
from hindi_detector.ctc import compare, log_likelihood
from hindi_detector.prosody import analyse


def collapse(path: tuple[int, ...]) -> list[int]:
    return [p for i, p in enumerate(path) if p != 0 and (i == 0 or p != path[i - 1])]


class CTCValidation(unittest.TestCase):
    def test_matches_exhaustive_paths_including_repeated_phones(self):
        probs = np.random.default_rng(17).dirichlet(np.ones(3), size=5)
        for labels in [[], [1], [1, 1], [1, 2], [1, 2, 1]]:
            exact = sum(math.prod(probs[t, p] for t, p in enumerate(path))
                        for path in itertools.product(range(3), repeat=5) if collapse(path) == labels)
            self.assertAlmostEqual(log_likelihood(np.log(probs), labels), math.log(exact), places=12)

    def test_impossible_repeated_labels(self):
        self.assertEqual(log_likelihood(np.log(np.full((2, 3), 1/3)), [1, 1]), -math.inf)

    def test_probability_input_validation(self):
        for values in [np.array([[float("nan"), -1]]), np.array([[0., 0.]]), np.array([[-1., -1.]])]:
            with self.assertRaises(ValueError):
                log_likelihood(values, [1])

    def test_no_evidence_for_identical_hypotheses(self):
        with self.assertRaises(ValueError):
            compare(np.log(np.full((3, 3), 1/3)), [1], [1])

    def test_ratio_direction(self):
        values = np.log([[0.01, 0.98, 0.01], [0.98, 0.01, 0.01], [0.01, 0.01, 0.98]])
        self.assertGreater(compare(values, [1], [1, 2]).hindi_minus_target_log_likelihood, 0)


class AudioValidation(unittest.TestCase):
    def test_silence_is_not_a_zero_accent_score(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "silence.wav"
            sf.write(path, np.zeros(16000), 16000)
            with self.assertRaises(ValueError):
                analyse(path)

    def test_known_tone_pitch_and_interior_pause(self):
        sr = 16000
        tone = 0.2 * np.sin(2 * np.pi * 200 * np.arange(sr // 2) / sr)
        audio = np.concatenate([np.zeros(3200), tone, np.zeros(4000), tone, np.zeros(3200)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tone.wav"
            sf.write(path, audio, sr)
            result = analyse(path)
            self.assertEqual(result["interior_pause_count"], 1)
            self.assertAlmostEqual(result["f0_median_hz"], 200, delta=3)

    def test_schwa_slot_alignment(self):
        self.assertEqual(schwa_slots(["k", "ax", "r", "m", "ax"]), (("k", "r", "m"), (0, 1, 0, 1)))


if __name__ == "__main__":
    unittest.main()
