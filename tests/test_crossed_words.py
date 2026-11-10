from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hindi_detector.crossed_words import LABELS, PREFERRED, PRIORS, crossed_pairs
from hindi_detector.word_model import bag_weights, load_bags


def example():
    bags, jobs = [], {}
    for i in range(3):
        job = f'job_{i}'
        jobs[job] = {'id': job, 'speaker_id': str(i), 'split': 'train', 'prompt_id': 'prompt', 'focus_word': 'कर्म'}
        for system in LABELS:
            identity = job + '_' + system
            bags.append({'clip_id': identity, 'audio_sha256': identity, 'sentence_id': job, 'speaker_id': str(i),
                         'split': 'train', 'system': system, 'class_id': None, 'words': [{'word': 'कर्म'}],
                         'phonetics': np.ones((1, 2)), 'prosody': np.ones((1, 1)), 'phone_mask': np.array([True])})
    return bags, jobs


class CrossedWordIntegrity(unittest.TestCase):
    def test_reference_is_preferred_and_its_voice_differs(self):
        bags, jobs = example(); pairs = crossed_pairs(bags, jobs)
        self.assertEqual(len(pairs), 18)
        self.assertTrue(all(p['reference_system'] == PREFERRED and p['speaker_id'] != p['reference_speaker_id'] for p in pairs))
        self.assertTrue(all(p['focus_index'] == 0 for p in pairs))

    def test_incomplete_matrix_does_not_make_fake_negatives(self):
        bags, jobs = example()
        with self.assertRaisesRegex(ValueError, 'three actual'):
            crossed_pairs(bags[:-1], jobs)

    def test_source_priors_are_not_changed_by_duplicate_pair_counts(self):
        bags, _ = example(); weights = bag_weights(bags, PRIORS)
        for system, prior in PRIORS.items():
            self.assertAlmostEqual(float(sum(w for w, b in zip(weights, bags, strict=True) if b['system'] == system)), prior, places=6)

    def test_held_out_feature_files_are_not_read_during_training_load(self):
        with tempfile.TemporaryDirectory() as name:
            p = Path(name)
            (p / 'manifest.json').write_text(json.dumps({'items': [{'status': 'ok_unreviewed_word_evidence', 'split': 'reserved_evaluation', 'detail': 'must-not-read.json', 'features': 'must-not-read.npz'}]}))
            bags, _ = load_bags(p, splits={'train'})
            self.assertEqual(bags, [])
