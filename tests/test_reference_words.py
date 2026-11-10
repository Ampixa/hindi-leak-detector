from __future__ import annotations

import unittest

import numpy as np

from hindi_detector.reference_words import pair_words


def bag(identifier, value, word='कर्म', split='train'):
    return dict(clip_id=identifier, audio_sha256=identifier, words=[{'word': word}],
                speaker_id='verified_source_group', sentence_id='sentence', split=split,
                system='native_recording', class_id=0, phonetics=np.array([[value, 1.]]),
                prosody=np.array([[value]]), phone_mask=np.array([True]))


class ReferenceIntegrity(unittest.TestCase):
    def test_rejects_self_and_cross_split_reference(self):
        with self.assertRaises(ValueError):
            pair_words(bag('a', 1), bag('a', 1))
        with self.assertRaises(ValueError):
            pair_words(bag('a', 1), bag('b', 2, split='reserved_evaluation'))

    def test_same_text_is_required(self):
        with self.assertRaises(ValueError):
            pair_words(bag('a', 1), bag('b', 2, word='अर्थ'))

    def test_shared_feature_offset_cancels(self):
        a = pair_words(bag('a', 2), bag('b', 1))
        b = pair_words(bag('a', 102), bag('b', 101))
        np.testing.assert_array_equal(a['phonetics'], b['phonetics'])
        np.testing.assert_array_equal(a['prosody'], b['prosody'])
