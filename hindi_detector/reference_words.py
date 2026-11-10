"""Compare aligned words with a preferred reading of the same text."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from hindi_detector.word_model import LABELS, predict_bag

RECIPE = 'signed_and_absolute_query_minus_preferred_word_v1'
PREFERRED = {'native_recording', 'audio8_r3'}


def pair_words(query: dict, reference: dict) -> dict:
    if query['clip_id'] == reference['clip_id'] or query['audio_sha256'] == reference['audio_sha256']:
        raise ValueError('Self-reference would make negative pairs artificially easy')
    if query['split'] != reference['split']:
        raise ValueError('Reference and query must remain in the same split')
    if [w['word'] for w in query['words']] != [w['word'] for w in reference['words']]:
        raise ValueError('Reference must read exactly the same word sequence')
    result = {k: query[k] for k in ['words', 'speaker_id', 'sentence_id', 'split', 'system', 'class_id']}
    result.update(clip_id=query['clip_id'] + '__vs__' + reference['clip_id'],
                  query_clip_id=query['clip_id'], reference_clip_id=reference['clip_id'],
                  query_audio_sha256=query['audio_sha256'], reference_audio_sha256=reference['audio_sha256'],
                  reference_speaker_id=reference['speaker_id'], reference_system=reference['system'], recipe=RECIPE)
    for key in ['phonetics', 'prosody']:
        left, right = np.asarray(query[key]), np.asarray(reference[key])
        if left.shape != right.shape or left.ndim != 2 or not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError('Reference word feature dimensions/values disagree')
        delta = left - right
        result[key] = np.concatenate([delta, np.abs(delta)], axis=1)
    result['phone_mask'] = query['phone_mask'] & reference['phone_mask']
    return result


def preferred_pairs(bags: list[dict]) -> list[dict]:
    """Two positive comparisons and two non-self negative comparisons per text."""
    groups = defaultdict(dict)
    for bag in bags:
        if bag['system'] not in LABELS:
            continue
        group = groups[(bag['sentence_id'], bag['speaker_id'], bag['split'])]
        if bag['system'] in group:
            raise ValueError('Duplicate source reading for a text/reference group')
        group[bag['system']] = bag
    if not groups:
        raise ValueError('No matched reference groups')
    result = []
    for group in groups.values():
        if set(group) != set(LABELS):
            raise ValueError('Need matched F5, Audio8 R3 and native words; never fabricate a missing reference')
        for query, reference in [('indicf5_stock', 'native_recording'), ('indicf5_stock', 'audio8_r3'),
                                 ('native_recording', 'audio8_r3'), ('audio8_r3', 'native_recording')]:
            result.append(pair_words(group[query], group[reference]))
    return result


def predict_reference(model: dict, query: dict, reference: dict) -> tuple[float, list[float | None]]:
    if model.get('reference_recipe') != RECIPE or reference['system'] not in PREFERRED:
        raise ValueError('Unsupported reference recipe or unapproved reference source')
    return predict_bag(model, pair_words(query, reference))
