"""Preferred-word comparisons whose query and reference voice groups differ."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from hindi_detector.evaluation import binary_metrics
from hindi_detector.reference_words import pair_words
from hindi_detector.word_model import predict_bag

PREFERRED = 'pocket_nepali_v1c'
LABELS = {'indicf5_stock': 1, 'pocket_hindi_parent': 1, PREFERRED: 0}
PRIORS = {'indicf5_stock': .25, 'pocket_hindi_parent': .25, PREFERRED: .5}


def crossed_pairs(bags: list[dict], jobs: dict[str, dict]) -> list[dict]:
    cells = defaultdict(dict)
    prompts = defaultdict(list)
    for original in bags:
        job = jobs[original['sentence_id']]
        if original['system'] not in LABELS or any(original[k] != job[k] for k in ['speaker_id', 'split']):
            raise ValueError('Crossed source/job metadata mismatch')
        bag = original | {'class_id': LABELS[original['system']]}
        if bag['system'] in cells[job['id']]:
            raise ValueError('Duplicate TTS source in a crossed cell')
        cells[job['id']][bag['system']] = bag
    for job_id, sources in cells.items():
        if set(sources) != set(LABELS):
            raise ValueError('Every crossed cell needs all three actual source readings')
        job = jobs[job_id]
        prompts[(job['split'], job['prompt_id'])].append((job, sources))
    if not prompts:
        raise ValueError('No crossed prompts')
    pairs = []
    for group in prompts.values():
        group.sort(key=lambda row: row[0]['speaker_id'])
        speakers = [j['speaker_id'] for j, _ in group]
        if len(set(speakers)) != len(speakers) or len(speakers) < 2:
            raise ValueError('Need at least two distinct reference source groups per prompt')
        for index, (job, sources) in enumerate(group):
            for offset in range(1, min(2, len(group) - 1) + 1):
                other_job, other = group[(index + offset) % len(group)]
                if job['focus_word'] != other_job['focus_word']:
                    raise ValueError('Reference focus word mismatch')
                reference = other[PREFERRED]
                for query in sources.values():
                    if query['speaker_id'] == reference['speaker_id']:
                        raise ValueError('Same-voice reference would weaken the nuisance control')
                    pair = pair_words(query, reference)
                    indices = [i for i, w in enumerate(pair['words']) if w['word'] == job['focus_word']]
                    if len(indices) != 1:
                        raise ValueError('Focus word must occur exactly once in the actual aligned sequence')
                    pair.update(focus_word=job['focus_word'], focus_index=indices[0], prompt_id=job['prompt_id'])
                    pairs.append(pair)
    return pairs


def grouped_predictions(model: dict, pairs: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for pair in pairs:
        bag, words = predict_bag(model, pair)
        groups[pair['query_clip_id']].append((pair, bag, words))
    output = []
    for identifier, rows in groups.items():
        p = rows[0][0]
        focus = [words[p['focus_index']] for _, _, words in rows]
        score = float(np.mean(focus)) if all(v is not None for v in focus) else None
        output.append({k: p[k] for k in ['system', 'class_id', 'speaker_id', 'sentence_id', 'split', 'focus_word', 'focus_index']} |
                      {'query_clip_id': identifier, 'reference_clip_ids': [row[0]['reference_clip_id'] for row in rows],
                       'reference_speaker_ids': [row[0]['reference_speaker_id'] for row in rows],
                       'bag_source_evidence': float(np.mean([row[1] for row in rows])), 'focus_source_evidence': score,
                       'words': [{'word': word['word'], 'start': word['start'], 'end': word['end'],
                                  'source_evidence': float(np.mean([r[2][i] for r in rows])) if all(r[2][i] is not None for r in rows) else None}
                                 for i, word in enumerate(p['words'])]})
    return output


def focus_log_loss(model: dict, pairs: list[dict]) -> float:
    rows = grouped_predictions(model, pairs)
    if any(row['focus_source_evidence'] is None for row in rows):
        raise ValueError('Focus CV coverage incomplete; never fill missing phones with low risk')
    loss = 0.
    for system, prior in PRIORS.items():
        group = [r for r in rows if r['system'] == system]
        if not group:
            raise ValueError('Focus CV missing a source')
        p = np.clip([r['focus_source_evidence'] for r in group], 1e-8, 1 - 1e-8)
        y = LABELS[system]
        loss += prior * float(np.mean(-(y * np.log(p) + (1 - y) * np.log1p(-p))))
    return loss


def metrics(rows: list[dict]) -> dict:
    result = {'queries': len(rows), 'focus_score_meaning': 'Weak source evidence on the held-out focal word, not certified pronunciation correctness'}
    for name, systems in [('pocket_same_architecture', {PREFERRED, 'pocket_hindi_parent'}),
                          ('indicf5_vs_pocket', {PREFERRED, 'indicf5_stock'})]:
        members = [r for r in rows if r['system'] in systems]
        valid = [r for r in members if r['focus_source_evidence'] is not None]
        if {r['system'] for r in valid} != systems:
            raise ValueError('Cannot report both-source focus metrics without coverage')
        result[name] = {'n': len(members), 'focus_coverage': len(valid) / len(members),
                        'focus_source_metrics': binary_metrics(np.array([r['class_id'] for r in valid]), np.array([r['focus_source_evidence'] for r in valid])),
                        'whole_bag_source_metrics': binary_metrics(np.array([r['class_id'] for r in members]), np.array([r['bag_source_evidence'] for r in members]))}
    return result
