"""Low-dimensional word comparisons against actual same-text human speech."""
from __future__ import annotations

from collections import defaultdict
import hashlib

import numpy as np

from hindi_detector.local_features import PROSODY_NAMES

PHONE_NAMES = ['hellinger', 'jensen_shannon', 'posterior_l1', 'posterior_cosine_distance',
               'spread_l1', 'spread_l2', 'entropy_delta', 'spread_mass_delta', 'central_vowel_mass_delta']
PHONE_NAMES += [f'{s}_mass_delta' for s in ('ə', 'ɐ', 'ʌ')] + [f'{s}_spread_delta' for s in ('ə', 'ɐ', 'ʌ')]
PROSODY_COMPARISON_NAMES = ['log_relative_duration_ratio', 'pause_before_delta', 'pause_after_delta',
                           'voiced_fraction_delta', 'energy_mean_delta', 'energy_change_delta']
PROSODY_COMPARISON_NAMES += [f'energy_shape_delta_{i}' for i in range(8)]
PROSODY_COMPARISON_NAMES += ['pitch_pair_available', 'pitch_mean_delta', 'pitch_range_delta', 'pitch_change_delta', 'pitch_shape_rmse']
PROSODY_COMPARISON_NAMES += [f'pitch_shape_delta_{i}' for i in range(8)]
SOURCE_LABELS = {'indicf5_stock': 1, 'pocket_nepali_v1c': 0}


def metadata_partitions(contexts: list[dict]) -> dict:
    """Keep target lexemes, source speakers and their transitive overlaps together."""
    parent = {}

    def find(node):
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(left, right):
        parent[find(left)] = find(right)

    ids = set()
    for row in contexts:
        if row['id'] in ids or not row['target_words'] or len(set(row['target_words'])) != len(row['target_words']):
            raise ValueError('Need unique contexts with explicit unique target words')
        ids.add(row['id'])
        for word in row['target_words']:
            union('speaker:' + row['native_speaker_id'], 'word:' + word)
    groups = defaultdict(list)
    for row in contexts:
        groups[find('speaker:' + row['native_speaker_id'])].append(row)
    components = []
    for rows in groups.values():
        words = sorted({w for r in rows for w in r['target_words']})
        speakers = sorted({r['native_speaker_id'] for r in rows})
        identifier = hashlib.sha256(('native-reference-v10:' + '|'.join(words + speakers)).encode()).hexdigest()
        components.append({'id': identifier, 'contexts': sorted(r['id'] for r in rows), 'words': words, 'native_source_groups': speakers})
    names = ['train', 'development', 'reserved_evaluation']
    sizes = dict.fromkeys(names, 0); target = dict(zip(names, [len(contexts) * .6, len(contexts) * .2, len(contexts) * .2], strict=True))
    assigned = {}; component_splits = {}
    for component in sorted(components, key=lambda c: (-len(c['contexts']), c['id'])):
        split = max(names, key=lambda s: (target[s] - sizes[s], -names.index(s)))
        component_splits[component['id']] = split
        for identifier in component['contexts']:
            assigned[identifier] = split
        sizes[split] += len(component['contexts'])
    if any(sizes[s] < 2 for s in names):
        raise ValueError('Not enough independent context components for three partitions')
    return {'context_partitions': assigned, 'components': components, 'component_partitions': component_splits, 'context_counts': sizes}


def phone_comparison(query: np.ndarray, reference: np.ndarray, vocabulary: dict[str, int]) -> np.ndarray:
    n = len(vocabulary); q, r = np.asarray(query, float), np.asarray(reference, float)
    if set(vocabulary.values()) != set(range(n)) or q.shape != (2 * n,) or r.shape != q.shape:
        raise ValueError('Need actual dense phone vocabulary and matching mean/spread vectors')
    if not np.isfinite(q).all() or not np.isfinite(r).all() or (q < 0).any() or (r < 0).any():
        raise ValueError('Invalid conditional phone posterior/spread')
    a, b, sa, sb = q[:n], r[:n], q[n:], r[n:]
    if not np.isclose(a.sum(), 1., atol=2e-5) or not np.isclose(b.sum(), 1., atol=2e-5):
        raise ValueError('Unavailable phone evidence is not a zero-risk feature vector')
    mixture = (a + b) / 2
    js = .5 * (np.sum(a[a > 0] * np.log(a[a > 0] / mixture[a > 0])) + np.sum(b[b > 0] * np.log(b[b > 0] / mixture[b > 0])))
    entropy = lambda p: -float(np.sum(p[p > 0] * np.log(p[p > 0])))
    central = [vocabulary[token] for token in ('ə', 'ɐ', 'ʌ')]
    values = [np.sqrt(max(0., 1. - float(np.sqrt(a * b).sum()))), js, np.abs(a - b).sum(),
              1. - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)),
              np.abs(sa - sb).sum(), np.linalg.norm(sa - sb), entropy(a) - entropy(b),
              sa.sum() - sb.sum(), a[central].sum() - b[central].sum(),
              *(a[central] - b[central]), *(sa[central] - sb[central])]
    result = np.asarray(values, dtype=float)
    if result.shape != (len(PHONE_NAMES),) or not np.isfinite(result).all():
        raise ValueError('Invalid native phone comparison')
    return result


def prosody_comparison(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    q, r = np.asarray(query, float), np.asarray(reference, float)
    if q.shape != (len(PROSODY_NAMES),) or r.shape != q.shape or not np.isfinite(q).all() or not np.isfinite(r).all():
        raise ValueError('Need matching observed relative-prosody vectors')
    q, r = dict(zip(PROSODY_NAMES, q, strict=True)), dict(zip(PROSODY_NAMES, r, strict=True))
    if min(q['duration_relative'], r['duration_relative']) <= 0:
        raise ValueError('Word duration reference must be positive')
    if any(not 0 <= p['voiced_fraction'] <= 1 or p['pitch_available'] not in (0, 1) for p in (q, r)):
        raise ValueError('Invalid measured pitch availability/voicing')
    energy_q = np.array([q[f'energy_bin_{i}'] for i in range(8)])
    energy_r = np.array([r[f'energy_bin_{i}'] for i in range(8)])
    energy = (energy_q - energy_q.mean()) - (energy_r - energy_r.mean())
    available = bool(q['pitch_available'] and r['pitch_available'] and min(q['voiced_fraction'], r['voiced_fraction']) >= .5)
    pitch = np.zeros(8); differences = [0., 0., 0., 0.]
    if available:
        a = np.array([q[f'pitch_bin_{i}'] for i in range(8)])
        b = np.array([r[f'pitch_bin_{i}'] for i in range(8)])
        pitch = (a - a.mean()) - (b - b.mean())
        differences = [q['pitch_mean_st'] - r['pitch_mean_st'], q['pitch_range_st'] - r['pitch_range_st'],
                       q['pitch_change_st'] - r['pitch_change_st'], float(np.sqrt(np.mean(pitch ** 2)))]
    result = np.array([np.log(q['duration_relative'] / r['duration_relative']),
                       q['pause_before_relative'] - r['pause_before_relative'], q['pause_after_relative'] - r['pause_after_relative'],
                       q['voiced_fraction'] - r['voiced_fraction'], q['energy_mean_db'] - r['energy_mean_db'],
                       q['energy_change_db'] - r['energy_change_db'], *energy, float(available), *differences, *pitch])
    if result.shape != (len(PROSODY_COMPARISON_NAMES),) or not np.isfinite(result).all():
        raise ValueError('Invalid native prosody comparison')
    return result


def pair_with_native(query: dict, reference: dict, vocabulary: dict[str, int], split: str) -> dict:
    return pair_with_reference(query, reference, vocabulary, split, reference_kind='human')


def pair_with_reference(query: dict, reference: dict, vocabulary: dict[str, int], split: str, *, reference_kind: str, inference: bool = False) -> dict:
    if query['clip_id'] == reference['clip_id'] or query['audio_sha256'] == reference['audio_sha256'] or query['speaker_id'] == reference['speaker_id']:
        raise ValueError('A native comparison requires non-self audio and a distinct source group')
    kinds = {'human': 'human_nepali_unlabeled', 'pocket': 'pocket_nepali_v1c'}
    if reference_kind not in kinds or reference['system'] != kinds[reference_kind] or (not inference and query['system'] not in SOURCE_LABELS):
        raise ValueError('Reference kind or declared generated comparison source differs')
    if query['text'] != reference['text'] or [w['word'] for w in query['words']] != [w['word'] for w in reference['words']]:
        raise ValueError('Human prosody reference must read the identical full text/word sequence')
    n = len(query['words'])
    if any(np.asarray(b['phonetics']).shape != (n, 2 * len(vocabulary)) or np.asarray(b['prosody']).shape != (n, len(PROSODY_NAMES)) or np.asarray(b['phone_mask']).shape != (n,) for b in (query, reference)):
        raise ValueError('Native comparison feature geometry differs')
    mask = query['phone_mask'] & reference['phone_mask']
    phonetics = np.zeros((n, len(PHONE_NAMES)))
    for index in np.flatnonzero(mask):
        phonetics[index] = phone_comparison(query['phonetics'][index], reference['phonetics'][index], vocabulary)
    prosody = np.stack([prosody_comparison(a, b) for a, b in zip(query['prosody'], reference['prosody'], strict=True)])
    return {**{k: query[k] for k in ('clip_id', 'audio_sha256', 'text', 'sentence_id', 'speaker_id', 'system', 'words')},
            'class_id': SOURCE_LABELS.get(query['system']), 'split': split, 'phonetics': phonetics,
            'prosody': prosody, 'phone_mask': mask, 'reference_clip_id': reference['clip_id'],
            'reference_audio_sha256': reference['audio_sha256'], 'reference_speaker_id': reference['speaker_id'],
            'reference_kind': reference_kind}
