"""Explicit synthetic request supervision, separate from human leakage judgments."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from hindi_detector.candidates import schwa_slots

LABEL_SOURCE = 'requested_single_schwa_count_intervention_not_human_error_gold'
CONDITIONS = {'lexicon_path': 0, 'site_count_alternative': 1}


def require_partition(rows: list[dict], partition: str) -> None:
    if partition not in {'train', 'development', 'reserved_evaluation'} or not rows:
        raise ValueError('Need a nonempty named intervention partition')
    pairs = defaultdict(list)
    for row in rows:
        if row['partition'] != partition or row['word_split'] != partition or row['voice_split'] != partition:
            raise ValueError('Intervention word/voice partition mismatch')
        if row['label_source'] != LABEL_SOURCE or row['intervention_class_id'] != CONDITIONS[row['condition']]:
            raise ValueError('Only explicit requested intervention labels may supervise this component')
        if row.get('actual_hindi_leakage_label') is not None or row.get('human_pronunciation_gold') is not False:
            raise ValueError('Synthetic request labels must never masquerade as human leakage gold')
        pairs[(row['word'], row['candidate_id'], row['speaker_id'], row['length_scale'])].append(row['condition'])
    if any(sorted(v) != sorted(CONDITIONS) for v in pairs.values()):
        raise ValueError('Each word/voice/rate needs both intervention conditions exactly once')


def require_disjoint(left: list[dict], right: list[dict]) -> None:
    for key in ('word', 'speaker_id', 'id', 'sha256'):
        if {r[key] for r in left} & {r[key] for r in right}:
            raise ValueError('Intervention split leakage: ' + key)


def intervention_features(log_probs: np.ndarray, site_evidence: dict, candidate: dict, vocab: dict[str, int], special: set[int]) -> dict[str, np.ndarray]:
    values = np.exp(np.asarray(log_probs, dtype=float))
    if values.ndim != 2 or not np.isfinite(values).all() or len(values) < 1:
        raise ValueError('Need nonempty finite phone posterior frames')
    values[:, sorted(special)] = 0
    mass = values.sum(axis=1)
    if mass.sum() <= 1e-5:
        raise ValueError('No usable non-special phone evidence')
    mean = values.sum(axis=0) / mass.sum()
    conditional = values / np.maximum(mass[:, None], 1e-20)
    spread = np.sqrt(((conditional - mean) ** 2 * mass[:, None]).sum(axis=0) / mass.sum())
    skeleton, slots = schwa_slots(candidate['target_phones'])
    if len(candidate['sites']) != 1 or not skeleton:
        raise ValueError('Need a single declared candidate site and a lexical skeleton')
    site = candidate['sites'][0]
    if site['target_count'] != 1 or site['hindi_count'] != 0:
        raise ValueError('This component supports one-to-zero intervention requests only')
    # These target descriptors are identical for both sides of each pair. No
    # requested condition, source voice, realized duration or decoder timing is input.
    central = float(sum(mean[vocab[s]] for s in ('ʌ', 'ə', 'ɐ')))
    structural = np.array([np.log1p(len(skeleton)), np.log1p(sum(slots)),
                           site['gap_index'] / len(skeleton), float(site['gap_index'] == len(skeleton)),
                           central * len(candidate['target_phones']) - sum(slots)], dtype=float)
    variants = site_evidence['vowel_conditioned_results']
    if [r['central_vowel'] for r in variants] != ['ʌ', 'ə', 'ɐ']:
        raise ValueError('Site evidence vowel nuisance recipe changed')
    marginal_native = float(np.logaddexp.reduce([r['lexicon_log_likelihood'] for r in variants]) - np.log(3))
    marginal_alt = float(np.logaddexp.reduce([r['alternative_log_likelihood'] for r in variants]) - np.log(3))
    temporal = np.array([site_evidence['marginal_alternative_minus_lexicon'], marginal_native, marginal_alt,
                         *[r['alternative_minus_lexicon'] for r in variants]], dtype=float) / len(values)
    output = {'ctc_site': np.r_[temporal, structural], 'pooled_phone': np.r_[mean, spread, structural]}
    if any(not np.isfinite(v).all() for v in output.values()):
        raise ValueError('Nonfinite intervention feature vector')
    return {name: value.astype('float32') for name, value in output.items()}
