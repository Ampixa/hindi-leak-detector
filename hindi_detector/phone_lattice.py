"""Phone-set CTC evidence for explicit pronunciation hypotheses.

Sets are an auditable acoustic-model representation, not asserted phonetic
equivalence. Scores are path evidence; they are never accent probabilities.
"""
from __future__ import annotations

import math
import re

import numpy as np

from hindi_detector.ctc import validate_log_probs

VOWELS = {
    'a', 'aː', 'ɑ', 'ɑː', 'æ', 'ä', 'ɒ', 'ə', 'ʌ', 'ɐ', 'ɜ', 'ɜː',
    'e', 'eː', 'ɛ', 'ɛː', 'e̞', 'i', 'iː', 'ɪ', 'i:', 'o', 'oː', 'ɔ', 'ɔː',
    'u', 'uː', 'ʊ', 'aɪ', 'ai', 'əɪ', 'æi', 'ɛɪ', 'aʊ', 'au', 'oɪ', 'ɔɪ',
    'eɪ', 'oʊ', 'əʊ', 'iə', 'ʊə', 'eə', 'ã', 'ã', 'ɑ̃', 'ɐ̃', 'ɛ̃', 'ɔ̃', 'ĩ', 'ũ', 'ũ', 'ẽ', 'õ',
}
REPRESENTATIONS = {
    'ʌ': {'ə', 'ʌ', 'ɐ', 'a', 'ɑ'},
    'ə': {'ə', 'ʌ', 'ɐ', 'a', 'ɑ'},
    'ɐ': {'ə', 'ʌ', 'ɐ', 'a', 'ɑ'},
    'i': {'i', 'iː', 'ɪ', 'i:'}, 'ɪ': {'i', 'iː', 'ɪ', 'i:'},
    'iː': {'i', 'iː', 'ɪ', 'i:'},
    'u': {'u', 'uː', 'ʊ'}, 'uː': {'u', 'uː', 'ʊ'}, 'ʊ': {'u', 'uː', 'ʊ'},
    'e': {'e', 'eː', 'ɛ', 'ɛː', 'e̞'}, 'eː': {'e', 'eː', 'ɛ', 'ɛː', 'e̞'},
    'ɛ': {'e', 'eː', 'ɛ', 'ɛː', 'e̞'}, 'ɛː': {'e', 'eː', 'ɛ', 'ɛː', 'e̞'},
    'o': {'o', 'oː', 'ɔ', 'ɔː'}, 'oː': {'o', 'oː', 'ɔ', 'ɔː'},
    'a': {'a', 'aː', 'ɑ', 'ɑː', 'æ', 'ä'}, 'aː': {'a', 'aː', 'ɑ', 'ɑː', 'æ', 'ä'},
    'ʌi̯': {'aɪ', 'ai', 'əɪ', 'æi', 'ɛɪ', 'a', 'ɑ', 'ɐ'},
    'ai̯': {'aɪ', 'ai', 'əɪ', 'æi', 'ɛɪ', 'a', 'ɑ', 'ɐ'},
    'ʌu̯': {'aʊ', 'au', 'əʊ'}, 'au̯': {'aʊ', 'au', 'əʊ'},
    'oi̯': {'oɪ', 'ɔɪ'},
    'ɦ': {'h'}, 'h': {'h'},
    'r': {'r', 'ɾ'}, 'ɾ': {'r', 'ɾ'},
    'w': {'w', 'ʋ', 'v'}, 'v': {'w', 'ʋ', 'v'},
    'bʱ': {'bʰ'}, 'd̪ʱ': {'dʰ'}, 'ɡʱ': {'ɡʰ'}, 'ɖʱ': {'ɖʰ'},
    't̪ʰ': {'tʰ', 'th'}, 'kʰ': {'kʰ', 'kh'}, 'pʰ': {'pʰ', 'ph'},
    't͡s': {'ts'}, 't͡sʰ': {'tsh'}, 'd͡z': {'dz'}, 'd͡zʱ': set(),
}


def represent(token, vocabulary):
    token = re.sub(r'[ˈˌ]', '', token)
    choices = REPRESENTATIONS.get(token, {token})
    ids = frozenset(vocabulary[t] for t in choices if t in vocabulary and vocabulary[t] > 3)
    if not ids:
        raise ValueError('No declared acoustic representation for ' + repr(token))
    return ids


def units_for_project(phones, ipa, vocabulary, *, any_schwa=False):
    all_vowels = frozenset(vocabulary[t] for t in VOWELS if t in vocabulary)
    units, projections = [], []
    for phone in phones:
        if phone == '.':
            continue
        if phone not in ipa or not ipa[phone]:
            raise ValueError('No lexical IPA for project phone ' + repr(phone))
        unit = all_vowels if any_schwa and phone == 'ax' else represent(ipa[phone], vocabulary)
        units.append(unit)
        projections.append({'project_phone': phone, 'ipa': ipa[phone],
                            'acoustic_tokens': sorted(t for t, i in vocabulary.items() if i in unit),
                            'any_vowel_site_presence_only': any_schwa and phone == 'ax'})
    return units, projections


def path_score(log_probs, units, *, trace=False):
    """CTC recurrence over explicit label sets, with optional maximum-path spans.

    Overlapping acoustic sets are a representation relaxation. Margins are
    uncalibrated evidence and must be checked against recorded native variation.
    """
    values = validate_log_probs(log_probs, 0)
    if not units or any(not u or 0 in u or min(u) < 0 or max(u) >= values.shape[1] for u in units):
        raise ValueError('Nonempty in-vocabulary nonblank unit sets required')
    states = [None]
    for unit in units:
        states.extend([frozenset(unit), None])
    emissions = np.column_stack([values[:, 0] if u is None else np.logaddexp.reduce(values[:, sorted(u)], axis=1) for u in states])
    skip = np.array([i > 1 and states[i] is not None and states[i] != states[i-2] for i in range(len(states))])
    alpha = np.full(len(states), -np.inf)
    alpha[:2] = emissions[0, :2]
    best = alpha.copy()
    pointers = np.zeros((len(values), len(states)), np.int8) if trace else None
    for frame in range(1, len(values)):
        one = np.r_[-np.inf, alpha[:-1]]
        two = np.r_[-np.inf, -np.inf, alpha[:-2]]
        two[~skip] = -np.inf
        alpha = np.logaddexp(np.logaddexp(alpha, one), two) + emissions[frame]
        if trace:
            choices = np.stack([best, np.r_[-np.inf, best[:-1]], np.where(skip, np.r_[-np.inf, -np.inf, best[:-2]], -np.inf)])
            pointers[frame] = choices.argmax(0)
            best = choices.max(0) + emissions[frame]
    likelihood = float(np.logaddexp(alpha[-1], alpha[-2]))
    if not trace:
        return likelihood
    if not math.isfinite(likelihood):
        raise ValueError('Insufficient phone frames for hypothesis')
    state = len(states) - 1 if best[-1] >= best[-2] else len(states) - 2
    route = np.empty(len(values), np.int32)
    for frame in range(len(values)-1, -1, -1):
        route[frame] = state
        if frame:
            state -= int(pointers[frame, state])
    spans = []
    for index in range(len(units)):
        frames = np.flatnonzero(route == 2*index + 1)
        if not len(frames):
            raise ValueError('Maximum path skipped a required unit')
        spans.append({'unit_index': index, 'start_frame': int(frames[0]), 'end_frame': int(frames[-1]+1),
                      'frames': len(frames), 'mean_log_emission': float(emissions[frames, 2*index+1].mean())})
    return {'log_score': likelihood, 'viterbi_score': float(max(best[-1], best[-2])), 'spans': spans}
