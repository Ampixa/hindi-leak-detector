"""Exact CTC marginal over explicit sets of allowed labels at each phone.

Unlike the legacy category-state relaxation, the chosen token stays fixed
inside one phone state. Adjacent phones may skip a blank only when their
actual emitted token IDs differ, including when the allowed sets overlap.
This is versioned separately so existing checkpoints retain their features.
"""
from __future__ import annotations

import math

import numpy as np

from hindi_detector.ctc import validate_log_probs


def _without_each(values):
    left = np.r_[-np.inf, np.logaddexp.accumulate(values)[:-1]]
    right = np.r_[np.logaddexp.accumulate(values[::-1])[::-1][1:], -np.inf]
    return np.logaddexp(left, right)


def path_score(log_probs, units, *, trace=False):
    values = validate_log_probs(log_probs, 0)
    labels, positions, blanks = [], [], [0]
    state_labels = [0]
    for unit in units:
        if not unit or 0 in unit or min(unit) < 0 or max(unit) >= values.shape[1]:
            raise ValueError('Phone alternatives must be nonempty in-vocabulary nonblank IDs')
        ids = np.array(sorted(unit), np.int64)
        at = np.arange(len(state_labels), len(state_labels) + len(ids))
        labels.append(ids)
        positions.append(at)
        state_labels.extend(ids.tolist())
        blanks.append(len(state_labels))
        state_labels.append(0)
    if not units:
        value = float(values[:, 0].sum())
        return {'log_score': value, 'viterbi_score': value, 'spans': []} if trace else value
    matches = []
    for i, ids in enumerate(labels):
        lookup = {} if i == 0 else {token: j for j, token in enumerate(labels[i - 1])}
        matches.append(np.array([lookup.get(token, -1) for token in ids], np.int64))
    alpha = np.full(len(state_labels), -np.inf)
    alpha[0] = values[0, 0]
    alpha[positions[0]] = values[0, labels[0]]
    best = alpha.copy()
    pointers = np.zeros((len(values), len(state_labels)), np.int32) if trace else None
    for frame in range(1, len(values)):
        nxt = np.full_like(alpha, -np.inf)
        next_best = np.full_like(alpha, -np.inf) if trace else None
        nxt[0] = alpha[0] + values[frame, 0]
        if trace:
            next_best[0] = best[0] + values[frame, 0]
        for i, (ids, states) in enumerate(zip(labels, positions, strict=True)):
            previous_blank, following_blank = blanks[i], blanks[i + 1]
            skip = np.full(len(ids), -np.inf)
            if i:
                preceding = alpha[positions[i - 1]]
                skip[:] = np.logaddexp.reduce(preceding)
                same = matches[i] >= 0
                skip[same] = _without_each(preceding)[matches[i][same]]
            nxt[states] = np.logaddexp(np.logaddexp(alpha[states], alpha[previous_blank]), skip) + values[frame, ids]
            nxt[following_blank] = np.logaddexp(alpha[following_blank], np.logaddexp.reduce(alpha[states])) + values[frame, 0]
            if trace:
                skip_best = np.full(len(ids), -np.inf)
                skip_parent = np.zeros(len(ids), np.int32)
                if i:
                    prev_states = positions[i - 1]
                    prev_scores = best[prev_states]
                    top = int(np.argmax(prev_scores))
                    second_scores = prev_scores.copy()
                    second_scores[top] = -np.inf
                    second = int(np.argmax(second_scores))
                    index = np.where(matches[i] == top, second, top)
                    skip_best = prev_scores[index].copy()
                    if len(prev_states) == 1:
                        skip_best[matches[i] == top] = -np.inf
                    skip_parent = prev_states[index]
                choices = np.stack([best[states], np.full(len(ids), best[previous_blank]), skip_best])
                choice = choices.argmax(0)
                next_best[states] = choices.max(0) + values[frame, ids]
                parents = np.stack([states, np.full(len(ids), previous_blank), skip_parent])
                pointers[frame, states] = parents[choice, np.arange(len(ids))]
                top_state = states[int(np.argmax(best[states]))]
                parent = following_blank if best[following_blank] >= best[top_state] else top_state
                next_best[following_blank] = best[parent] + values[frame, 0]
                pointers[frame, following_blank] = parent
        alpha = nxt
        if trace:
            best = next_best
    endings = np.r_[positions[-1], blanks[-1]]
    likelihood = float(np.logaddexp.reduce(alpha[endings]))
    if not trace:
        return likelihood
    if not math.isfinite(likelihood):
        raise ValueError('No feasible exact CTC path for the declared phone alternatives')
    state = int(endings[np.argmax(best[endings])])
    maximum = float(best[state])
    route = np.zeros(len(values), np.int32)
    for frame in range(len(values) - 1, -1, -1):
        route[frame] = state
        if frame:
            state = int(pointers[frame, state])
    spans = []
    for i, states in enumerate(positions):
        frames = np.flatnonzero(np.isin(route, states))
        if not len(frames):
            raise ValueError('Exact maximum path omitted a required phone')
        emitted = {state_labels[int(route[frame])] for frame in frames}
        if len(emitted) != 1:
            raise AssertionError('One exact CTC phone changed token identity without a phone transition')
        token = emitted.pop()
        spans.append({'unit_index': i, 'start_frame': int(frames[0]), 'end_frame': int(frames[-1] + 1),
                      'frames': len(frames), 'token_id': token,
                      'mean_log_emission': float(values[frames, token].mean())})
    return {'log_score': likelihood, 'viterbi_score': maximum, 'spans': spans}
