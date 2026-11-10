"""Linear deletion evidence with local features and grouped fitting helpers."""
from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from hindi_detector.anchored_vowels import FEATURE_NAMES as LOCAL_NAMES
from hindi_detector.corrected_synthesis import sha

# Anchor strength, censored fits and gap-duration availability remain diagnostics.
# The head sees local vowel evidence, not word duration or anchor damage.
INDICES = {'local14': list(range(7)) + list(range(12, 19)), 'primary7': list(range(7)), 'secondary7': list(range(12, 19)),
           'gap_mass4': [3, 4, 15, 16], 'no_margin8': [3, 4, 5, 6, 15, 16, 17, 18]}
ALL_NAMES = ['primary_' + n for n in LOCAL_NAMES] + ['secondary_' + n for n in LOCAL_NAMES]


def select(features, mode):
    values = np.asarray(features, np.float64)
    if values.ndim != 2 or values.shape[1] != 24 or not np.isfinite(values).all() or mode not in INDICES:
        raise ValueError('Expected24 finite anchored deletion features and an explicit mode')
    values = values.copy()
    values[:, [5, 17]] = np.clip(values[:, [5, 17]], -10., 0.)
    return values[:, INDICES[mode]]


def weights(rows):
    """Equal phonetic-group/source/class mass, split over same-source families."""
    per_source_class = Counter((row['pair_id'], row['label']) for row in rows)
    group_sources = {}
    for row in rows:
        group_sources.setdefault(row['phone_collision_group'], set()).add(row['pair_id'])
    result = np.array([1 / (len(group_sources[row['phone_collision_group']]) * per_source_class[row['pair_id'], row['label']]) for row in rows], float)
    return result / result.mean()


def cells(rows, folds=3):
    """Hold out both a phonetic group and a conditioning voice per CV cell."""
    assignments = {}
    for row in rows:
        identity = row['pair_id']
        cell = (int(hashlib.sha256(row['phone_collision_group'].encode()).hexdigest()[:12], 16) % folds, row['voice'])
        if identity in assignments and assignments[identity] != cell:
            raise ValueError('One source family crosses grouped cross-validation cells')
        assignments[identity] = cell
    group_ids = np.array([assignments[row['pair_id']][0] for row in rows])
    voices = np.array([row['voice'] for row in rows])
    for fold in range(folds):
        for voice in sorted(set(voices)):
            validation = (group_ids == fold) & (voices == voice)
            training = (group_ids != fold) & (voices != voice)
            if not validation.any():
                continue
            if training.sum() < 12:
                raise ValueError('Grouped CV cell has insufficient disjoint training rows')
            tr_groups = {row['phone_collision_group'] for row, keep in zip(rows, training, strict=True) if keep}
            dv_groups = {row['phone_collision_group'] for row, keep in zip(rows, validation, strict=True) if keep}
            if tr_groups & dv_groups or set(voices[training]) & set(voices[validation]):
                raise AssertionError('CV phonetic group or voice leakage')
            yield training, validation


def fit(features, labels, rows, mode, ridge):
    values = select(features, mode)
    y = np.asarray(labels, float)
    if len(y) != len(values) or len(rows) != len(y) or set(y.tolist()) != {0., 1.}:
        raise ValueError('Both measured physical labels and matching metadata required')
    if not np.array_equal(y, np.array([row['label'] for row in rows])):
        raise ValueError('Explicit fit labels do not match their source/family metadata')
    mean, scale = values.mean(0), np.maximum(values.std(0), .05)
    x = torch.from_numpy(np.clip((values - mean) / scale, -8, 8))
    target, w = torch.from_numpy(y), torch.from_numpy(weights(rows))
    network = nn.Linear(x.shape[1], 1, dtype=torch.float64)
    nn.init.zeros_(network.weight)
    nn.init.zeros_(network.bias)
    optimizer = torch.optim.LBFGS(network.parameters(), max_iter=120, tolerance_grad=1e-7, tolerance_change=1e-10, line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad(set_to_none=True)
        logits = network(x).squeeze(-1)
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none') * w).mean()
        loss = loss + ridge * network.weight.square().sum() / 2
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite local-deletion training objective')
        loss.backward()
        return loss
    optimizer.step(closure)
    state = {'schema': 'anchored_deletion_v71', 'feature_mode': mode, 'feature_names': [ALL_NAMES[i] for i in INDICES[mode]],
             'mean': mean, 'scale': scale, 'weights': network.state_dict(), 'ridge': ridge,
             'cause': 'vowel_deletion', 'actual_hindi_error_accuracy': None}
    return network.eval(), state


def score(network, state, features):
    values = select(features, state['feature_mode'])
    normalized = np.clip((values - state['mean']) / state['scale'], -8, 8)
    with torch.inference_mode():
        return torch.sigmoid(network(torch.from_numpy(normalized)).squeeze(-1)).cpu().numpy()


def load(path: Path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['schema'] != 'anchored_deletion_v71' or state['cause'] != 'vowel_deletion':
        raise ValueError('Not a reviewed anchored deletion checkpoint')
    if state['feature_names'] != [ALL_NAMES[i] for i in INDICES[state['feature_mode']]]:
        raise ValueError('Anchored feature schema changed')
    root = Path(__file__).resolve().parents[1]
    for name, digest in state['code_sha256'].items():
        if sha(root / name) != digest:
            raise ValueError('Checkpoint feature-code pin differs: ' + name)
    for name, pin in state['resource_files'].items():
        if sha(root / name) != pin:
            raise ValueError('Checkpoint resource pin differs: ' + name)
    network = nn.Linear(len(state['feature_names']), 1, dtype=torch.float64)
    network.load_state_dict(state['weights'], strict=True)
    return network.eval(), state
