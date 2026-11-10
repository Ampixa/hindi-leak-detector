"""Reference-free physical deletion head over explicitly query-only columns."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from hindi_detector.corrected_synthesis import sha
from hindi_detector.intrinsic_vowels import FEATURE_NAMES as THIRD_NAMES
from hindi_detector.paired_local_vowels import FEATURE_NAMES as PAIRED_NAMES

QUERY_INDICES = [i for i, name in enumerate(PAIRED_NAMES)
                 if name.startswith('query_') or name in {'positive_margin_fraction', 'is_vowel_deletion', 'is_vowel_quality'}]
QUERY_NAMES = [prefix + PAIRED_NAMES[i] for prefix in ('primary_', 'secondary_') for i in QUERY_INDICES]
MODES = {'dual46': QUERY_NAMES, 'third33': ['independent_' + name for name in THIRD_NAMES] + ['independent_available'],
         'combined79': QUERY_NAMES + ['independent_' + name for name in THIRD_NAMES] + ['independent_available']}
if len(QUERY_INDICES) != 23 or any('reference' in name or 'delta_' in name for name in QUERY_NAMES):
    raise ValueError('Reference-free feature schema unexpectedly contains reference inputs')


def inputs(paired, third, available, mode):
    x, other, valid = np.asarray(paired, np.float32), np.asarray(third, np.float32), np.asarray(available, bool)
    if x.ndim != 2 or x.shape[1] != 114 or other.shape != (len(x), 32) or valid.shape != (len(x),):
        raise ValueError('Query-only input shapes differ from explicit measured schemas')
    if not np.isfinite(x).all() or not np.isfinite(other).all():
        raise ValueError('Query-only input values must be finite')
    query = x[:, QUERY_INDICES + [i + 57 for i in QUERY_INDICES]]
    independent = np.column_stack([np.where(valid[:, None], other, 0), valid.astype(np.float32)])
    if mode == 'dual46':
        return query
    if mode == 'third33':
        return independent
    if mode == 'combined79':
        return np.column_stack([query, independent])
    raise ValueError('Unknown query-only feature mode')


class QueryDeletion(nn.Module):
    def __init__(self, mode, architecture):
        super().__init__()
        if mode not in MODES:
            raise ValueError('Unknown query-only feature mode')
        width = len(MODES[mode])
        if architecture == 'linear':
            self.head = nn.Linear(width, 1)
        elif architecture == 'mlp8':
            self.head = nn.Sequential(nn.Linear(width, 8), nn.Tanh(), nn.Linear(8, 1))
        else:
            raise ValueError('Unknown query-only deletion architecture')

    def forward(self, values):
        return self.head(values).squeeze(-1)


def score(network, state, paired, third, available):
    values = inputs(paired, third, available, state['feature_mode'])
    normalized = np.clip((values - state['mean']) / state['scale'], -8, 8).astype(np.float32)
    with torch.inference_mode():
        return torch.sigmoid(network(torch.from_numpy(normalized))).cpu().numpy()


def load(path: Path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['schema'] != 'query_full_deletion_v66' or state['feature_names'] != MODES[state['feature_mode']]:
        raise ValueError('Query-only deletion checkpoint schema changed')
    root = Path(__file__).resolve().parents[1]
    for name, digest in state['code_sha256'].items():
        if sha(root / name) != digest:
            raise ValueError('Query-only checkpoint code changed: ' + name)
    network = QueryDeletion(state['feature_mode'], state['architecture'])
    network.load_state_dict(state['weights'], strict=True)
    return network.eval(), state
