"""A voice/source-free head over explicit competing vowel evidence."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from hindi_detector.pronunciation_sites import FEATURE_NAMES


class VowelSiteNetwork(nn.Module):
    def __init__(self, architecture='linear'):
        super().__init__()
        if architecture == 'linear':
            self.head = nn.Linear(len(FEATURE_NAMES), 1)
        elif architecture == 'mlp16':
            self.head = nn.Sequential(nn.Linear(len(FEATURE_NAMES), 16), nn.Tanh(), nn.Linear(16, 1))
        else:
            raise ValueError('Unknown site network architecture')

    def forward(self, values):
        return self.head(values).squeeze(-1)


def score(network, state, features):
    values = np.asarray(features, np.float32)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES) or not np.isfinite(values).all():
        raise ValueError('Wrong measured vowel feature geometry')
    normalized = np.clip((values-state['mean'])/state['scale'], -8, 8).astype(np.float32)
    with torch.inference_mode():
        return torch.sigmoid(network(torch.from_numpy(normalized))).cpu().numpy()


def direction_supported(features):
    values = np.asarray(features)
    # A Hindi-compatible alternative must win consistently, while the actual
    # reference supports the expected category. A score alone is insufficient.
    return ((values[:, 1] > 0) & (values[:, 8] > 0) & (values[:, 14] == 1)
            & (values[:, 7] < 0) & (values[:, 13] > -1.5))


def decisions(state, features, probabilities):
    supported = direction_supported(features)
    output = []
    for row, value, available in zip(features, probabilities, supported, strict=True):
        kind = 'vowel_deletion' if row[-2] else 'vowel_quality'
        points = state['operating_points'][kind]
        output.append('strong_vowel_change' if available and value >= points['high'] else
                      'no_strong_vowel_change' if value <= points['low'] else 'uncertain')
    return output


def load(path: Path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['schema'] != 'paired_vowel_site_v19' or state['feature_names'] != FEATURE_NAMES:
        raise ValueError('Wrong vowel site model schema')
    network = VowelSiteNetwork(state['architecture'])
    network.load_state_dict(state['weights'], strict=True)
    return network.eval(), state
