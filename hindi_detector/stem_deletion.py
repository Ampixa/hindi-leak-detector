"""Five-millisecond frozen XLSR frontend; local vowel presence veto."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from hindi_detector.nucleus_deletion import PhoneContextNetwork

SCHEMA = 'stem_deletion_v83'


def query_grid(bounds, samples):
    start, end = bounds
    left = max(0, int(np.floor((start - .04) * 16000)))
    right = min(samples, int(np.ceil((end + .04) * 16000)))
    count = (right - left) // 80 + 1
    return left + np.clip((np.arange(128) + .5) * count / 128 - .5, 0, count - 1) * 80


def interpolate_stem(features, bounds, samples):
    """Sample actual conv centers onto the immutable mel/target time grid."""
    if features.ndim != 2 or features.shape[0] != 512:
        raise ValueError('Need verified512-channel fifth convolution output')
    # Kernels10,3,3,3,3 and strides5,2,2,2,2: RF160, stride80.
    expected = (samples - 160) // 80 + 1
    if features.shape[1] != expected:
        raise ValueError('Actual early frontend clock differs from pinned convolution recipe')
    at = (query_grid(bounds, samples) - 79.5) / 80
    # Only waveform-edge padding may clamp: the first valid convolution center
    # is4.96875ms into the utterance. No oracle/edit/source location is accepted.
    at = np.clip(at, 0, features.shape[1] - 1)
    lo = np.floor(at).astype(np.int64)
    hi = np.minimum(lo + 1, features.shape[1] - 1)
    fraction = (at - lo).astype(np.float32)
    return features[:, lo] * (1 - fraction) + features[:, hi] * fraction


class StemVowelNetwork(PhoneContextNetwork):
    def __init__(self, vocabulary_size):
        super().__init__(vocabulary_size)
        self.temporal = nn.Sequential(nn.Conv1d(512, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv1d(32, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.SiLU())
        self.nucleus = nn.Sequential(nn.Conv1d(32, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU(), nn.Conv1d(16, 1, 1))
        self.vowel_veto = nn.Parameter(torch.tensor(0.))

    def forward(self, mel, context, phone, *, auxiliary=False):
        # Name retained for the generic fitting adapter; these512channels are
        # pretrained convolution features, not mel bands. Hints excluded.
        dense = self.nucleus(self.temporal(mel[:, :512])).squeeze(1)
        focus = mel[:, 512]
        focus = torch.where(focus >= .25, focus, torch.zeros_like(focus))
        if torch.any(focus.sum(1) <= 0):
            raise ValueError('No observed acoustic focus support')
        values = (dense.detach() + torch.log(focus.clamp_min(1e-30))).masked_fill(focus == 0, -torch.inf)
        present = torch.logsumexp(values, dim=1) - torch.log(focus.sum(1))
        logit = super().forward(mel, context, phone) - F.softplus(self.vowel_veto) * present
        return (logit, dense) if auxiliary else logit
