"""Localized vowel-region supervision; intervention geometry is target-only."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

SCHEMA = 'nucleus_deletion_v82'


def supervision(row, samples):
    """Known manipulation coordinates create LOSS targets, never model inputs."""
    start, end = row['production_word']['start'], row['production_word']['end']
    left = max(0, int(np.floor((start - .04) * 16000)))
    right = min(samples, int(np.ceil((end + .04) * 16000)))
    raw_frames = (right - left) // 80 + 1  # Verified centered512 FFT, hop80.
    at = np.clip((np.arange(128) + .5) * raw_frames / 128 - .5, 0, raw_frames - 1)
    times = left / 16000 + at * 80 / 16000
    half = np.maximum(np.gradient(times) / 2, .0025)
    a, b = row['source_region']['start_sample'], row['source_region']['end_sample']
    edit = row['edit']
    if row['family'] in {'P1', 'P1w', 'S2', 'S0', 'S_world'}:
        if (edit['start_sample'], edit['end_sample']) != (a, b):
            raise ValueError('Physical edit and source nucleus coordinates disagree')
    if row['family'] in {'S0', 'S_world'} and edit['replacement_samples'] != b - a:
        raise ValueError('Retained control unexpectedly changes nucleus duration')
    if row['family'] in {'P1', 'P1w'}:
        if edit['replacement_samples'] != 0:
            raise ValueError('Deletion target has retained replacement samples')
        lo, hi, value = edit['start_sample'] / 16000 - .02, edit['start_sample'] / 16000 + .02, 0.
    else:
        value = 1.
        if row['family'] == 'S2':
            a, b = edit['start_sample'], edit['start_sample'] + edit['replacement_samples']
        elif row['family'] == 'S1':
            if edit['end_sample'] <= a:
                shift = edit['replacement_samples'] - (edit['end_sample'] - edit['start_sample'])
                a, b = a + shift, b + shift
            elif edit['start_sample'] < b:
                raise ValueError('Off-target control overlaps the target nucleus')
        elif row['family'] not in {'S_original', 'S0', 'S_world'}:
            raise ValueError('Unregistered auxiliary target family')
        lo, hi = a / 16000, b / 16000
        trim = min(.0125, (hi - lo) / 4)
        lo, hi = lo + trim, hi - trim
    if not 0 <= lo < hi <= samples / 16000:
        raise ValueError('Known physical target outside emitted waveform')
    mask = np.maximum(0., np.minimum(times + half, hi) - np.maximum(times - half, lo)) / (2 * half)
    if mask.sum() <= 0:
        raise ValueError('Auxiliary target has no overlap with observed production word crop')
    return np.full(128, value, np.float32), mask.astype(np.float32)


class PhoneContextNetwork(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.context = nn.Embedding(vocabulary_size, 4)
        self.head = nn.Sequential(nn.Linear(22, 16), nn.SiLU(), nn.Dropout(.1), nn.Linear(16, 1))

    def forward(self, mel, context, phone):
        return self.head(torch.cat([phone, self.context(context).flatten(1)], dim=1)).squeeze(1)


class NucleusDeletionNetwork(PhoneContextNetwork):
    def __init__(self, vocabulary_size):
        super().__init__(vocabulary_size)
        self.temporal = nn.Sequential(nn.Conv1d(64, 24, 5, padding=2), nn.GroupNorm(6, 24), nn.SiLU(),
            nn.Conv1d(24, 24, 3, padding=1), nn.GroupNorm(6, 24), nn.SiLU(),
            nn.Conv1d(24, 24, 3, dilation=2, padding=2), nn.GroupNorm(6, 24), nn.SiLU())
        self.nucleus = nn.Sequential(nn.Conv1d(24, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU(), nn.Conv1d(16, 1, 1))
        self.vowel_veto = nn.Parameter(torch.tensor(0.))

    def forward(self, mel, context, phone, *, auxiliary=False):
        embedded = self.context(context).flatten(1)
        values = self.temporal(mel[:, :64])
        dense = self.nucleus(values).squeeze(1)
        # Only the observed deletion-anchor neighborhood can supply the acoustic
        # veto. No global acoustic pool and no known edit-location input.
        focus = mel[:, 64]
        focus = torch.where(focus >= .25, focus, torch.zeros_like(focus))
        if torch.any(focus.sum(1) <= 0):
            raise ValueError('No observed local focus support')
        # Mechanistic arm: the word-label loss cannot train the vowel head to
        # become a second deletion classifier. Only local auxiliary labels do.
        local = (dense.detach() + torch.log(focus.clamp_min(1e-30))).masked_fill(focus == 0, -torch.inf)
        present = torch.logsumexp(local, dim=1) - torch.log(focus.sum(1))
        logits = self.head(torch.cat([phone, embedded], dim=1)).squeeze(1) - F.softplus(self.vowel_veto) * present
        return (logits, dense) if auxiliary else logits
