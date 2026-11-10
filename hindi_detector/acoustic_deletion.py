"""Versioned query-only high-resolution word acoustics for deletion learning."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio

from hindi_detector.anchored_detector import select

SCHEMA = 'acoustic_deletion_v81'
RECIPE = {'sample_rate': 16000, 'n_fft': 512, 'window_samples': 400, 'hop_samples': 80,
          'mel_bands': 64, 'f_min': 80., 'f_max': 7600., 'time_bins': 128,
          'word_padding_seconds': .04, 'focus_sigma_fraction': .08,
          'normalization': 'Per-query-word RMS, then global log-mel mean/std; no dataset or reference normalization.',
          'spectral_floor': 'Relative to query crop maximum power,80dB down.',
          'duration_caveat': 'No duration scalar enters the acoustic head; fixed padding/resampling still encode duration indirectly. Compare explicit duration baseline.',
          'clock': 'Actual duration / verified convolution output frame count, matching pinned phone runtime.',
          'forbidden_inputs': ['edit geometry', 'mapped_word', 'source_region', 'reference waveform',
                               'source/speaker/word ID', 'physical label', 'absolute duration scalar']}
PROCESSING = {'S_original': 'original_pcm', 'S0': 'original_pcm_splice', 'P1': 'original_pcm_splice',
              'S1': 'original_pcm_splice', 'S_world': 'WORLD_base', 'P1w': 'WORLD_base', 'S2': 'WORLD_base'}


def require_query_evidence(value):
    if isinstance(value, dict):
        forbidden = {'edit', 'mapped_word', 'source_region', 'source_features', 'oracle', 'reference',
                     'reference_audio', 'label', 'word_id', 'voice', 'speaker_id'}
        if forbidden & value.keys():
            raise ValueError('Non-query provenance/label entered acoustic evidence')
        for child in value.values():
            require_query_evidence(child)
    elif isinstance(value, list):
        for child in value:
            require_query_evidence(child)


def frame_count(samples, configuration):
    if configuration['model_type'] != 'wav2vec2' or configuration.get('add_adapter'):
        raise ValueError('Unsupported phone output geometry')
    kernels, strides = configuration['conv_kernel'], configuration['conv_stride']
    if not kernels or len(kernels) != len(strides):
        raise ValueError('Incomplete convolution geometry')
    value = int(samples)
    for kernel, stride in zip(kernels, strides, strict=True):
        if not isinstance(kernel, int) or not isinstance(stride, int) or kernel < 1 or stride < 1:
            raise ValueError('Invalid convolution geometry')
        value = (value - kernel) // stride + 1
    if value < 1:
        raise ValueError('Too few samples for phone frontend')
    return value


class WordAcoustics:
    def __init__(self, phone_vocabulary, phone_configurations):
        if len(phone_configurations) != 2 or len(set(phone_vocabulary.values())) != len(phone_vocabulary):
            raise ValueError('Need explicit two-probe geometry and unique project phone IDs')
        self.vocabulary = phone_vocabulary
        self.configurations = phone_configurations
        self.window = torch.hann_window(RECIPE['window_samples'])
        self.filters = torchaudio.functional.melscale_fbanks(
            n_freqs=RECIPE['n_fft'] // 2 + 1, f_min=RECIPE['f_min'], f_max=RECIPE['f_max'],
            n_mels=RECIPE['mel_bands'], sample_rate=16000, norm='slaney', mel_scale='htk')

    def extract(self, samples, word_bounds, evidence, native_phones, site_index, local_features):
        # Deliberately accept individual runtime-observable arguments instead of
        # a training row containing source labels and intervention coordinates.
        wave = np.asarray(samples, np.float32)
        if wave.ndim != 1 or not np.isfinite(wave).all() or len(wave) < 3200 or np.max(np.abs(wave)) < 1e-5:
            raise ValueError('Need finite non-silent16k mono utterance')
        start, end = word_bounds
        if not 0 <= start < end <= len(wave) / 16000 or len(evidence) != 2:
            raise ValueError('Invalid production word bounds or probe evidence')
        require_query_evidence(evidence)
        if not 0 < site_index < len(native_phones) - 1 or native_phones[site_index] != 'ax':
            raise ValueError('Only internal inherent-vowel deletion sites are supported')
        neighbors = [native_phones[site_index - 1], native_phones[site_index + 1]]
        if any(p not in self.vocabulary for p in neighbors):
            raise ValueError('Neighbor project phone missing from pinned input vocabulary')
        left = max(0, int(math.floor((start - .04) * 16000)))
        right = min(len(wave), int(math.ceil((end + .04) * 16000)))
        crop = torch.from_numpy(wave[left:right].copy())
        if len(crop) <= RECIPE['n_fft'] // 2:
            raise ValueError('Observed word crop too short for spectral analysis')
        with torch.inference_mode():
            crop = crop / crop.square().mean().sqrt().clamp_min(1e-6)
            spectrum = torch.stft(crop, n_fft=512, hop_length=80, win_length=400,
                window=self.window, center=True, return_complex=True).abs().square()
            power = self.filters.T @ spectrum
            mel = torch.log(torch.maximum(power, power.max().clamp_min(1e-12) * 1e-8))
            mel = (mel - mel.mean()) / mel.std(unbiased=False).clamp_min(.1)
            mel = F.interpolate(mel[None], size=128, mode='linear', align_corners=False)[0].clamp(-8, 8)
        gaps, nuclei, fallbacks = [], [], 0
        fallback = start + (end - start) * (site_index + .5) / len(native_phones)
        for config, probe in zip(self.configurations, evidence, strict=True):
            dt = len(wave) / 16000 / frame_count(len(wave), config)
            if len(probe['windows']) != 3:
                raise ValueError('Need all three production boundary recipes')
            for window in probe['windows']:
                gaps.append((window['gap_start_frame'] + window['gap_end_frame']) / 2 * dt)
                span = window['native_focal_span']
                fallbacks += span is None
                nuclei.append(fallback if span is None else
                    (window['start_frame'] + (span['start_frame'] + span['end_frame']) / 2) * dt)
        grid = np.linspace(0, 1, 128, dtype=np.float32)
        masks = []
        for centers in (gaps, nuclei):
            seconds = float(np.mean(centers))
            if not start <= seconds <= end:
                raise ValueError('Observed phone-location hint outside production word bounds')
            center = (seconds - left / 16000) / ((right - left) / 16000)
            masks.append(np.exp(-.5 * ((grid - center) / .08) ** 2))
        acoustic = np.concatenate([mel.numpy(), np.stack(masks)], axis=0).astype(np.float32)
        if acoustic.shape != (66, 128) or not np.isfinite(acoustic).all():
            raise ValueError('Invalid high-resolution spectral features')
        diagnostic = {'native_hint_fallback_windows': int(fallbacks), 'hint_windows': 6,
                      'observed_word_seconds': end - start, 'observed_crop_seconds': (right - left) / 16000}
        return acoustic, np.array([self.vocabulary[p] for p in neighbors], np.int64), select(np.asarray(local_features)[None], 'local14')[0].astype(np.float32), diagnostic


class AcousticDeletionNetwork(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.context = nn.Embedding(vocabulary_size, 4)
        self.temporal = nn.Sequential(
            nn.Conv1d(66, 24, 5, padding=2), nn.GroupNorm(6, 24), nn.SiLU(),
            nn.Conv1d(24, 24, 3, padding=1), nn.GroupNorm(6, 24), nn.SiLU(),
            nn.Conv1d(24, 24, 3, padding=2, dilation=2), nn.GroupNorm(6, 24), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(94, 24), nn.SiLU(), nn.Dropout(.1), nn.Linear(24, 1))

    def forward(self, mel, context, phone):
        if mel.ndim != 3 or mel.shape[1:] != (66, 128) or context.shape != (len(mel), 2) or phone.shape != (len(mel), 14):
            raise ValueError('Acoustic head input schema differs')
        values = self.temporal(mel)
        pooled = [values.mean(-1)]
        for mask in (mel[:, 64:65], mel[:, 65:66]):
            pooled.append((values * mask).sum(-1) / mask.sum(-1).clamp_min(1e-6))
        return self.head(torch.cat([*pooled, self.context(context).flatten(1), phone], dim=1)).squeeze(1)


def pair_indices(rows):
    lookup = {}
    for i, row in enumerate(rows):
        key = row['pair_id'], row['family']
        if key in lookup:
            raise ValueError('Duplicate source/family in paired acoustic objective')
        lookup[key] = i
    pairs = []
    for identity in sorted({r['pair_id'] for r in rows}):
        for positive, negative in [('P1', 'S0'), ('P1w', 'S_world'), ('P1w', 'S2')]:
            if (identity, positive) in lookup and (identity, negative) in lookup:
                a, b = lookup[identity, positive], lookup[identity, negative]
                if rows[a]['label'] != 1 or rows[b]['label'] != 0:
                    raise ValueError('Physical family/label disagreement in ranking objective')
                if PROCESSING[positive] != PROCESSING[negative]:
                    raise ValueError('Rank pair does not share its original/WORLD processing base')
                pairs.append((a, b))
    return pairs
