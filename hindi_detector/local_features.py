"""Word-level timing/pitch/energy features with explicit nuisance normalization."""
from __future__ import annotations

import numpy as np

PROSODY_NAMES = ['duration_relative', 'pause_before_relative', 'pause_after_relative',
                 'voiced_fraction', 'pitch_available', 'pitch_mean_st', 'pitch_range_st',
                 'pitch_start_st', 'pitch_end_st', 'pitch_change_st',
                 'energy_mean_db', 'energy_change_db', 'position_relative']
PROSODY_NAMES += [f'pitch_bin_{i}' for i in range(8)] + [f'energy_bin_{i}' for i in range(8)]


def relative_tracks(f0: np.ndarray, rms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f0, rms = np.asarray(f0, dtype=float), np.asarray(rms, dtype=float)
    if f0.ndim != 1 or f0.shape != rms.shape or not len(f0):
        raise ValueError('Matching nonempty pitch/energy tracks required')
    if not np.isfinite(f0).all() or not np.isfinite(rms).all() or (f0 < 0).any() or (rms < 0).any():
        raise ValueError('Pitch/energy must be finite nonnegative measurements')
    voiced = f0 > 0
    pitch = np.full(len(f0), np.nan)
    if voiced.any():
        log_pitch = np.log2(f0[voiced])
        pitch[voiced] = 12 * (log_pitch - np.median(log_pitch))
    if not np.any(rms > 0):
        raise ValueError('Silent energy cannot establish prosody')
    active = rms > np.percentile(rms, 95) * .05
    reference = np.median(rms[active])
    if reference <= 0:
        raise ValueError('No active energy reference')
    energy = 20 * np.log10(np.maximum(rms / reference, 1e-6))
    return pitch, energy


def word_prosody(times: np.ndarray, f0: np.ndarray, rms: np.ndarray, spans: list[dict]) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) != len(f0) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Strictly increasing matching frame times required')
    if not spans:
        raise ValueError('Need aligned words')
    pitch, energy = relative_tracks(f0, rms)
    previous_end = -float('inf')
    for span in spans:
        start, end = span['start'], span['end']
        if not np.isfinite([start, end]).all() or start < 0 or end <= start or start < previous_end - 1e-6:
            raise ValueError('Invalid or overlapping word spans')
        previous_end = end
    duration_ref = np.median([s['end'] - s['start'] for s in spans])
    vectors = []
    for index, span in enumerate(spans):
        start, end = span['start'], span['end']
        inside = (times >= start) & (times <= end)
        if not inside.any():
            raise ValueError('No prosody frame inside a word; alignment needs inspection')
        voiced = inside & np.isfinite(pitch)
        available = np.sum(voiced) >= 2
        query = np.linspace(start, end, 8)
        pitch_bins = np.interp(query, times[voiced], pitch[voiced]) if available else np.zeros(8)
        energy_bins = np.interp(query, times, energy)
        before = max(0., start - spans[index - 1]['end']) / duration_ref if index else 0.
        after = max(0., spans[index + 1]['start'] - end) / duration_ref if index + 1 < len(spans) else 0.
        vectors.append([(end - start) / duration_ref, before, after, float(np.sum(voiced) / np.sum(inside)), float(available),
                        float(np.mean(pitch[voiced])) if available else 0.,
                        float(np.percentile(pitch[voiced], 90) - np.percentile(pitch[voiced], 10)) if available else 0.,
                        float(pitch_bins[0]), float(pitch_bins[-1]), float(pitch_bins[-1] - pitch_bins[0]),
                        float(np.mean(energy[inside])), float(energy_bins[-1] - energy_bins[0]),
                        index / max(1, len(spans) - 1), *pitch_bins.tolist(), *energy_bins.tolist()])
    result = np.asarray(vectors, dtype=np.float32)
    if result.shape != (len(spans), len(PROSODY_NAMES)) or not np.isfinite(result).all():
        raise ValueError('Nonfinite/inconsistent word prosody features')
    return result
