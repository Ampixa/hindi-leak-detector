"""Monotone duration intervention with fixed carrier spectra, pitch and pauses."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


def duration_map(carrier_words, donor_words, seconds, *, minimum_word_seconds=.08):
    if minimum_word_seconds not in {.08, .03}:
        raise ValueError('Only the original80ms or explicitly revised30ms protocol is supported')
    if [w['word'] for w in carrier_words] != [w['word'] for w in donor_words] or len(carrier_words) < 3:
        raise ValueError('Identical canonical word sequence required for duration transfer')
    c = np.array([[w['start'], w['end']] for w in carrier_words], float)
    d = np.array([[w['start'], w['end']] for w in donor_words], float)
    for spans in (c, d):
        if (not np.isfinite(spans).all() or np.any(spans[:, 0] < 0) or np.any(spans[:, 1] <= spans[:, 0])
                or np.any(spans[1:, 0] < spans[:-1, 1] - 1e-8)):
            raise ValueError('Measured nonoverlapping positive word spans required')
    if not np.isfinite(seconds) or seconds < c[-1, 1] - 1e-7:
        raise ValueError('Aligned word extends beyond full carrier duration')
    original = c[:, 1] - c[:, 0]
    desired = (d[:, 1] - d[:, 0]) * original.sum() / (d[:, 1] - d[:, 0]).sum()
    ratio = desired / original
    if np.any((ratio < .4) | (ratio > 2.5)) or np.any(desired < minimum_word_seconds):
        raise ValueError('Local word stretch outside frozen.4..2.5 or word below explicit minimum')
    target = c.copy()
    for i in range(len(c)):
        target[i, 0] = c[0, 0] if i == 0 else target[i - 1, 1] + c[i, 0] - c[i - 1, 1]
        target[i, 1] = target[i, 0] + desired[i]
    if abs(target[-1, 1] - c[-1, 1]) > 1e-7:
        raise AssertionError('Relative word redistribution changed total utterance duration')
    source_knots, target_knots = [0.], [0.]
    for old, new in zip(c.reshape(-1), target.reshape(-1), strict=True):
        if abs(old - source_knots[-1]) < 1e-8:
            if abs(new - target_knots[-1]) > 1e-7:
                raise ValueError('Zero carrier gap acquired synthetic duration')
            continue
        source_knots.append(float(old))
        target_knots.append(float(new))
    if seconds > source_knots[-1] + 1e-7:
        source_knots.append(float(seconds))
        target_knots.append(float(seconds))
    if np.any(np.diff(source_knots) <= 0) or np.any(np.diff(target_knots) <= 0):
        raise ValueError('Duration intervention must preserve strictly ordered sound positions')
    return np.array(source_knots), np.array(target_knots), target, ratio


def retime_parameters(times, f0, spectrum, aperiodicity, source_knots, target_knots):
    times, f0, spectrum, aperiodicity = [np.asarray(v, dtype=float) for v in (times, f0, spectrum, aperiodicity)]
    if (times.ndim != 1 or f0.shape != times.shape or spectrum.ndim != 2 or spectrum.shape != aperiodicity.shape
            or len(spectrum) != len(times) or any(not np.isfinite(v).all() for v in (times, f0, spectrum, aperiodicity))
            or np.any(np.diff(times) <= 0) or np.any(f0 < 0)):
        raise ValueError('Finite matching measured WORLD parameters required')
    source_time = np.interp(times, target_knots, source_knots)
    # Nearest-frame F0 preserves the carrier voiced/unvoiced values rather than
    # inventing intermediate pitches at voicing transitions. Spectral envelopes
    # interpolate in time; no donor spectrum, aperiodicity or pitch is imported.
    nearest = np.clip(np.searchsorted(times, source_time), 1, len(times) - 1)
    nearest -= (np.abs(times[nearest - 1] - source_time) < np.abs(times[nearest] - source_time))
    pitch = f0[nearest].copy()
    spectral = interp1d(times, spectrum, axis=0, bounds_error=True)(source_time)
    noise = interp1d(times, aperiodicity, axis=0, bounds_error=True)(source_time)
    return np.ascontiguousarray(pitch), np.ascontiguousarray(spectral), np.ascontiguousarray(noise)
