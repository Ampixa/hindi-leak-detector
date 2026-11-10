"""Voice-controlled pitch interventions and measurements of emitted waveforms."""
from __future__ import annotations

import numpy as np

from hindi_detector.local_features import word_prosody
from hindi_detector.structured_detector import context


def transfer_pitch(times, carrier_f0, donor_tracks, carrier_span, donor_span, *, knots=20, edge_seconds=.03):
    times = np.asarray(times, dtype=float)
    original = np.asarray(carrier_f0, dtype=float)
    if (times.ndim != 1 or original.shape != times.shape or len(times) < 2
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or not np.isfinite(original).all() or np.any(original < 0)):
        raise ValueError('Finite increasing carrier times and nonnegative matching F0 required')
    if knots != 20 or edge_seconds != .03:
        raise ValueError('Frozen transfer uses20 knots and30ms edge blending')
    start, end = float(carrier_span['start']), float(carrier_span['end'])
    ds, de = float(donor_span['start']), float(donor_span['end'])
    if not np.isfinite([start, end, ds, de]).all() or not 0 <= start < end <= times[-1] or not 0 <= ds < de:
        raise ValueError('Valid measured donor/carrier word spans required')
    dt = np.asarray(donor_tracks['times'], dtype=float)
    dp = np.asarray([np.nan if v is None else v for v in donor_tracks['pitch_relative_st']], dtype=float)
    if (dt.ndim != 1 or dt.shape != dp.shape or len(dt) < 2 or not np.isfinite(dt).all()
            or np.any(np.diff(dt) <= 0) or np.isinf(dp).any() or de > dt[-1]):
        raise ValueError('Donor must supply verified utterance-relative measured F0 track')
    available = (dt >= ds) & (dt <= de) & np.isfinite(dp)
    if np.sum(available) < 3:
        raise ValueError('Fewer than three voiced donor word frames')
    knots_pitch = np.interp(np.linspace(ds, de, knots), dt[available], dp[available])
    voiced = original > 0
    inside = (times >= start) & (times <= end) & voiced
    if np.sum(inside) < 3:
        raise ValueError('Fewer than three voiced carrier word frames')
    median_log = float(np.median(np.log2(original[voiced])))
    desired_log = median_log + np.interp((times[inside] - start) / (end - start),
                                       np.linspace(0, 1, knots), knots_pitch) / 12
    distance = np.minimum(times[inside] - start, end - times[inside])
    weight = .5 - .5 * np.cos(np.pi * np.clip(distance / edge_seconds, 0, 1))
    output = original.copy()
    output[inside] = np.exp2((1 - weight) * np.log2(original[inside]) + weight * desired_log)
    if np.any((output[inside] < 50) | (output[inside] > 650)):
        raise ValueError('Transferred pitch leaves frozen50..650Hz analysis range')
    if not np.array_equal(output > 0, voiced) or not np.array_equal(output[~inside], original[~inside]):
        raise AssertionError('Intervention changed the carrier voicing mask or another word')
    return output, {'voiced_word_frames': int(np.sum(inside)), 'knots': knots,
                    'intended_median_absolute_change_st': float(np.median(np.abs(12 * np.log2(output[inside] / original[inside]))))}


def measure_waveform(samples, spans, world):
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 1 or len(samples) < 3200 or not np.isfinite(samples).all() or not np.any(samples):
        raise ValueError('Finite non-silent mono16k waveform required')
    f0, times = world.dio(samples, 16000, f0_floor=50., f0_ceil=650., frame_period=5.)
    f0 = world.stonemask(samples, f0, times, 16000)
    smooth = np.sqrt(np.maximum(np.convolve(samples ** 2, np.ones(320) / 320, mode='same'), 0))
    rms = np.interp(times, np.arange(len(samples)) / 16000, smooth)
    values = word_prosody(times, f0, rms, spans)
    return values, f0, times


def pair_prosody(query, reference, ordinal):
    query, reference = np.asarray(query), np.asarray(reference)
    if (query.ndim != 2 or query.shape != reference.shape or query.shape[1] != 29
            or not 0 <= ordinal < len(query) or not np.isfinite(query).all() or not np.isfinite(reference).all()):
        raise ValueError('Identical same-text measured29-column prosody matrices required')
    delta = context(query)[ordinal] - context(reference)[ordinal]
    return np.concatenate([delta, np.abs(delta)]).astype(np.float32)
