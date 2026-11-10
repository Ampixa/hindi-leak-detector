"""Conservative waveform correspondence checks, never pronunciation labels."""
from __future__ import annotations

import numpy as np


def comparison(source: np.ndarray, source_rate: int, staged: np.ndarray, staged_rate: int) -> dict:
    if any(type(rate) is not int or rate <= 0 for rate in (source_rate, staged_rate)):
        raise ValueError('Sample rates must be positive integer Hz')
    if source.ndim != 1 or staged.ndim != 1 or not np.isfinite(source).all() or not np.isfinite(staged).all():
        raise ValueError('Need finite mono source and staged waveforms')
    original_frames = len(source)
    if source_rate != staged_rate:
        from scipy.signal import resample_poly

        divisor = int(np.gcd(source_rate, staged_rate))
        source = resample_poly(source, staged_rate // divisor, source_rate // divisor)
    n = min(len(source), len(staged))
    if n < 1 or np.dot(source[:n], source[:n]) <= 0:
        raise ValueError('Silent/empty original waveform')
    a, b = source[:n].astype(float), staged[:n].astype(float)
    gain = float(np.dot(a, b) / np.dot(a, a))
    residual = b - a * gain
    std = float(np.std(a) * np.std(b))
    correlation = float(np.mean((a - a.mean()) * (b - b.mean())) / std) if std else None
    equivalent = source_rate == staged_rate and len(source) == len(staged) and float(np.max(np.abs(residual))) <= 1e-4
    return {'source_sample_rate': source_rate, 'source_frames': original_frames, 'staged_sample_rate': staged_rate,
            'staged_frames': len(staged), 'comparison_resampling': 'scipy.signal.resample_poly' if source_rate != staged_rate else None,
            'comparison_frames': n, 'resampled_frame_count_difference': len(staged) - len(source),
            'comparison_shift_samples': 0, 'least_squares_gain': gain,
            'pearson_correlation': correlation, 'gain_adjusted_rmse': float(np.sqrt(np.mean(residual ** 2))),
            'gain_adjusted_max_error': float(np.max(np.abs(residual))),
            'same_rate_length_and_gain_adjusted_samples_within_1e_minus4': equivalent,
            'meaning': 'Waveform correspondence under an explicit gain/resampling comparison. This is not an accent, speaker-identity or native-acceptability label.'}

