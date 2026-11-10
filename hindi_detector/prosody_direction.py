"""Same-word pitch direction against independent Native and Hindi references."""
from __future__ import annotations

import numpy as np

PITCH_COLUMNS = [5, 6, 7, 8, 9, *range(13, 21)]
FEATURE_NAMES = ['native_distance', 'hindi_distance', 'distance_difference', 'direction_projection',
                 'direction_cosine', 'direction_size', 'fraction_channels_closer_to_hindi']


def direction_features(query, native, hindi, scale):
    query, native, hindi, scale = [np.asarray(v, dtype=float) for v in (query, native, hindi, scale)]
    if any(v.shape != (13,) or not np.isfinite(v).all() for v in (query, native, hindi, scale)) or np.any(scale <= 0):
        raise ValueError('Measured13-dimensional same-word pitch profiles and positive train-only scales required')
    q = (query - native) / scale
    h = (hindi - native) / scale
    nd, hd, size = np.linalg.norm(q), np.linalg.norm(q - h), np.linalg.norm(h)
    projection = float(np.dot(q, h) / max(size ** 2, .01))
    cosine = float(np.dot(q, h) / max(nd * size, .01))
    return np.asarray([nd, hd, nd - hd, projection, cosine, size, np.mean(np.abs(q - h) < np.abs(q))], np.float32)
