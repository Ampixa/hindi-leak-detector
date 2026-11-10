"""Converged proxy learning with natural-speech anchors and grouped validation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from hindi_detector.evaluation import ENCODER_REVISION

SYSTEM_LABELS = {'indicf5_stock': 1, 'audio8_r3': 0, 'native_recording': 0}
SYSTEM_PRIORS = {'indicf5_stock': .5, 'audio8_r3': .25, 'native_recording': .25}


def load_windows(directory: Path) -> tuple[list[dict], np.ndarray]:
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['encoder_revision'] != ENCODER_REVISION or manifest['dimension'] != 4096 or manifest['status'] != 'complete_features_only':
        raise ValueError('Unsupported or unfinished window feature group')
    identifiers = set()
    vectors = []
    for row in manifest['items']:
        if row['id'] in identifiers:
            raise ValueError('Duplicate window identity')
        identifiers.add(row['id'])
        path = (directory / row['path']).resolve(strict=True)
        if not path.is_relative_to(directory.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Window feature path/hash mismatch')
        with np.load(path, allow_pickle=False) as stored:
            checks = {'parent_audio_sha256': row['audio_sha256'], 'driver_sha256': manifest['driver_sha256'],
                      'encoder_revision': ENCODER_REVISION, 'start_sample': row['start_sample'], 'end_sample': row['end_sample']}
            if any(stored[key].item() != value for key, value in checks.items()):
                raise ValueError('Window metadata does not match its input identity')
            vector = stored['speech'].copy()
        if vector.shape != (4096,) or not np.isfinite(vector).all():
            raise ValueError('Invalid window feature vector')
        vectors.append(vector)
    return manifest['items'], np.stack(vectors)


def source_weights(rows: list[dict]) -> np.ndarray:
    systems = np.array([r['system'] for r in rows])
    if set(systems) != set(SYSTEM_LABELS):
        raise ValueError('Fitting/selection requires both TTS sources and natural negative proxies')
    return np.array([SYSTEM_PRIORS[s] / np.sum(systems == s) for s in systems])


def full_pairs(rows: list[dict]) -> list[tuple[int, int]]:
    grouped = {}
    for index, row in enumerate(rows):
        if row['variant'] != 'full':
            continue
        key = (row['speaker_id'], row['sentence_id'], row['text_sha256'])
        group = grouped.setdefault(key, {})
        if row['system'] in group:
            raise ValueError('Duplicate full-utterance system in paired group')
        group[row['system']] = index
    if not grouped or any(set(g) != set(SYSTEM_LABELS) for g in grouped.values()):
        raise ValueError('Every utterance must retain its matched F5, Audio8 and natural full views')
    return [(g['indicf5_stock'], g[negative]) for g in grouped.values() for negative in ['audio8_r3', 'native_recording']]


def logistic_objective(theta, z, labels, weights, pairs, l2, pair_coefficient=.5):
    w, bias = theta[:-1], theta[-1]
    logits = z @ w + bias
    loss = float(weights @ (np.logaddexp(0, logits) - labels * logits) + l2 / 2 * (w @ w))
    residual = weights * (np.exp(-np.logaddexp(0, -logits)) - labels)
    grad = z.T @ residual + l2 * w
    if pairs:
        differences = z[[a for a, _ in pairs]] - z[[b for _, b in pairs]]
        margins = differences @ w
        loss += pair_coefficient * float(np.mean(np.logaddexp(0, -margins)))
        grad -= pair_coefficient * (differences.T @ np.exp(-np.logaddexp(0, margins))) / len(pairs)
    return loss, np.r_[grad, residual.sum()]


def fit_head(rows: list[dict], x: np.ndarray, representation: str, l2: float) -> dict:
    # Keep the optional acoustic fitter out of the base NumPy-only test path.
    from scipy.optimize import minimize

    columns = {'layer12': np.arange(2048), 'layer24': np.arange(2048, 4096), 'both': np.arange(4096)}
    if representation not in columns or x.shape != (len(rows), 4096) or not np.isfinite(x).all() or l2 <= 0:
        raise ValueError('Invalid fit recipe/features')
    selected = columns[representation]
    values = x[:, selected].astype(np.float64)
    mean = values.mean(0)
    raw_scale = values.std(0)
    floor = max(float(np.median(raw_scale)) * .1, 1e-4)
    scale = np.maximum(raw_scale, floor)
    z = (values - mean) / scale
    labels = np.array([SYSTEM_LABELS[r['system']] for r in rows])
    weights, pairs = source_weights(rows), full_pairs(rows)
    result = minimize(logistic_objective, np.zeros(len(selected) + 1),
                      args=(z, labels, weights, pairs, l2), jac=True, method='L-BFGS-B',
                      options={'maxiter': 1000, 'ftol': 1e-11, 'gtol': 1e-7})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f'Convex head did not converge: {result.message}')
    full_weight, full_mean, full_scale = np.zeros(4096), np.zeros(4096), np.ones(4096)
    full_weight[selected], full_mean[selected], full_scale[selected] = result.x[:-1], mean, scale
    return {'weight': full_weight.tolist(), 'bias': float(result.x[-1]),
            'feature_mean': full_mean.tolist(), 'feature_scale': full_scale.tolist(),
            'representation': representation, 'l2': l2, 'scale_floor': floor,
            'solver': {'name': 'scipy L-BFGS-B', 'converged': bool(result.success),
                       'iterations': int(result.nit), 'objective': float(result.fun),
                       'gradient_inf_norm': float(np.max(np.abs(result.jac))), 'message': str(result.message)}}


def validation_loss(rows: list[dict], scores: np.ndarray) -> float:
    y = np.array([SYSTEM_LABELS[r['system']] for r in rows])
    p = np.clip(scores, 1e-10, 1 - 1e-10)
    return float(source_weights(rows) @ -(y * np.log(p) + (1 - y) * np.log1p(-p)))
