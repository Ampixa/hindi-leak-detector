"""Exact CTC likelihood ratio for reviewed phone hypotheses."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


def validate_log_probs(log_probs: np.ndarray, blank: int) -> np.ndarray:
    values = np.asarray(log_probs, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("log probabilities must have shape [nonempty frames, vocabulary]")
    if not 0 <= blank < values.shape[1]:
        raise ValueError("blank token ID is outside vocabulary")
    if np.isnan(values).any() or np.isposinf(values).any() or (values > 1e-7).any():
        raise ValueError("expected log probabilities, not logits or NaN")
    if not np.allclose(np.logaddexp.reduce(values, axis=1), 0, atol=1e-5):
        raise ValueError("each probability row must sum to one")
    return values


def log_likelihood(log_probs: np.ndarray, labels: list[int], blank: int = 0) -> float:
    values = validate_log_probs(log_probs, blank)
    if any(type(label) is not int or not 0 <= label < values.shape[1] or label == blank for label in labels):
        raise ValueError("labels must be vocabulary IDs and cannot contain blank")
    if not labels:
        return float(values[:, blank].sum())
    if len(labels) + sum(a == b for a, b in zip(labels, labels[1:])) > len(values):
        return -math.inf
    extended = [blank]
    for label in labels:
        extended.extend([label, blank])
    alpha = np.full(len(extended), -np.inf)
    alpha[0], alpha[1] = values[0, blank], values[0, labels[0]]
    for frame in values[1:]:
        previous = alpha
        alpha = np.full(len(extended), -np.inf)
        for i, token in enumerate(extended):
            total = previous[i]
            if i > 0:
                total = np.logaddexp(total, previous[i - 1])
            if i > 1 and token != blank and token != extended[i - 2]:
                total = np.logaddexp(total, previous[i - 2])
            alpha[i] = total + frame[token]
    return float(np.logaddexp(alpha[-1], alpha[-2]))


@dataclass(frozen=True)
class Evidence:
    target_log_likelihood: float
    hindi_log_likelihood: float
    hindi_minus_target_log_likelihood: float
    status: str = "uncalibrated_acoustic_evidence"

    def to_dict(self) -> dict:
        return asdict(self)


def compare(log_probs: np.ndarray, target: list[int], hindi: list[int], blank: int = 0) -> Evidence:
    if target == hindi:
        raise ValueError("identical hypotheses cannot diagnose transfer")
    native = log_likelihood(log_probs, target, blank)
    source = log_likelihood(log_probs, hindi, blank)
    if not math.isfinite(native) or not math.isfinite(source):
        raise ValueError("at least one hypothesis is impossible; crop/labels need review")
    # Raw ratio, NOT a calibrated probability of Hindi leakage. Do not divide
    # each sequence by its own phone count: that changes the hypothesis test.
    return Evidence(native, source, source - native)
