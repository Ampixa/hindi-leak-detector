"""Honest binary metrics and held-out checks for reviewed leakage judgments."""
from __future__ import annotations

import numpy as np

ENCODER_REVISION = "2c733782da5604684829819a5eb744c193fe9398"


def require_feature_recipe(manifest: dict) -> None:
    if (manifest.get("model_revision") != ENCODER_REVISION
            or manifest.get("feature_dimension") != 4096
            or manifest.get("layers") != [12, 24]
            or manifest.get("pooling") != ["mean", "std"]):
        raise ValueError("unsupported feature encoder or pooling recipe")
    identifiers = [row["clip_id"] for row in manifest["items"]]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate feature clip IDs")


def predict_linear(model: dict, features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    weights = np.asarray(model["weight"], dtype=np.float64)
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    if x.ndim != 2 or any(value.shape != (x.shape[1],) for value in [weights, mean, scale]):
        raise ValueError("model/feature dimensions do not agree")
    if not all(np.isfinite(value).all() for value in [x, weights, mean, scale]) or not np.isfinite(model["bias"]):
        raise ValueError("nonfinite model or features")
    if (scale <= 0).any():
        raise ValueError("feature scales must be positive")
    logits = ((x - mean) / scale) @ weights + model["bias"]
    result = np.empty_like(logits)
    positive = logits >= 0
    result[positive] = 1 / (1 + np.exp(-logits[positive]))
    values = np.exp(logits[~positive])
    result[~positive] = values / (1 + values)
    return result


def require_held_out(model: dict, rows: list[dict]) -> None:
    forbidden_speakers = set(model["training_speaker_ids"]) | set(model["development_speaker_ids"])
    forbidden_texts = set(model["training_text_sha256"]) | set(model["development_text_sha256"])
    for row in rows:
        if row["split"] != "reserved_evaluation":
            raise ValueError("final evaluation requires reserved-evaluation items")
        if row["speaker_id"] in forbidden_speakers:
            raise ValueError("evaluation speaker ID occurred in training/development")
        if row["text_sha256"] in forbidden_texts:
            raise ValueError("evaluation normalized text occurred in training/development")


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or labels.shape != scores.shape or not len(labels):
        raise ValueError("metrics require matching nonempty one-dimensional arrays")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("metrics need binary labels and finite probabilities in [0,1]")
    if not 0 <= threshold <= 1:
        raise ValueError("invalid prediction threshold")
    predictions = scores >= threshold
    tp = int((predictions & (labels == 1)).sum())
    fp = int((predictions & (labels == 0)).sum())
    fn = int((~predictions & (labels == 1)).sum())
    tn = int((~predictions & (labels == 0)).sum())
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {"n": len(labels), "positive_labels": int((labels == 1).sum()),
            "negative_labels": int((labels == 0).sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else None, "recall": recall,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None,
            "accuracy": (tp + tn) / len(labels),
            "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
            "brier": float(np.mean((scores - labels) ** 2))}


def agreement_decisions(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if first.ndim != 1 or first.shape != second.shape or not len(first):
        raise ValueError('Agreement requires matching nonempty score vectors')
    if any(not np.isfinite(s).all() or ((s < 0) | (s > 1)).any() for s in [first, second]):
        raise ValueError('Agreement requires finite scores in [0,1]')
    a, b = first >= .5, second >= .5
    return np.where(a == b, a.astype(int), -1)


def selective_metrics(labels: np.ndarray, decisions: np.ndarray) -> dict:
    labels, decisions = np.asarray(labels), np.asarray(decisions)
    if labels.ndim != 1 or labels.shape != decisions.shape or not len(labels):
        raise ValueError('Selective metrics require matching nonempty vectors')
    if not np.isin(labels, [0, 1]).all() or not np.isin(decisions, [-1, 0, 1]).all():
        raise ValueError('Invalid reference/decision classes')
    covered = decisions >= 0
    correct = (decisions == labels) & covered
    positive, negative = labels == 1, labels == 0
    return {'n': len(labels), 'covered': int(covered.sum()), 'abstained': int((~covered).sum()),
            'coverage': float(covered.mean()),
            'covered_accuracy': float(correct[covered].mean()) if covered.any() else None,
            'correct_fraction_of_all_items': float(correct.mean()),
            'positive_recall_including_abstentions': float((decisions[positive] == 1).mean()) if positive.any() else None,
            'negative_flag_rate_including_abstentions': float((decisions[negative] == 1).mean()) if negative.any() else None}


def grouped_intervals(labels: np.ndarray, scores: np.ndarray, speakers: list[str], draws: int = 1000) -> dict:
    labels, scores = np.asarray(labels), np.asarray(scores)
    if len(speakers) != len(labels) or any(not isinstance(s, str) or not s for s in speakers):
        raise ValueError("every test item requires a valid speaker group")
    if draws < 100:
        raise ValueError("at least 100 bootstrap draws are required")
    binary_metrics(labels, scores)
    unique = sorted(set(speakers))
    keys = ["precision", "recall", "false_positive_rate", "balanced_accuracy"]
    result = {"speaker_groups": len(unique), "draws": draws, "method": "speaker-group bootstrap percentile 95%"}
    if len(unique) < 2:
        result["intervals"] = {key: {"ci95": None, "defined_draws": 0} for key in keys}
        result["limitation"] = "Fewer than two speaker groups; uncertainty is not estimable."
        return result
    by_group = {group: np.flatnonzero(np.array(speakers) == group) for group in unique}
    rng = np.random.default_rng(0)
    values = {key: [] for key in keys}
    for _ in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([by_group[group] for group in sampled])
        metrics = binary_metrics(labels[indices], scores[indices])
        for key in keys:
            if metrics[key] is not None:
                values[key].append(metrics[key])
    result["intervals"] = {
        key: {"ci95": np.percentile(value, [2.5, 97.5]).tolist() if len(value) >= draws * 0.9 else None,
              "defined_draws": len(value)} for key, value in values.items()}
    result["limitation"] = "Small or perfect test samples can give misleadingly narrow bootstrap intervals; report counts and collect more independent speakers."
    return result
