"""Weakly supervised word evidence; no per-word Hindi gold is fabricated."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

LABELS = {'indicf5_stock': 1, 'audio8_r3': 0, 'native_recording': 0}
PRIORS = {'indicf5_stock': .5, 'audio8_r3': .25, 'native_recording': .25}


def load_bags(directory: Path, *, splits: set[str] | None = None, clip_ids: set[str] | None = None) -> tuple[list[dict], dict]:
    manifest = json.loads((directory / 'manifest.json').read_text())
    bags = []
    for row in manifest['items']:
        if row['status'] != 'ok_unreviewed_word_evidence':
            continue
        if splits is not None and row['split'] not in splits:
            continue
        if clip_ids is not None and row['clip_id'] not in clip_ids:
            continue
        detail, feature = directory / row['detail'], directory / row['features']
        if any(not p.resolve().is_relative_to(directory.resolve()) for p in [detail, feature]):
            raise ValueError('Evidence paths outside group')
        if hashlib.sha256(detail.read_bytes()).hexdigest() != row['detail_sha256'] or hashlib.sha256(feature.read_bytes()).hexdigest() != row['feature_sha256']:
            raise ValueError('Word evidence checksum mismatch')
        data = json.loads(detail.read_text())
        if any(data[k] != row[k] for k in ('clip_id', 'system', 'speaker_id', 'sentence_id', 'split', 'text')) or data['audio_sha256'] != row['audio_sha256']:
            raise ValueError('Word detail/manifest identity mismatch')
        with np.load(feature, allow_pickle=False) as stored:
            if stored['parent_audio_sha256'].item() != row['audio_sha256']:
                raise ValueError('Word features/audio identity mismatch')
            phonetics, prosody = stored['phonetics'].copy(), stored['prosody'].copy()
        n = len(data['words'])
        if phonetics.shape != (n, 2 * len(manifest['phone_vocabulary'])) or prosody.shape != (n, len(manifest['prosody_feature_names'])):
            raise ValueError('Word feature schema mismatch')
        if not np.isfinite(phonetics).all() or not np.isfinite(prosody).all():
            raise ValueError('Nonfinite word features')
        system = {'audio8_dpo_R3': 'audio8_r3'}.get(row['system'], row['system'])
        bags.append({'clip_id': row['clip_id'], 'audio_sha256': row['audio_sha256'],
                     'text': row['text'],
                     'sentence_id': row['sentence_id'], 'speaker_id': row['speaker_id'], 'split': row['split'],
                     'system': system, 'class_id': LABELS.get(system), 'words': data['words'],
                     'phonetics': phonetics, 'prosody': prosody,
                     'phone_mask': np.array([w['phone_evidence_available'] for w in data['words']], dtype=bool)})
    if clip_ids is not None and {b['clip_id'] for b in bags} != clip_ids:
        raise ValueError('Explicit word-feature selection has missing or failed clips')
    if len({b['clip_id'] for b in bags}) != len(bags):
        raise ValueError('Duplicate word feature clip identity')
    return bags, manifest


def bag_weights(bags: list[dict], priors: dict[str, float] | None = None) -> np.ndarray:
    priors = PRIORS if priors is None else priors
    systems = [b['system'] for b in bags]
    if set(systems) != set(priors) or not priors or any(not np.isfinite(v) or v <= 0 for v in priors.values()) or not np.isclose(sum(priors.values()), 1.):
        raise ValueError('Need every declared source and finite positive priors summing to one')
    return np.array([priors[s] / systems.count(s) for s in systems], dtype=np.float32)


def vectors(bag: dict, branch: str) -> tuple[np.ndarray, np.ndarray]:
    if branch == 'prosody':
        return bag['prosody'], np.ones(len(bag['words']), dtype=bool)
    if branch == 'phonetics':
        return bag['phonetics'], bag['phone_mask']
    if branch == 'combined':
        return np.concatenate([bag['phonetics'], bag['prosody']], axis=1), bag['phone_mask']
    raise ValueError('Unknown word branch')


def pool_logits(logits: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 1 or not len(logits) or not np.isfinite(logits).all():
        raise ValueError('Need finite nonempty word logits')
    maximum = float(logits.max())
    return maximum + float(np.log(np.exp(logits - maximum).mean()))


def predict_bag(model: dict, bag: dict) -> tuple[float, list[float | None]]:
    values, valid = vectors(bag, model['branch'])
    if not valid.any():
        raise ValueError('No usable words; cannot infer a native/Hindi bag class')
    mean, scale, weight = (np.asarray(model[k]) for k in ['feature_mean', 'feature_scale', 'weight'])
    if values.shape[1] != len(weight) or mean.shape != weight.shape or scale.shape != weight.shape or (scale <= 0).any():
        raise ValueError('Word model/feature recipe mismatch')
    if not all(np.isfinite(a).all() for a in [values, mean, scale, weight]) or not np.isfinite(model['bias']):
        raise ValueError('Nonfinite word model/input')
    logits = ((values[valid] - mean) / scale) @ weight + model['bias']
    probabilities = np.exp(-np.logaddexp(0, -logits))
    scores = [None] * len(values)
    for index, probability in zip(np.flatnonzero(valid), probabilities, strict=True):
        scores[int(index)] = float(probability)
    pooled = pool_logits(logits)
    return float(np.exp(-np.logaddexp(0, -pooled))), scores


def fit_word_head(bags: list[dict], branch: str, l2: float, *, source_priors: dict[str, float] | None = None, negative_word_weight: float = .25) -> dict:
    import torch

    torch.set_num_threads(2)
    torch.manual_seed(0)
    if not np.isfinite(l2) or l2 <= 0 or not np.isfinite(negative_word_weight) or negative_word_weight < 0:
        raise ValueError('Need positive regularization and nonnegative explicit negative-word weight')
    raw, masks = zip(*(vectors(b, branch) for b in bags))
    if any(not m.any() for m in masks):
        raise ValueError('Training contains an unusable word bag')
    weights = bag_weights(bags, source_priors)
    if any(b['class_id'] not in {0, 1} for b in bags):
        raise ValueError('Training requires explicit weak source labels')
    # Equal source/utterance weight: long utterances do not dominate scaling.
    mean = sum(float(w) * x[m].mean(0) for w, x, m in zip(weights, raw, masks, strict=True))
    variance = sum(float(w) * ((x[m] - mean) ** 2).mean(0) for w, x, m in zip(weights, raw, masks, strict=True))
    scale = np.maximum(np.sqrt(variance), max(float(np.median(np.sqrt(variance))) * .1, 1e-4))
    max_words, dimension = max(int(m.sum()) for m in masks), len(mean)
    z, mask = np.zeros((len(bags), max_words, dimension), dtype=np.float32), np.zeros((len(bags), max_words), dtype=bool)
    for i, (x, m) in enumerate(zip(raw, masks, strict=True)):
        n = int(m.sum()); z[i, :n] = (x[m] - mean) / scale; mask[i, :n] = True
    z = torch.tensor(z); mask = torch.tensor(mask)
    y = torch.tensor([b['class_id'] for b in bags], dtype=torch.float32)
    mass = torch.tensor(weights)
    coefficient = torch.nn.Parameter(torch.zeros(dimension)); bias = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([coefficient, bias], lr=.01)
    final_loss = None
    for _ in range(400):
        optimizer.zero_grad(set_to_none=True)
        local = z @ coefficient + bias
        bag_logits = torch.logsumexp(local.masked_fill(~mask, -float('inf')), dim=1) - torch.log(mask.sum(1).float())
        bag_loss = torch.nn.functional.binary_cross_entropy_with_logits(bag_logits, y, reduction='none')
        negative_word_loss = (torch.nn.functional.softplus(local) * mask).sum(1) / mask.sum(1)
        loss = (mass * (bag_loss + negative_word_weight * (1 - y) * negative_word_loss)).sum() + l2 / 2 * coefficient.square().sum()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite weak-word training loss')
        loss.backward(); optimizer.step(); final_loss = float(loss.detach())
    if not torch.isfinite(coefficient).all() or not torch.isfinite(bias):
        raise ValueError('Nonfinite fitted word head')
    return {'branch': branch, 'weight': coefficient.detach().numpy().tolist(), 'bias': float(bias.detach()),
            'feature_mean': mean.tolist(), 'feature_scale': scale.tolist(), 'l2': l2,
            'steps': 400, 'seed': 0, 'training_loss': final_loss,
            'source_priors': PRIORS if source_priors is None else source_priors,
            'negative_word_weight': negative_word_weight,
            'pooling': 'log-mean-exp of word logits, temperature 1',
            'word_score_meaning': 'Weak evidence for the positive source bag; not a calibrated Hindi word label'}
