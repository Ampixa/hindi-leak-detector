"""Train the user-authorized IndicF5/Audio8 proxy; never report gold accuracy."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn

from hindi_detector.artifacts import write_group
from hindi_detector.evaluation import ENCODER_REVISION, binary_metrics, grouped_intervals, predict_linear
from hindi_detector.proxy import LABEL_SOURCE, PROXY_LABELS, check_disjoint, check_proxy_pairs, load_pack


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--development-pack', type=Path, required=True)
    parser.add_argument('--development-features', type=Path, required=True)
    parser.add_argument('--audio8-pack', type=Path, required=True)
    parser.add_argument('--audio8-features', type=Path, required=True)
    parser.add_argument('--test-pack', type=Path, required=True)
    parser.add_argument('--test-features', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a new output group; existing results are immutable')
    dev_rows, dev_x = load_pack(args.development_features, args.development_pack)
    a8_rows, a8_x = load_pack(args.audio8_features, args.audio8_pack)
    test_rows, test_x = load_pack(args.test_features, args.test_pack)
    all_rows, all_x = dev_rows + a8_rows, np.concatenate([dev_x, a8_x])

    def select(split):
        indices = [i for i, row in enumerate(all_rows) if row['split'] == split and row['system'] in PROXY_LABELS]
        rows = [all_rows[i] for i in indices]
        check_proxy_pairs(rows, split)
        return rows, all_x[indices]

    train, xt = select('train')
    development, xd = select('development')
    if len(train) < 16 or len(development) < 4:
        raise ValueError('Insufficient matched development data')
    check_disjoint(train, development)
    if any(r['split'] != 'reserved_evaluation' for r in test_rows):
        raise ValueError('Final test contains a non-reserved item')
    check_disjoint(train + development, test_rows)
    test_indices = [i for i, row in enumerate(test_rows) if row['system'] in PROXY_LABELS]
    test = [test_rows[i] for i in test_indices]
    check_proxy_pairs(test, 'reserved_evaluation')
    yt = torch.tensor([r['class_id'] for r in train], dtype=torch.float32)
    yd = torch.tensor([r['class_id'] for r in development], dtype=torch.float32)
    mean, scale = xt.mean(0), np.maximum(xt.std(0), 1e-4)
    xt = torch.tensor((xt - mean) / scale, dtype=torch.float32)
    xd = torch.tensor((xd - mean) / scale, dtype=torch.float32)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    head = nn.Linear(4096, 1)
    optimizer = torch.optim.Adam(head.parameters(), lr=.01)
    best_loss, best_state, best_step = float('inf'), None, None
    for step in range(1, 601):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(head(xt).squeeze(1), yt) + .05 * head.weight.square().sum()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            with torch.no_grad():
                dev_loss = float(nn.functional.binary_cross_entropy_with_logits(head(xd).squeeze(1), yd))
            if not np.isfinite(dev_loss):
                raise ValueError('Nonfinite development loss')
            if dev_loss < best_loss:
                best_loss, best_step = dev_loss, step
                best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    if best_state is None:
        raise RuntimeError('No checkpoint selected')
    head.load_state_dict(best_state, strict=True)
    model = {'architecture': 'frozen XLSR layers 12/24 mean+std -> train-standardized 4096D -> L2-regularized logistic head',
             'encoder_revision': ENCODER_REVISION, 'class_ids': PROXY_LABELS,
             'label_source': LABEL_SOURCE, 'status': 'trained_source_proxy_not_validated_hindi_leakage',
             'weight': head.weight.detach().numpy()[0].tolist(), 'bias': float(head.bias.detach()[0]),
             'feature_mean': mean.tolist(), 'feature_scale': scale.tolist(),
             'training_speaker_ids': sorted({r['speaker_id'] for r in train}),
             'development_speaker_ids': sorted({r['speaker_id'] for r in development}),
             'training_text_sha256': sorted({r['text_sha256'] for r in train}),
             'development_text_sha256': sorted({r['text_sha256'] for r in development}),
             'seed': 0, 'learning_rate': .01, 'l2_weight_squared_coefficient': .05,
             'selected_step': best_step, 'max_steps': 600,
             'score_meaning': 'Similarity to IndicF5 rather than Audio8 under the user source-label assumption; not a calibrated probability of Hindi leakage'}
    probabilities = predict_linear(model, test_x)
    y = np.array([r['class_id'] for r in test])
    p = probabilities[test_indices]
    by_system = {}
    for system in sorted({r['system'] for r in test_rows}):
        indices = [i for i, row in enumerate(test_rows) if row['system'] == system]
        scores = probabilities[indices]
        by_system[system] = {'n': len(indices), 'median_indicf5_like_score': float(np.median(scores)),
                             'mean_indicf5_like_score': float(np.mean(scores)),
                             'fraction_over_0_5': float(np.mean(scores >= .5))}
    pair_scores = {}
    for row, score in zip(test, p, strict=True):
        pair_scores.setdefault(row['sentence_id'], {})[row['system']] = float(score)
    margins = [scores['indicf5_stock'] - scores['audio8_r3'] for scores in pair_scores.values()]
    report = {'status': model['status'], 'label_source': LABEL_SOURCE,
              'user_assumption': 'Original IndicF5 is the Hindi-like source proxy; Audio8 R3 is the Nepali-like source proxy.',
              'training_clips': len(train), 'training_speaker_groups': len(model['training_speaker_ids']),
              'development_clips': len(development), 'development_speaker_groups': len(model['development_speaker_ids']),
              'test_clips': len(test), 'test_speaker_groups': len({r['speaker_id'] for r in test}),
              'development_bce': best_loss, 'selected_step': best_step,
              'held_out_source_proxy_metrics': binary_metrics(y, p),
              'held_out_source_proxy_uncertainty': grouped_intervals(y, p, [r['speaker_id'] for r in test]),
              'paired_ordering': {'pairs': len(margins), 'indicf5_above_audio8_fraction': float(np.mean(np.array(margins) > 0)),
                                  'mean_margin': float(np.mean(margins))},
              'reserved_controls_score_distributions': by_system,
              'verified_hindi_leakage_accuracy': None, 'human_label_events_used': 0,
              'limits': ['Training and test labels are source-level assumptions, not observed per-clip accent labels.',
                         'Both TTS architectures appear in training and test; unseen-sentence/voice results do not establish unseen-system accent generalization.',
                         'The head may use vocoder, channel, duration or speaker artifacts. Scores are uncalibrated.',
                         'Native recordings and Audio8 R2 are diagnostic controls, not human-labeled accuracy examples.',
                         'Cross-corpus real-person identity and upstream training-text exposure remain unverified.',
                         'No schwa-site localization or causal checkpoint attribution is established.']}
    predictions = [{'clip_id': row['clip_id'], 'sentence_id': row['sentence_id'], 'speaker_id': row['speaker_id'],
                    'system': row['system'], 'source_system': row['source_system'],
                    'proxy_class': row['class_id'], 'indicf5_like_score': float(score),
                    'label_source': LABEL_SOURCE, 'split': row['split']} for row, score in zip(test_rows, probabilities, strict=True)]
    training_inventory = [{k: row[k] for k in ['clip_id', 'audio_sha256', 'speaker_id', 'sentence_id', 'split', 'system', 'source_system', 'class_id', 'label_source']} for row in train + development]
    write_group(args.out, {'model.json': model, 'report.json': report, 'predictions.json': predictions,
                           'training_inventory.json': training_inventory},
                'User-authorized source-proxy baseline; exploratory score, not verified Hindi-leakage accuracy.',
                [f'Driver SHA256 {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}',
                 'python -m scripts.train_source_proxy with the three documented pack/feature inputs',
                 *[f'{p.resolve()} manifest SHA256 {hashlib.sha256((p / "manifest.json").read_bytes()).hexdigest()}'
                   for p in [args.development_features, args.audio8_features, args.test_features]]])
    print(json.dumps(report))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Source-proxy training failed; no validated leakage result may be claimed')
        raise
