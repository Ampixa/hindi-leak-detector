"""V2: grouped model selection, natural anchors, short-window robustness."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from hindi_detector.anchored_proxy import SYSTEM_LABELS, fit_head, full_pairs, load_windows, validation_loss
from hindi_detector.artifacts import write_group
from hindi_detector.evaluation import ENCODER_REVISION, binary_metrics, predict_linear
from hindi_detector.proxy import LABEL_SOURCE, check_disjoint, load_pack

LOG = logging.getLogger(__name__)


def source_metrics(rows, scores):
    indices = [i for i, row in enumerate(rows) if row['system'] in {'indicf5_stock', 'audio8_r3'}]
    y = np.array([SYSTEM_LABELS[rows[i]['system']] for i in indices])
    return binary_metrics(y, scores[indices])


def distributions(rows, scores):
    result = {}
    for system in sorted({r['system'] for r in rows}):
        indices = [i for i, row in enumerate(rows) if row['system'] == system]
        values = scores[indices]
        result[system] = {'n': len(indices), 'high_score_count': int(np.sum(values >= .5)),
                          'median': float(np.median(values)), 'mean': float(np.mean(values))}
    return result


def native_probe(rows, scores):
    result = {}
    for variant in sorted({r['variant'] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r['variant'] == variant]
        n = len(indices); count = int(np.sum(scores[indices] >= .5)); rate = count / n
        # Each view stratum has one recording from each distinct source group.
        if len({rows[i]['speaker_id'] for i in indices}) != n:
            raise ValueError('Probe stratum must have one item per source speaker group')
        z = 1.959963984540054; denominator = 1 + z * z / n
        center = (rate + z * z / (2 * n)) / denominator
        half = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
        result[variant] = {'n': n, 'high_score_count': count, 'flag_rate': rate,
                           'wilson95_for_proxy_flag_rate': [max(0., float(center - half)), min(1., float(center + half))]}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--windows', type=Path, required=True)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--test-pack', type=Path, required=True)
    parser.add_argument('--test-features', type=Path, required=True)
    parser.add_argument('--v1-head', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a new immutable result group')
    plan = json.loads(args.plan.read_text())
    if plan['decision_threshold'] != .5 or plan['pair_loss_coefficient'] != .5:
        raise ValueError('Implementation must match the locked recipe')
    rows, x = load_windows(args.windows)
    train_indices = [i for i, r in enumerate(rows) if r['split'] == 'train']
    dev_indices = [i for i, r in enumerate(rows) if r['split'] == 'development']
    if len(train_indices) + len(dev_indices) != len(rows):
        raise ValueError('Reserved inputs reached the fitting feature group')
    train, dev = [rows[i] for i in train_indices], [rows[i] for i in dev_indices]
    check_disjoint(train, dev)
    full_pairs(train); full_pairs(dev)
    speakers = sorted({r['speaker_id'] for r in train}, key=lambda s: hashlib.sha256(('v2cv:' + s).encode()).hexdigest())
    if len(speakers) < 12:
        raise ValueError('Need twelve training source-speaker groups')
    folds = {speaker: i % 3 for i, speaker in enumerate(speakers)}
    candidates = []
    for representation in plan['candidate_representations']:
        for l2 in plan['candidate_l2']:
            losses, solves = [], []
            for fold in range(3):
                fit_i = [i for i in train_indices if folds[rows[i]['speaker_id']] != fold]
                val_i = [i for i in train_indices if folds[rows[i]['speaker_id']] == fold]
                fit_rows, val_rows = [rows[i] for i in fit_i], [rows[i] for i in val_i]
                check_disjoint(fit_rows, val_rows)
                head = fit_head(fit_rows, x[fit_i], representation, l2)
                scores = predict_linear(head, x[val_i])
                losses.append(validation_loss(val_rows, scores)); solves.append(head['solver'])
            candidate = {'representation': representation, 'l2': l2,
                         'mean_grouped_cv_log_loss': float(np.mean(losses)), 'fold_losses': losses,
                         'solver_checks': solves}
            candidates.append(candidate)
            LOG.info('CV %s L2=%s weighted log loss %.6f', representation, l2, candidate['mean_grouped_cv_log_loss'])
    selected = min(candidates, key=lambda row: row['mean_grouped_cv_log_loss'])
    model = fit_head(train, x[train_indices], selected['representation'], selected['l2'])
    model.update({'architecture': 'Frozen XLSR pooled features; native negative anchors; short-window augmentation; L2 logistic + matched full-utterance ranking',
                  'encoder_revision': ENCODER_REVISION, 'label_source': LABEL_SOURCE,
                  'class_ids': SYSTEM_LABELS, 'status': 'trained_anchored_source_proxy_not_validated_hindi_leakage',
                  'decision_threshold': .5, 'pair_loss_coefficient': .5,
                  'native_anchor_assumption': 'Corpus language/native-recording identity is a weak negative, not human accent gold.',
                  'score_meaning': 'IndicF5-like versus Audio8/natural-corpus anchors under weak source assumptions; uncalibrated for Hindi leakage',
                  'training_speaker_ids': sorted({r['speaker_id'] for r in train}),
                  'development_speaker_ids': sorted({r['speaker_id'] for r in dev}),
                  'training_text_sha256': sorted({r['text_sha256'] for r in train}),
                  'development_text_sha256': sorted({r['text_sha256'] for r in dev}),
                  'candidate_selection': 'Three-fold grouped CV on training speakers only; no development/test fitting or threshold tuning'})
    # Candidate and fit are fixed before reading either challenge feature group.
    dev_scores = predict_linear(model, x[dev_indices])
    test_rows, test_x = load_pack(args.test_features, args.test_pack)
    probe_rows, probe_x = load_windows(args.probe)
    check_disjoint(train + dev, test_rows); check_disjoint(train + dev, probe_rows)
    if any(r['split'] != 'reserved_evaluation' for r in test_rows + probe_rows):
        raise ValueError('Challenge inputs are not reserved evaluation')
    v1 = json.loads(args.v1_head.read_text())
    test_scores, old_scores = predict_linear(model, test_x), predict_linear(v1, test_x)
    probe_scores, old_probe = predict_linear(model, probe_x), predict_linear(v1, probe_x)
    full_dev = [i for i, r in enumerate(dev) if r['variant'] == 'full']
    paired = {}
    for row, score in zip(test_rows, test_scores, strict=True):
        if row['system'] in SYSTEM_LABELS:
            paired.setdefault((row['speaker_id'], row['sentence_id']), {})[row['system']] = float(score)
    ordering = {negative: {'pairs': len(paired), 'indicf5_above_negative_count': sum(g['indicf5_stock'] > g[negative] for g in paired.values())}
                for negative in ['audio8_r3', 'native_recording']}
    report = {'status': model['status'], 'machine': 'k2 Apple M2 Ultra CPU; frozen features, no L40S job',
              'selected_recipe': selected, 'candidate_count': len(candidates),
              'threshold': .5, 'test_used_for_selection': False,
              'training': {'views': len(train), 'parent_clips': len({r['parent_clip_id'] for r in train}),
                           'speaker_groups': len(model['training_speaker_ids'])},
              'development': {'views': len(dev), 'parent_clips': len({r['parent_clip_id'] for r in dev}),
                              'source_proxy_metrics_full': source_metrics([dev[i] for i in full_dev], dev_scores[full_dev]),
                              'score_distributions_full': distributions([dev[i] for i in full_dev], dev_scores[full_dev])},
              'previously_inspected_benchmark': {'v1_source_proxy_metrics': source_metrics(test_rows, old_scores),
                                                'v2_source_proxy_metrics': source_metrics(test_rows, test_scores),
                                                'v1_controls': distributions(test_rows, old_scores), 'v2_controls': distributions(test_rows, test_scores),
                                                'v2_pair_ordering': ordering},
              'fresh_native_negative_proxy_probe': {'speaker_groups': len({r['speaker_id'] for r in probe_rows}),
                                                    'v1': native_probe(probe_rows, old_probe), 'v2': native_probe(probe_rows, probe_scores)},
              'human_label_events_used': 0, 'verified_hindi_leakage_accuracy': None,
              'limitations': ['Source labels and native negative anchors are weak assumptions; no human accent gold was invented.',
                              'The original benchmark was inspected during v1 development. Its v2 comparison is exploratory, not pristine test accuracy.',
                              'The fresh probe tests only native negative proxies, not Hindi-like sensitivity or unseen-TTS architecture generalization.',
                              'Full and cropped views share parents; group identities stay together and per-view-stratum uncertainty uses 32 distinct source groups.',
                              'A classifier may still learn synthesizer/channel cues. No schwa localization or causal Hindi checkpoint attribution is established.',
                              'Cross-corpus real-person identity and upstream model training-text exposure remain unverified.']}
    predictions = [{'clip_id': r['clip_id'], 'system': r['system'], 'sentence_id': r['sentence_id'],
                    'speaker_id': r['speaker_id'], 'v1_score': float(a), 'v2_score': float(b)}
                   for r, a, b in zip(test_rows, old_scores, test_scores, strict=True)]
    probe_predictions = [{**r, 'v1_score': float(a), 'v2_score': float(b)}
                         for r, a, b in zip(probe_rows, old_probe, probe_scores, strict=True)]
    write_group(args.out, {'model.json': model, 'report.json': report, 'candidates.json': candidates,
                           'predictions.json': predictions, 'probe_predictions.json': probe_predictions,
                           'training_inventory.json': train, 'development_inventory.json': dev},
                'V2 native-anchored duration-robust proxy improvement; no validated accent accuracy claim.',
                [f'Plan SHA256 {hashlib.sha256(args.plan.read_bytes()).hexdigest()}',
                 f'Driver SHA256 {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}',
                 f'Fitter SHA256 {hashlib.sha256((Path(__file__).resolve().parents[1] / "hindi_detector/anchored_proxy.py").read_bytes()).hexdigest()}',
                 *[f'{p.resolve()} manifest SHA256 {hashlib.sha256((p / "manifest.json").read_bytes()).hexdigest()}' for p in [args.windows, args.probe, args.test_features]],
                 'python -m scripts.train_anchored_proxy with documented windows/probe/test/v1-head/plan paths'])
    print(json.dumps(report))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Anchored proxy revision failed; keep v1 and diagnose explicit failure')
        raise
