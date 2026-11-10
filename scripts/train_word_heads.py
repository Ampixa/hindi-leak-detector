"""Phone-only, prosody-only and combined weak MIL branches, grouped by speaker."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from hindi_detector.artifacts import write_group
from hindi_detector.evaluation import binary_metrics
from hindi_detector.word_model import LABELS, bag_weights, fit_word_head, load_bags, predict_bag

LOG = logging.getLogger(__name__)


def weighted_loss(model, bags):
    scores = np.array([predict_bag(model, b)[0] for b in bags])
    y = np.array([b['class_id'] for b in bags]); p = np.clip(scores, 1e-8, 1 - 1e-8)
    return float(bag_weights(bags) @ -(y * np.log(p) + (1 - y) * np.log1p(-p)))


def evaluate(model, bags):
    scores = np.array([predict_bag(model, b)[0] for b in bags])
    source = [i for i, b in enumerate(bags) if b['system'] in {'indicf5_stock', 'audio8_r3'}]
    controls = {}
    for system in sorted({b['system'] for b in bags}):
        ix = [i for i, b in enumerate(bags) if b['system'] == system]
        controls[system] = {'n': len(ix), 'high_bag_scores': int(np.sum(scores[ix] >= .5)), 'median_bag_score': float(np.median(scores[ix]))}
    return {'source_bag_metrics': binary_metrics(np.array([bags[i]['class_id'] for i in source]), scores[source]),
            'controls': controls}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a new immutable word-head result directory')
    bags, manifest = load_bags(args.training)
    train = [b for b in bags if b['split'] == 'train']; dev = [b for b in bags if b['split'] == 'development']
    if len(train) + len(dev) != len(bags) or set(b['system'] for b in train) != set(LABELS):
        raise ValueError('Word fitting inputs have unexpected splits/sources')
    train_speakers, dev_speakers = {b['speaker_id'] for b in train}, {b['speaker_id'] for b in dev}
    if train_speakers & dev_speakers or len(train_speakers) < 12:
        raise ValueError('Need twelve training groups disjoint from development')
    speakers = sorted(train_speakers, key=lambda s: hashlib.sha256(('word-cv-v3:' + s).encode()).hexdigest())
    fold_of = {s: i % 3 for i, s in enumerate(speakers)}
    models, selection, reports = {}, {}, {}
    for branch in ['phonetics', 'prosody', 'combined']:
        candidates = []
        for l2 in [.01, .1, 1.]:
            losses = []
            for fold in range(3):
                fitting = [b for b in train if fold_of[b['speaker_id']] != fold]
                validating = [b for b in train if fold_of[b['speaker_id']] == fold]
                model = fit_word_head(fitting, branch, l2)
                losses.append(weighted_loss(model, validating))
            candidate = {'branch': branch, 'l2': l2, 'fold_log_losses': losses, 'mean_cv_log_loss': float(np.mean(losses))}
            candidates.append(candidate)
            LOG.info('Word branch %s L2=%s grouped CV %.5f', branch, l2, candidate['mean_cv_log_loss'])
        chosen = min(candidates, key=lambda c: c['mean_cv_log_loss'])
        models[branch] = fit_word_head(train, branch, chosen['l2'])
        models[branch].update(status='weak_source_bag_word_evidence_not_validated_word_accent',
                              training_speakers=sorted(train_speakers), development_speakers=sorted(dev_speakers),
                              word_accuracy=None, human_word_labels_used=0,
                              phone_encoder_revision=manifest['phone_encoder_revision'],
                              prosody_feature_names=manifest['prosody_feature_names'])
        selection[branch] = {'selected': chosen, 'candidates': candidates}
        reports[branch] = {'development': evaluate(models[branch], dev)}
    # Every branch and regularization choice is fixed before reading test features.
    test, test_manifest = load_bags(args.test)
    if any(b['split'] != 'reserved_evaluation' for b in test) or train_speakers & {b['speaker_id'] for b in test}:
        raise ValueError('Word test split/source-speaker ID leakage')
    source_rows = []
    for branch, model in models.items():
        reports[branch]['previously_inspected_benchmark'] = evaluate(model, test)
    for bag in test:
        words = [{'word': w['word'], 'start': w['start'], 'end': w['end']} for w in bag['words']]
        bag_scores = {}
        for branch, model in models.items():
            bag_score, scores = predict_bag(model, bag); bag_scores[branch] = bag_score
            for word, score in zip(words, scores, strict=True):
                word[branch + '_source_evidence'] = score
        source_rows.append({k: bag[k] for k in ['clip_id', 'audio_sha256', 'sentence_id', 'speaker_id', 'system']} | {'words': words, 'bag_scores': bag_scores})
    report = {'status': 'three_weak_localization_branches_trained', 'train_bags': len(train), 'development_bags': len(dev),
              'train_words': sum(len(b['words']) for b in train), 'development_words': sum(len(b['words']) for b in dev),
              'test_bags': len(test), 'test_words': sum(len(b['words']) for b in test),
              'training_explicit_failures': manifest['failures'], 'test_explicit_failures': test_manifest['failures'],
              'branches': reports, 'human_word_labels_used': 0, 'validated_word_hindi_accuracy': None,
              'limits': ['Only utterance source bags supply weak supervision. Individual F5 words are not labeled Hindi by fiat.',
                         'These local scores indicate evidence for a positive source bag, not certified word-level accent.',
                         'Pitch/energy/register normalization is not proof of voice-conversion or speaker invariance.',
                         'The phonetic probe is unvalidated on Nepali; apparent phone strings and schwa posteriors need component testing.',
                         'The original benchmark/pilot have already been examined; bag metrics are exploratory source comparisons.',
                         'Source speaker IDs are disjoint; cross-corpus real-person identity and upstream text exposure are unverified.']}
    write_group(args.out, {'models.json': models, 'report.json': report, 'selection.json': selection, 'word_predictions.json': source_rows},
                'Separate phonetic/prosodic local evidence trained with weak utterance-bag labels; no word-level accuracy claim.',
                [f'{args.training.resolve()} manifest SHA {hashlib.sha256((args.training / "manifest.json").read_bytes()).hexdigest()}',
                 f'{args.test.resolve()} manifest SHA {hashlib.sha256((args.test / "manifest.json").read_bytes()).hexdigest()}',
                 f'Driver SHA {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}',
                 'python -m scripts.train_word_heads --training reports/wordlab_training_v3 --test reports/wordlab_test_v3 --out NEW'])
    print(json.dumps(report))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Weak word-head training failed; no substitute word labels/scores')
        raise
