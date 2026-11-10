"""Test preferred-reading word differences before scaling to crossed voices."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from hindi_detector.artifacts import write_group
from hindi_detector.reference_words import RECIPE, preferred_pairs
from hindi_detector.word_model import fit_word_head, load_bags, predict_bag
from scripts.train_word_heads import evaluate, weighted_loss

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)


def source_snapshot():
    members = [Path(__file__), ROOT / 'hindi_detector/reference_words.py', ROOT / 'hindi_detector/word_model.py', ROOT / 'scripts/train_word_heads.py']
    return [{'path': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'source_utf8': p.read_text()} for p in members]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Need a new immutable reference-word experiment')
    plan_path = ROOT / 'reports/reference_word_v3_design/plan.json'
    plan = json.loads(plan_path.read_text())
    if plan['recipe'] != RECIPE:
        raise ValueError('Reference recipe differs from locked design')
    snapshots = source_snapshot()
    bags, manifest = load_bags(args.training)
    train = preferred_pairs([b for b in bags if b['split'] == 'train'])
    dev = preferred_pairs([b for b in bags if b['split'] == 'development'])
    train_speakers = {b['speaker_id'] for b in train}; dev_speakers = {b['speaker_id'] for b in dev}
    if train_speakers & dev_speakers or len(train_speakers) != 12:
        raise ValueError('Need the inspected twelve disjoint training groups')
    speakers = sorted(train_speakers, key=lambda s: hashlib.sha256(('reference-cv-v3:' + s).encode()).hexdigest())
    fold_of = {s: i % 3 for i, s in enumerate(speakers)}
    models, choices, reports = {}, {}, {}
    for branch in plan['branches']:
        candidates = []
        for l2 in plan['l2_grid']:
            losses = []
            for fold in range(3):
                fitting = [b for b in train if fold_of[b['speaker_id']] != fold]
                validating = [b for b in train if fold_of[b['speaker_id']] == fold]
                model = fit_word_head(fitting, branch, l2)
                losses.append(weighted_loss(model, validating))
            candidates.append({'l2': l2, 'fold_log_losses': losses, 'mean_cv_log_loss': float(np.mean(losses))})
            LOG.info('%s L2 %s CV %s', branch, l2, candidates[-1]['mean_cv_log_loss'])
        chosen = min(candidates, key=lambda c: c['mean_cv_log_loss'])
        model = fit_word_head(train, branch, chosen['l2'])
        model.update(reference_recipe=RECIPE, status='weak_reference_conditioned_source_evidence',
                     human_word_labels_used=0, validated_word_hindi_accuracy=None,
                     phone_encoder_revision=manifest['phone_encoder_revision'])
        models[branch] = model; choices[branch] = {'selected': chosen, 'candidates': candidates}
        reports[branch] = {'development': evaluate(model, dev)}
    # Model choices are complete before the reused exploratory benchmark is read.
    test_bags, _ = load_bags(args.test)
    if any(b['split'] != 'reserved_evaluation' for b in test_bags) or (train_speakers | dev_speakers) & {b['speaker_id'] for b in test_bags}:
        raise ValueError('Reference experiment split overlap')
    test = preferred_pairs(test_bags)
    rows = []
    for branch, model in models.items():
        reports[branch]['previously_inspected_benchmark'] = evaluate(model, test)
    for pair in test:
        words = [{k: w[k] for k in ['word', 'start', 'end']} for w in pair['words']]
        scores = {}
        for branch, model in models.items():
            score, local = predict_bag(model, pair); scores[branch] = score
            for word, value in zip(words, local, strict=True):
                word[branch + '_source_evidence'] = value
        rows.append({k: v for k, v in pair.items() if k not in ['phonetics', 'prosody', 'phone_mask', 'words']} | {'words': words, 'bag_scores': scores})
    report = {'status': 'reference_conditioned_branches_trained', 'training_pair_bags': len(train),
              'development_pair_bags': len(dev), 'test_pair_bags': len(test),
              'branches': reports, 'validated_word_hindi_accuracy': None,
              'limits': ['Source bag labels are weak. No individual F5 word is assigned Hindi gold.',
                         'Several comparisons share the same query/reference audio and are statistically dependent.',
                         'Native and Audio8 form non-self negative pairs; this does not prove voice/vocoder invariance.',
                         'These benchmark texts have already been inspected; results are exploratory.',
                         'Crossed different-voice preferred readings are the next stronger control; their generation is separate.']}
    write_group(args.out, {'models.json': models, 'report.json': report, 'selection.json': choices,
                          'word_predictions.json': rows, 'source_snapshot.json': snapshots},
                'Reference-conditioned pronunciation/prosody evidence; evaluate whether same-word differences improve the weak baseline.',
                [f'Design SHA {hashlib.sha256(plan_path.read_bytes()).hexdigest()}',
                 str(args.training.resolve()), str(args.test.resolve()),
                 'python -m scripts.train_reference_words --training reports/wordlab_training_v3 --test reports/wordlab_test_v3 --out NEW'])
    print(json.dumps(report))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Reference-conditioned word fit failed; no replacement results')
        raise
