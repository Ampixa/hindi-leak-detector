"""Select word-reference heads with both reference voices and lexemes held out."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from hindi_detector.artifacts import write_group
from hindi_detector.crossed_words import LABELS, PREFERRED, PRIORS, crossed_pairs, focus_log_loss, grouped_predictions, metrics
from hindi_detector.word_model import fit_word_head, load_bags

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--f5-evidence', type=Path, required=True)
    parser.add_argument('--pocket-evidence', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a new immutable crossed-word result group')
    plan = json.loads(args.plan.read_text())
    if plan['experiment_id'] != 'pocket_wordmatrix_v4' or set(plan['systems']) != {'pocket_hindi_parent', PREFERRED}:
        raise ValueError('Unexpected locked Pocket dataset')
    jobs = {j['id']: j for j in plan['jobs']}
    if len(jobs) != 512:
        raise ValueError('Unexpected matrix shape')
    dirs = [args.f5_evidence, args.pocket_evidence]
    inputs = {}
    for directory, expected_count in zip(dirs, [512, 1024], strict=True):
        m = json.loads((directory / 'manifest.json').read_text())
        if m['failures'] or m['successes'] != expected_count or len(m['items']) != expected_count:
            raise ValueError('Investigate failed evidence before fitting a complete crossed matrix')
        inputs[str(directory)] = hashlib.sha256((directory / 'manifest.json').read_bytes()).hexdigest()
    sources = []
    for p in [Path(__file__), ROOT / 'hindi_detector/crossed_words.py', ROOT / 'hindi_detector/reference_words.py', ROOT / 'hindi_detector/word_model.py']:
        sources.append({'path': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'source_utf8': p.read_text()})
    sets = {split: {'voices': {j['speaker_id'] for j in jobs.values() if j['split'] == split},
                    'words': {j['focus_word'] for j in jobs.values() if j['split'] == split}}
            for split in ['train', 'development', 'reserved_evaluation']}
    for i, left in enumerate(sets):
        for right in list(sets)[i + 1:]:
            if sets[left]['voices'] & sets[right]['voices'] or sets[left]['words'] & sets[right]['words']:
                raise ValueError('Crossed held-out voice/focus-word leakage')
    train, dev = [], []
    for directory in dirs:
        selected, _ = load_bags(directory, splits={'train', 'development'})
        train += [b for b in selected if b['split'] == 'train']
        dev += [b for b in selected if b['split'] == 'development']
    for split, bags in [('train', train), ('development', dev)]:
        expected_jobs = {j['id'] for j in jobs.values() if j['split'] == split}
        if len(bags) != 3 * len(expected_jobs) or {b['sentence_id'] for b in bags} != expected_jobs:
            raise ValueError('Incomplete source coverage for split ' + split)
    speakers = sorted(sets['train']['voices'], key=lambda s: hashlib.sha256(('pocket-cross-v4:voice:' + s).encode()).hexdigest())
    words = sorted(sets['train']['words'], key=lambda s: hashlib.sha256(('pocket-cross-v4:word:' + s).encode()).hexdigest())
    if len(speakers) != 8 or len(words) != 8:
        raise ValueError('Expected eight training voice groups and focal lexemes')
    voice_fold = {s: i % 3 for i, s in enumerate(speakers)}
    word_fold = {w: i % 3 for i, w in enumerate(words)}
    folds = []
    for fold in range(3):
        fitting = [b for b in train if voice_fold[b['speaker_id']] != fold and word_fold[jobs[b['sentence_id']]['focus_word']] != fold]
        validating = [b for b in train if voice_fold[b['speaker_id']] == fold and word_fold[jobs[b['sentence_id']]['focus_word']] == fold]
        a, b = crossed_pairs(fitting, jobs), crossed_pairs(validating, jobs)
        train_voices = {p[k] for p in a for k in ['speaker_id', 'reference_speaker_id']}
        val_voices = {p[k] for p in b for k in ['speaker_id', 'reference_speaker_id']}
        if train_voices & val_voices or {p['focus_word'] for p in a} & {p['focus_word'] for p in b}:
            raise ValueError('CV query/reference/focal-word leakage')
        folds.append((a, b))
    full = crossed_pairs(train, jobs); development = crossed_pairs(dev, jobs)
    models, selection, report = {}, {}, {}
    for branch in ['phonetics', 'prosody', 'combined']:
        candidates = []
        for l2 in [.01, .1, 1.]:
            losses = []
            for fitting, validating in folds:
                model = fit_word_head(fitting, branch, l2, source_priors=PRIORS)
                losses.append(focus_log_loss(model, validating))
            candidate = {'l2': l2, 'focus_fold_log_losses': losses, 'mean_focus_cv_log_loss': float(np.mean(losses))}
            candidates.append(candidate); LOG.info('%s L2 %s focal CV %.5f', branch, l2, candidate['mean_focus_cv_log_loss'])
        chosen = min(candidates, key=lambda row: row['mean_focus_cv_log_loss'])
        model = fit_word_head(full, branch, chosen['l2'], source_priors=PRIORS)
        model.update(reference_recipe='signed_and_absolute_query_minus_preferred_word_v1', preferred_source=PREFERRED,
                     status='weak_crossed_word_source_evidence_not_validated_hindi_accuracy',
                     label_source='user_preferred_nepali_checkpoint_and_hindi_source_roles_not_word_gold',
                     phone_encoder_revision=json.loads((args.f5_evidence / 'manifest.json').read_text())['phone_encoder_revision'])
        models[branch] = model; selection[branch] = {'selected': chosen, 'candidates': candidates}
        report[branch] = {'development': metrics(grouped_predictions(model, development))}
    # Reading test acoustics happens only after all branch/regularization choices are fixed.
    test_bags = []
    for directory in dirs:
        selected, _ = load_bags(directory, splits={'reserved_evaluation'})
        test_bags += selected
    test = crossed_pairs(test_bags, jobs)
    predictions = {}
    for branch, model in models.items():
        predictions[branch] = grouped_predictions(model, test)
        report[branch]['held_out_voice_and_focus_word'] = metrics(predictions[branch])
    summary = {'status': 'three_crossed_reference_heads_trained', 'branches': report,
               'train_queries': len(train), 'development_queries': len(dev), 'test_queries': len(test_bags),
               'train_pairs': len(full), 'test_pairs': len(test), 'source_labels': LABELS, 'source_priors': PRIORS,
               'cv_policy': 'Split both voice groups and focal lexemes, including reference voices; choose by focal-word weak source log loss.',
               'human_word_gold_used': 0, 'validated_word_hindi_accuracy': None, 'reliable_automatic_detector': False,
               'limits': ['Focal scores are weak checkpoint/source evidence, not proof a word is a Hindi pronunciation error.',
                          'Positive utterance bags never label every word as erroneous. Focal model selection is still weakly supervised.',
                          'Pocket parent versus Nepali changes checkpoint within one architecture; exact historical initialization hash is unverified.',
                          'The F5 matrix has already been inspected; predefined voice/lexeme splits do not make it pristine untouched evaluation.',
                          'Native-speech false alarms and verified localized/prosody errors remain necessary reliability checks.',
                          'Underlying phonetic probe and G2P candidate limitations remain; no performance claim for Nepal Bhasa/Newa.']}
    write_group(args.out, {'models.json': models, 'report.json': summary, 'selection.json': selection,
                          'test_predictions.json': predictions, 'source_snapshot.json': sources,
                          'folds.json': {'voice_folds': voice_fold, 'word_folds': word_fold}},
                'Train word-focused preferred-reading comparisons across different voices with same-architecture checkpoint controls.',
                ['Plan SHA ' + hashlib.sha256(args.plan.read_bytes()).hexdigest(),
                 'Input evidence manifest hashes ' + json.dumps(inputs),
                 'python -m scripts.train_crossed_words --f5-evidence reports/wordlab_matrix_f5_v3 --pocket-evidence reports/wordlab_pocket_matrix_v4 --plan data/pocket_wordmatrix_v4/plan.json --out NEW'])
    print(json.dumps(summary))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Crossed focal-word training failed; no substitute labels or scores')
        raise
