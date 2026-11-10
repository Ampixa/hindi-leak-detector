"""Train/dev freeze before opening physically transferred held-out pitch styles."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from scripts.extract_corrected_training import read_rows

ROOT = Path(__file__).resolve().parents[1]


class PitchStyleNetwork(nn.Module):
    def __init__(self, dimension=300, hidden=16):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(dimension, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(.2), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.head(x).squeeze(-1)


def load_arrays(directory, rows, conditions, *, feature_key='paired_prosody', dimension=300):
    values, labels, metadata = [], [], []
    for row in rows:
        for output in row['outputs']:
            if output['condition'] not in conditions:
                continue
            path = (directory / output['features']).resolve(strict=True)
            if not path.is_relative_to(directory.resolve()) or sha(path) != output['features_sha256']:
                raise ValueError('Physical output features changed')
            with np.load(path, allow_pickle=False) as arrays:
                x = arrays[feature_key].astype(np.float32)
            if x.shape != (dimension,) or not np.isfinite(x).all():
                raise ValueError('Actual paired contextual prosody dimensions/values differ')
            values.append(x)
            labels.append(output['source_style_label'])
            metadata.append({'id': row['id'], 'word': row['word'], 'condition': output['condition'],
                             'split': row['split'], 'label': output['source_style_label']})
    if not values:
        raise ValueError('No eligible physical pitch interventions')
    return np.stack(values), np.array(labels, np.float32), metadata


def evaluate(scores, labels, threshold):
    positive = labels == 1
    detected = scores >= threshold
    if not positive.any() or positive.all():
        raise ValueError('Both known intervention styles required')
    return {'positive_count': int(positive.sum()), 'negative_count': int((~positive).sum()),
            'true_positive': int(np.sum(detected & positive)), 'false_positive': int(np.sum(detected & ~positive)),
            'recall': float(np.mean(detected[positive])), 'specificity': float(np.mean(~detected[~positive])),
            'balanced_accuracy': float((np.mean(detected[positive]) + np.mean(~detected[~positive])) / 2),
            'actual_hindi_error_accuracy': None}


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--rhythm', action='store_true')
    modes.add_argument('--rhythm-v52', action='store_true')
    args = parser.parse_args()
    rhythm = args.rhythm or args.rhythm_v52
    version = 'v52' if args.rhythm_v52 else 'v51'
    directory = ROOT / ('reports/rhythm_interventions_' + version if rhythm else 'reports/prosody_interventions_v49')
    out = ROOT / ('reports/rhythm_transfer_training_' + version if rhythm else 'reports/prosody_transfer_training_v49')
    geometry = {'feature_key': 'paired_rhythm', 'dimension': 40} if rhythm else {}
    dimension, hidden = (40, 8) if rhythm else (300, 16)
    if out.exists():
        raise FileExistsError(out)
    rows = [row for row in read_rows(directory / 'rows.jsonl') if row['status'] == ('ok_rhythm_interventions' if rhythm else 'ok_pitch_interventions')]
    by_part = {part: [row for row in rows if row['split'] == part] for part in ('train', 'development', 'reserved_evaluation')}
    if any(len(value) < minimum for value, minimum in zip(by_part.values(), (60, 15, 25), strict=True)):
        raise ValueError('Insufficient qualified independent word contexts; do not silently weaken quotas')
    word_sets = {part: {row['word'] for row in value} for part, value in by_part.items()}
    group_sets = {part: {v for row in value for v in row['conditioning_groups'].values()} for part, value in by_part.items()}
    for sets in (word_sets, group_sets):
        parts = list(sets)
        if any(sets[a] & sets[b] for i, a in enumerate(parts) for b in parts[i + 1:]):
            raise ValueError('Word or conditioning identity crossed evaluation partitions')
    if {'हैन', 'किनकि'} & set.union(*word_sets.values()):
        raise ValueError('Known user development words leaked into physical training')
    write_group(out, {'inputs.json': {'physical_rows_sha256': sha(directory / 'rows.jsonl'),
        'word_sets': {k: sorted(v) for k, v in word_sets.items()},
        'conditioning_groups': {k: sorted(v) for k, v in group_sets.items()},
        'seed': 20261109, 'mode': 'word_rhythm' if rhythm else 'pitch_only',
        'dimensions': [dimension, hidden, 1], 'fit_conditions': ['sham', 'native_transfer', 'hindi_transfer'],
        'threshold_policy': 'High=max(.5,max dev negative+.00001); reserve never tunes threshold.',
        'model_selection': 'Minimum development balanced binary cross-entropy across300 fixed epochs; no reserved inputs loaded before model/threshold saved.'},
        'source_snapshot.json': {'sha256': sha(Path(__file__)), 'source_utf8': Path(__file__).read_text()}},
        ('Rhythm-only40->8->1 candidate, measured emitted MMS word durations and relative pauses; no pitch features. ' if rhythm else '') +
        'Prosody-only source-style candidate trained on known physical transfers '
        'with identical Native carrier audio for both labels. model.pt stores selected state, '
        'training-only normalization and frozen threshold; frozen_selection.json proves model '
        'identity/selection before reserved score loading. history.json training/dev losses; '
        'report.json held-out known-intervention metrics and nuisance controls, scores.json '
        'per-context scores. Inputs pin distinct word/conditioning-group partitions. No phone '
        'features, voice embeddings, speaker/system IDs or word IDs enter network. NOT a '
        'validated Hindi-error detector, never automatically promoted to live vowel model. '
        'Consumer causal prosody admission experiment, review2026-11-10. Keep selected state '
        'and compact metrics; no historical checkpoints/WAV copies/HF upload; inherited rights.',
        ['.venv-indicf5/bin/python -m scripts.train_prosody_transfer' + (' --rhythm-v52' if args.rhythm_v52 else ' --rhythm' if rhythm else '')])
    conditions = {'sham', 'native_transfer', 'hindi_transfer'}
    train, yt, mt = load_arrays(directory, by_part['train'], conditions, **geometry)
    dev, yd, md = load_arrays(directory, by_part['development'], conditions, **geometry)
    mean = train.mean(axis=0)
    scale = np.maximum(train.std(axis=0), .2)
    transform = lambda x: torch.from_numpy(np.clip((x - mean) / scale, -8, 8).astype(np.float32))
    xt, xd = transform(train), transform(dev)
    ytt, ydd = torch.from_numpy(yt), torch.from_numpy(yd)
    torch.set_num_threads(2)
    torch.manual_seed(20261109)
    network = PitchStyleNetwork(dimension, hidden)
    optimizer = torch.optim.AdamW(network.parameters(), lr=.001, weight_decay=.1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.))
    history = []
    best = None
    for epoch in range(1, 301):
        network.train()
        optimizer.zero_grad()
        loss = criterion(network(xt), ytt)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite physical prosody fitting loss')
        loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), 2.)
        optimizer.step()
        network.eval()
        with torch.inference_mode():
            dev_loss = float(criterion(network(xd), ydd))
        history.append({'epoch': epoch, 'train_loss': float(loss.detach()), 'dev_loss': dev_loss})
        if best is None or dev_loss < best['dev_loss']:
            best = {'epoch': epoch, 'dev_loss': dev_loss, 'state_dict': {k: v.detach().clone() for k, v in network.state_dict().items()}}
    network.load_state_dict(best['state_dict'], strict=True)
    network.eval()
    with torch.inference_mode():
        dev_scores = network(xd).sigmoid().numpy()
    threshold = max(.5, float(dev_scores[yd == 0].max()) + .00001)
    state = {'schema': 'rhythm_transfer_style_' + version if rhythm else 'pitch_transfer_style_v49',
             'state_dict': best['state_dict'], 'mean': mean, 'scale': scale,
             'high_threshold': threshold, 'epoch': best['epoch'], 'input_dimension': dimension, 'hidden_dimension': hidden,
             'actual_hindi_error_accuracy': None, 'labels': 'Known stockIndicF5-versus-third-Native transferred style, not real accent/error gold',
             'physical_input_sha256': sha(directory / 'rows.jsonl'), 'source_sha256': sha(Path(__file__))}
    torch.save(state, out / 'model.pt')
    freeze = {'model_sha256': sha(out / 'model.pt'), 'threshold': threshold, 'epoch': best['epoch'],
              'created_utc': datetime.now(timezone.utc).isoformat(), 'reserved_score_inputs_loaded': False}
    (out / 'frozen_selection.json').write_text(json.dumps(freeze, indent=2) + '\n')
    # The first reserved feature loading/scoring happens strictly after the state
    # and threshold receipt above has been written. No rerun/tuning on this set.
    reserved, yr, mr = load_arrays(directory, by_part['reserved_evaluation'], conditions | {
        'sham_pitch_plus3', 'sham_pitch_minus3', 'sham_gain'}, **geometry)
    with torch.inference_mode():
        rs = network(transform(reserved)).sigmoid().numpy()
        ts = network(xt).sigmoid().numpy()
    if not all(np.isfinite(s).all() for s in (rs, ts, dev_scores)):
        raise ValueError('Nonfinite frozen pitch-style scores')
    base = np.array([row['condition'] in conditions for row in mr])
    scored = [{**row, 'source_style_score': float(score), 'high_source_style_flag': bool(score >= threshold),
               'actual_hindi_error_probability': None} for row, score in zip(mt + md + mr, np.concatenate([ts, dev_scores, rs]), strict=True)]
    indexed = {(row['id'], row['condition']): row for row in scored if row['split'] == 'reserved_evaluation'}
    nuisance = {}
    for condition in ('sham_pitch_plus3', 'sham_pitch_minus3', 'sham_gain'):
        controls = [row for row in scored if row['condition'] == condition]
        nuisance[condition] = {'contexts': len(controls),
            'added_flags_over_sham': sum(row['high_source_style_flag'] and not indexed[row['id'], 'sham']['high_source_style_flag'] for row in controls),
            'max_absolute_score_difference': max(abs(row['source_style_score'] - indexed[row['id'], 'sham']['source_style_score']) for row in controls)}
    report = {'status': 'complete_frozen_rhythm_transfer_experiment' if rhythm else 'complete_frozen_pitch_transfer_experiment',
        'model_sha256': freeze['model_sha256'],
        'threshold': threshold, 'best_epoch': best['epoch'],
        'word_contexts': {k: len(v) for k, v in by_part.items()},
        'train': evaluate(ts, yt, threshold), 'development': evaluate(dev_scores, yd, threshold),
        'reserved': evaluate(rs[base], yr[base], threshold), 'nuisance': nuisance,
        'reserved_by_condition_flags': dict(Counter(row['condition'] for row in scored if row['split'] == 'reserved_evaluation' and row['high_source_style_flag'])),
        'actual_hindi_error_accuracy': None, 'promotion': 'not_promoted_pending_real_prosody_error_validation',
        'limits': ['Held-out words and conditioning groups measure known source-style transfers only.',
                   ('Native spectra/pitch values retimed under a monotone duration map; WORD reconstruction can alter percepts and phone sequences.' if rhythm else
                    'Native spectral envelope/aperiodicity/voicing/spans fixed, but WORLD resynthesis and pitch remeasurement can alter percepts.'),
                   'Rhythm and pitch-only interventions are separate; neither validates natural Hindi-error detection.',
                   'No error labels exist for the real Swar prosody examples; source style does not establish an unacceptable pronunciation.']}
    if rhythm:
        # Do not hide possible phonetic damage from duration compression behind
        # a source-style metric. Compare controls to their WORLD sham baseline.
        stable_ids = {row['id'] for row in by_part['reserved_evaluation'] if all(
            output['additional_phone_edits_over_sham'] <= 0 for output in row['outputs'] if output['condition'] in conditions)}
        mask = base & np.array([row['id'] in stable_ids for row in mr])
        report['phone_stability'] = {'qualified_contexts': len(stable_ids), 'total_contexts': len(by_part['reserved_evaluation']),
            'criterion': 'Neither donor condition adds free-phone edits versus the original carrier beyond WORLD sham.',
            'interpretation': 'Probe stability diagnostic, not independent human phone truth.',
            'stable_subset_metrics': evaluate(rs[mask], yr[mask], threshold) if mask.any() else None}
        report['prior_exposure'] = 'Word families previously used in pitch-only audit; held out from rhythm fitting, not globally untouched.'
    for name, value in (('report.json', report), ('scores.json', scored), ('history.json', history)):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Physical prosody training/evaluation failed explicitly; retain frozen state and do not retune reserved scores')
        raise
