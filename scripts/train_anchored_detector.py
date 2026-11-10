"""Train-only joint group/voice CV; dev calibration; never opens test rows."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import numpy as np
import torch

from hindi_detector.anchored_detector import ALL_NAMES, INDICES, cells, fit, score, weights
from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from scripts.extract_corrected_training import read_rows

ROOT = Path(__file__).resolve().parents[1]


def read_partition(part):
    directory = ROOT / ('reports/native_main_controls_' + part + '_v71')
    content = ROOT / ('reports/native_main_content_' + part + '_v71/result.json')
    report = json.loads((directory / 'report.json').read_text())
    if report['status'] != 'complete_reviewed_local_controls' or sha(directory / 'items.jsonl') != report['items_sha256']:
        raise ValueError('Incomplete/changed partition: ' + part)
    screen = json.loads(content.read_text())
    accepted = set(screen['accepted_ids'])
    screens = {row['id']: row for row in screen['rows']}
    rows, rejected = [], []
    for row in read_rows(directory / 'items.jsonl'):
        if row['split'] != part or row['family'] == 'P1u':
            raise ValueError('Training/development partition contains wrong split/test-only family')
        if row['source_audio_sha256'] != screens[row['job_id']]['sha256']:
            raise ValueError('Content decision belongs to a different source waveform')
        if row['job_id'] not in accepted:
            rejected.append({key: row[key] for key in ('pair_id', 'word', 'voice', 'family', 'label')})
            continue
        if row.get('audio') and sha(directory / row['audio']) != row['audio_sha256']:
            raise ValueError('Training waveform identity differs')
        if row['family'] not in {'P1', 'P1w', 'S0', 'S1', 'S2', 'S_original', 'S_world'}:
            raise ValueError('Unregistered control family in fitting')
        if row['label'] != int(row['family'] in {'P1', 'P1w'}):
            raise ValueError('Control label/family mismatch')
        rows.append(row)
    # Validate explicit attempted pair identities, including retained failures.
    by_source = {}
    for row in rows:
        by_source.setdefault(row['pair_id'], []).append(row)
    for identity, group in by_source.items():
        if len({row['family'] for row in group}) != len(group):
            raise ValueError('Duplicate control family for ' + identity)
        if not {'S0', 'P1', 'S_world', 'P1w'} <= {row['family'] for row in group}:
            raise ValueError('Missing mandatory attempted control family for ' + identity)
        if len({(row['word_id'], row['voice'], row['phone_collision_group']) for row in group}) != 1:
            raise ValueError('One pair identity mixes lexical/voice/group identity')
    return rows, rejected, {'directory': str(directory), 'items_sha256': sha(directory / 'items.jsonl'),
                            'content_sha256': sha(content), 'runtime': json.loads((directory / 'runtime.json').read_text())}


def arrays(rows):
    if any(row['status'] != 'measured' or row['features'] is None for row in rows):
        raise ValueError('Only explicitly measured rows may enter a fit matrix')
    return np.stack([row['features'] for row in rows]), np.array([row['label'] for row in rows], float)


def main():
    out = ROOT / 'reports/anchored_training_v71'
    if out.exists():
        raise FileExistsError(out)
    tr_all, tr_rejected, tr_pin = read_partition('train')
    dv_all, dv_rejected, dv_pin = read_partition('development')
    if tr_pin['runtime'] != dv_pin['runtime']:
        raise ValueError('Train/development feature runtimes differ')
    for key in ('word_id', 'phone_collision_group', 'voice'):
        if {row[key] for row in tr_all} & {row[key] for row in dv_all}:
            raise ValueError('Train/development overlap: ' + key)
    tr = [row for row in tr_all if row['status'] == 'measured']
    dv = [row for row in dv_all if row['status'] == 'measured']
    if len({row['pair_id'] for row in tr}) < 100 or len({row['word_id'] for row in tr}) < 70:
        raise ValueError('Insufficient qualified training sources/lexemes for the reviewed run')
    if len({row['voice'] for row in tr}) != 4:
        raise ValueError('The frozen joint-CV protocol requires four training conditioning groups')
    for family in ('S0', 'S2'):
        if sum(row['family'] == family for row in dv) < 30:
            raise ValueError('Insufficient development calibration controls: ' + family)
    write_group(out, {'protocol.json': {
        'training_inputs': {key: value for key, value in tr_pin.items() if key != 'runtime'},
        'development_inputs': {key: value for key, value in dv_pin.items() if key != 'runtime'},
        'model_family': 'Convex linear logistic classifiers only; no word/source/voice/duration/availability/anchor-damage inputs.',
        'candidate_modes': list(INDICES), 'ridges': [.001, .01, .1],
        'cv': 'Train only:3 phonetic-collision folds x4 voice cells. Validation=(groupfold=k AND voice=v); training excludes groupfold=k OR voice=v. All source variants share a cell.',
        'selection': 'Minimum equal-group/source/class weighted pooled out-of-fold BCE, then fewer features, then stronger ridge. Final fit train only. No development early stopping.',
        'calibration': 'Development only: threshold=max(0.5,99th percentile of S0,99th percentile of S2)+1e-5. Hardest predeclared family; do not pool easy controls. Development results are calibration, not accuracy.',
        'post_edit_failures': 'Recorded in attempted datasets and denominator; missing features excluded from fitting only, with counts by label/family. Positive abstentions are misses in test; negative abstentions are not counted correct.',
        'unfiltered_sources': 'Canonical and natural edited requests are unlabeled realization diagnostics, not acceptable-Native/error gold and not used to set threshold.',
        'test': 'Separate test control directory is not read or created here. Open once after external hash freeze.',
        'admission_before_test': {'minimum_test_P1_sources': 80, 'minimum_test_lexemes': 40, 'minimum_test_conditioning_groups': 4,
            'P1_recall': .8, 'P1w_recall': .8, 'S0_false_flag_rate': .025, 'other_retained_vowel_control_false_flag_rate': .05,
            'maximum_abstention_per_family': .05, 'P1u_recall_drop_vs_P1': .1},
        'actual_hindi_error_accuracy': None, 'production_changed': False},
        'source_content_exclusions.json': tr_rejected + dv_rejected,
        'source_snapshot.json': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()} for p in (Path(__file__), ROOT / 'hindi_detector/anchored_detector.py')]},
        'Reviewed large local-deletion fitting. model.pt one selected linear head, train-only normalization, hardest-family dev threshold, full code/resource/encoder/aligner pins. '
        'selection.json all train-only CV candidate losses; cv_predictions.json exact out-of-fold physical scores and membership; fitting_rows.json metadata only; report.json measured/attempted source/family counts; '
        'frozen_selection.json model/threshold/protocol identity before any test open. No epoch checkpoints or raw WAV/encoder copy. Owner Ampixa/user, review2026-11-10; retain selected head/compact evidence while release/test consumes it. '
        'Not on HF; source/MMS research rights inherited. Physical-edit CV/calibration is not real Hindi-error accuracy. Every missing post-edit observation remains in data reports.',
        ['.venv-indicf5/bin/python -m scripts.train_anchored_detector'])
    torch.set_num_threads(2)
    x, y = arrays(tr)
    dx, dy = arrays(dv)
    fold_masks = list(cells(tr))
    candidates = []
    cv_records = []
    for mode in INDICES:
        for ridge in (.001, .01, .1):
            probabilities = np.full(len(tr), np.nan)
            for train_mask, val_mask in fold_masks:
                rows = [row for row, keep in zip(tr, train_mask, strict=True) if keep]
                network, state = fit(x[train_mask], y[train_mask], rows, mode, ridge)
                probabilities[val_mask] = score(network, state, x[val_mask])
            if not np.isfinite(probabilities).all():
                raise ValueError('CV did not score every measured training row exactly once')
            p = np.clip(probabilities, 1e-8, 1 - 1e-8)
            loss = float(np.mean((-y * np.log(p) - (1 - y) * np.log1p(-p)) * weights(tr)))
            candidates.append({'mode': mode, 'ridge': ridge, 'cv_bce': loss, 'feature_count': len(INDICES[mode])})
            cv_records.append({'mode': mode, 'ridge': ridge, 'scores': probabilities.tolist()})
            print(json.dumps(candidates[-1]), flush=True)
    selected = min(candidates, key=lambda row: (row['cv_bce'], row['feature_count'], -row['ridge']))
    # One fixed within-source label permutation is a leakage diagnostic, not
    # another model-selection candidate or a source of threshold tuning.
    rng = np.random.default_rng(710)
    shuffled = [dict(row) for row in tr]
    for identity in sorted({row['pair_id'] for row in shuffled}):
        at = [i for i, row in enumerate(shuffled) if row['pair_id'] == identity]
        labels = rng.permutation([shuffled[i]['label'] for i in at])
        for i, label in zip(at, labels, strict=True):
            shuffled[i]['label'] = int(label)
    shuffled_y = np.array([row['label'] for row in shuffled], float)
    shuffled_p = np.full(len(tr), np.nan)
    for train_mask, val_mask in fold_masks:
        rows = [row for row, keep in zip(shuffled, train_mask, strict=True) if keep]
        network, state = fit(x[train_mask], shuffled_y[train_mask], rows, selected['mode'], selected['ridge'])
        shuffled_p[val_mask] = score(network, state, x[val_mask])
    positives, negatives = shuffled_p[shuffled_y == 1], shuffled_p[shuffled_y == 0]
    shuffled_auc = float(np.mean((positives[:, None] > negatives).astype(float) + .5 * (positives[:, None] == negatives)))
    network, state = fit(x, y, tr, selected['mode'], selected['ridge'])
    dp = score(network, state, dx)
    quantiles = {family: float(np.quantile(dp[np.array([row['family'] == family for row in dv])], .99, method='higher')) for family in ('S0', 'S2')}
    state['threshold'] = max(.5, *quantiles.values()) + .00001
    if not 0 < state['threshold'] < 1:
        raise ValueError('No usable calibrated operating threshold')
    sources = ['hindi_detector/anchored_detector.py', 'hindi_detector/anchored_vowels.py', 'hindi_detector/exact_phone_ctc.py',
        'hindi_detector/ctc.py', 'hindi_detector/context_templates.py', 'hindi_detector/aspirate_context.py', 'hindi_detector/consonant_context.py',
        'hindi_detector/phone_lattice.py', 'hindi_detector/pronunciation_sites.py', 'hindi_detector/dual_runtime.py',
        'hindi_detector/word_evidence.py', 'hindi_detector/dual_g2p.py', 'hindi_detector/anchored_inference.py']
    state['code_sha256'] = {name: sha(ROOT / name) for name in sources}
    state['resource_files'] = {name: sha(ROOT / name) for name in ['data/swar_phone_plan_v18/ipa_map.json', 'data/panphon_context_v58/inventory.json',
        'models/phoneme_probe/inventory.json', 'models/phoneme_probe_lv60_v20/inventory.json', 'models/mms_fa/inventory.json',
        'data/swar_phone_plan_v18/inputs.json', 'data/swar_phone_plan_v18/words.json']}
    state['phone_inventory'], state['secondary_phone_inventory'] = tr_pin['runtime']['phone_inventories']
    state['aligner_inventory'] = tr_pin['runtime']['aligner_inventory']
    state['selected_cv'] = selected
    state['calibration_quantiles'] = quantiles
    state['requires_reference_audio'] = False
    torch.save(state, out / 'model.pt')
    membership = [{key: row[key] for key in ('pair_id', 'phone_collision_group', 'word_id', 'word', 'voice', 'split', 'family', 'label', 'status')} for row in tr_all + dv_all]
    report = {'status': 'complete_reviewed_anchored_fit_before_test', 'model_sha256': sha(out / 'model.pt'),
        'selected': selected, 'threshold': state['threshold'], 'calibration_quantiles': quantiles,
        'training_sources': len({row['pair_id'] for row in tr_all}), 'training_lexemes': len({row['word_id'] for row in tr_all}),
        'training_measured_rows': len(tr), 'training_attempted_rows': len(tr_all), 'calibration_sources': len({row['pair_id'] for row in dv_all}),
        'training_post_edit_abstentions': dict(Counter(row['family'] for row in tr_all if row['status'] != 'measured')),
        'calibration_post_edit_abstentions': dict(Counter(row['family'] for row in dv_all if row['status'] != 'measured')),
        'within_source_label_permutation_cv_auc': shuffled_auc,
        'actual_hindi_error_accuracy': None, 'test_scores_opened': False, 'production_changed': False,
        'coefficients': {name: float(value) for name, value in zip(state['feature_names'], network.weight.detach().numpy()[0], strict=True)}}
    for name, value in [('selection.json', candidates), ('cv_predictions.json', cv_records), ('fitting_rows.json', membership), ('report.json', report),
        ('permutation_diagnostic.json', {'auc': shuffled_auc, 'labels': shuffled_y.tolist(), 'scores': shuffled_p.tolist(), 'seed': 710, 'scope': 'Train-only within-source permutation, not an accuracy estimate.'})]:
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    (out / 'frozen_selection.json').write_text(json.dumps({'created_utc': datetime.now(timezone.utc).isoformat(),
        'model_sha256': sha(out / 'model.pt'), 'protocol_sha256': sha(out / 'protocol.json'), 'threshold': state['threshold'],
        'test_scores_opened': False, 'external_copy_required_before_test': True}, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Reviewed local training failed explicitly; no test open or model promotion')
        raise
