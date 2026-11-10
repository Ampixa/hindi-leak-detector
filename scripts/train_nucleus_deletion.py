"""Fixed loss-localized acoustic experiment, with a matched no-audio ablation."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from hindi_detector.acoustic_deletion import pair_indices
from hindi_detector.anchored_detector import cells
from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.nucleus_deletion import NucleusDeletionNetwork, PhoneContextNetwork, SCHEMA
from scripts.train_acoustic_deletion import assess, normalize, operating_point, predict_cnn, read_cache, source_weights, rank_loss

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/nucleus_training_v82'
EPOCHS = 32


def fit(data, rows, selected, vocabulary_size, acoustic, progress):
    torch.manual_seed(82)
    rng = np.random.default_rng(82)
    if not torch.backends.mps.is_available():
        raise RuntimeError('Required k2 MPS unavailable; no device substitution')
    device = 'mps'
    network = (NucleusDeletionNetwork if acoustic else PhoneContextNetwork)(vocabulary_size).to(device)
    phone, scaler = normalize(data['phone'], selected)
    keys = ('mel', 'context', 'labels') + (('targets', 'masks') if acoustic else ())
    arrays = {k: torch.from_numpy(data[k]).to(device) for k in keys}
    arrays['phone'] = torch.from_numpy(phone).to(device)
    current = [rows[i] for i in selected]
    weights = source_weights(current)
    lookup = {int(i): float(w) for i, w in zip(selected, weights, strict=True)}
    # Each auxiliary target class has exactly half the total source-weighted
    # mass. Normalize per row so short nuclei are not downweighted by length.
    classes = np.array([float(data['targets'][i, 0]) for i in selected]) if acoustic else None
    aux_lookup = {}
    if acoustic:
        for c in (0., 1.):
            mass = float(weights[classes == c].sum())
            if mass <= 0:
                raise ValueError('Missing auxiliary target class')
            for i, w, label in zip(selected, weights, classes, strict=True):
                if label == c:
                    aux_lookup[int(i)] = float(w * len(selected) / (2 * mass))
    groups, sources = {}, {}
    for i in selected:
        groups.setdefault(rows[i]['phone_collision_group'], set()).add(rows[i]['pair_id'])
        sources.setdefault(rows[i]['pair_id'], []).append(int(i))
    normalizer = float(np.mean([1 / len(groups[current[a]['phone_collision_group']]) for a, _ in pair_indices(current)]))
    rank = {int(i): 1 / len(groups[rows[i]['phone_collision_group']]) / normalizer for i in selected}
    identities = sorted(sources)
    optimizer = torch.optim.AdamW(network.parameters(), lr=.001, weight_decay=.01, foreach=False)
    history, started = [], time.monotonic()
    for epoch in range(EPOCHS):
        network.train()
        order, losses, auxiliaries = rng.permutation(len(identities)), [], []
        for start in range(0, len(order), 14):
            at = [i for n in order[start:start + 14] for i in sources[identities[n]]]
            ids = torch.tensor(at, device=device)
            optimizer.zero_grad(set_to_none=True)
            inputs = [arrays[k][ids] for k in ('mel', 'context', 'phone')]
            if acoustic:
                logits, dense = network(*inputs, auxiliary=True)
                frame = F.binary_cross_entropy_with_logits(dense, arrays['targets'][ids], reduction='none')
                masked = (frame * arrays['masks'][ids]).sum(1) / arrays['masks'][ids].sum(1)
                aux = (masked * torch.tensor([aux_lookup[i] for i in at], device=device)).mean()
            else:
                logits = network(*inputs)
                aux = torch.tensor(0., device=device)
            w = torch.tensor([lookup[i] for i in at], device=device)
            loss = (F.binary_cross_entropy_with_logits(logits, arrays['labels'][ids], reduction='none') * w).mean()
            loss = loss + .2 * rank_loss(logits, [rows[i] for i in at], [rank[i] for i in at]) + .5 * aux
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite localized acoustic objective')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach().cpu())); auxiliaries.append(float(aux.detach().cpu()))
        item = {'epoch': epoch + 1, 'loss': float(np.mean(losses)), 'auxiliary_loss': float(np.mean(auxiliaries)),
                'elapsed_seconds': time.monotonic() - started}
        if acoustic:
            item['vowel_veto_strength'] = float(F.softplus(network.vowel_veto).detach().cpu())
        history.append(item); progress(item)
    return network.eval().cpu(), scaler, history


def load_targets(data, part, cache, targets):
    report = json.loads((targets / 'report.json').read_text())['partitions'][part]
    if sha(targets / (part + '.npz')) != report['sha256'] or sha(cache / (part + '.index.json')) != report['input_index_sha256']:
        raise ValueError('Loss target/input identity changed')
    with np.load(targets / (part + '.npz'), allow_pickle=False) as z:
        data.update({k: z[k].copy() for k in ('targets', 'masks')})
    if data['targets'].shape != (len(data['labels']), 128) or data['masks'].shape != data['targets'].shape:
        raise ValueError('Loss target/input dimensions differ')
    if not np.isin(data['targets'], [0., 1.]).all() or not np.isfinite(data['masks']).all() or np.any(data['masks'] < 0) or np.any(data['masks'] > 1) or np.any(data['masks'].sum(1) <= 0):
        raise ValueError('Invalid loss-only targets')


def dense_diagnostic(network, scaler, data, indices):
    """OOF only: known-region vowel scores; not an inference input."""
    values = []
    phone = np.clip((data['phone'] - scaler['mean']) / scaler['scale'], -8, 8).astype(np.float32)
    with torch.inference_mode():
        for start in range(0, len(indices), 128):
            at = indices[start:start + 128]
            _, dense = network(torch.from_numpy(data['mel'][at]), torch.from_numpy(data['context'][at]), torch.from_numpy(phone[at]), auxiliary=True)
            mask = torch.from_numpy(data['masks'][at])
            values.extend(((dense * mask).sum(1) / mask.sum(1)).tolist())
    return np.array(values)


def main():
    cache = ROOT / 'reports/acoustic_features_v81_reviewed'
    targets = ROOT / 'reports/nucleus_targets_v82_reviewed'
    if OUT.exists():
        raise FileExistsError(OUT)
    cache_report = json.loads((cache / 'report.json').read_text())
    protocol = json.loads((cache / 'protocol.json').read_text())
    aux_protocol = json.loads((targets / 'protocol.json').read_text())
    if sha(ROOT / 'hindi_detector/acoustic_deletion.py') != protocol['feature_code_sha256'] or sha(ROOT / 'hindi_detector/nucleus_deletion.py') != aux_protocol['target_builder_sha256']:
        raise ValueError('Feature/target source differs from prepared bytes')
    all_train, rows, data = read_cache(cache, 'train', cache_report)
    all_dev, dev, devdata = read_cache(cache, 'development', cache_report)
    for key in ('word_id', 'voice', 'phone_collision_group'):
        if {r[key] for r in rows} & {r[key] for r in dev}:
            raise ValueError('Train/development leakage: ' + key)
    load_targets(data, 'train', cache, targets)
    # Development targets are never loaded into fitting arrays or a loss.
    write_group(OUT, {'protocol.json': {
        'schema': SCHEMA, 'epochs': EPOCHS, 'seed': 82,
        'architecture': 'Audio-only64mel temporal24x3 -> dense vowel logits. Observed gap hint only pools local dense.detach() values; nonnegative learned veto subtracts from phone14/context MLP. No global acoustic pool, no hint channels/context in dense head.',
        'objective': 'v81 equal-group/source/class BCE +0.2matched ranking. Aux0.5, balanced50/50 retained/absent target classes; masked known-region BCE normalized per row. Only aux loss trains dense head. Dev targets never enter loss.',
        'comparison': 'Fixed matched phone14/context MLP with identical word objective/splits/epochs; no acoustic input. Previous v81 linear/duration diagnostics reference only.',
        'evaluation': 'Same12 joint voice/group CV,32fixedepochs,train-only normalization. Fixed operating rule max(0,S0q97.5,otherretainedq95)+1e-5. No test or Swar/v80 read. Both CV/dev must pass P1/P1w>=.8,S0<=.025,otherretained<=.05,abstain<=.05 and40-60ms>=20sources10groups,minrecall>=.8,S2FPR<=.05.',
        'attribution': 'Readiness absolute task gates separate from mechanistic claim: >=.05 CV minrecall gain over matched no-audio arm and positive short-vowel gain, nonzero veto. P1w-only improvement not sufficient. Physical controlled results do not establish Hindi-error accuracy.',
        'processing_limit': 'Splice artifacts remain possible; OOF retained S0-original dense differences reported, no human phoneme-boundary gold.',
        'test': 'Oldv71test spent; generate fresh independent data only after immutable external candidate hash copy.',
        'actual_hindi_error_accuracy': None},
        'inputs.json': {'cache_report_sha256': sha(cache / 'report.json'), 'target_report_sha256': sha(targets / 'report.json'),
            'trainer_sha256': sha(Path(__file__)), 'nucleus_code_sha256': sha(ROOT / 'hindi_detector/nucleus_deletion.py'),
            'v81_helper_sha256': sha(ROOT / 'scripts/train_acoustic_deletion.py')}},
        'Localized vowel supervision experiment. protocol/inputs fixed fit rules and pins; status.json atomic live fold/epoch/PID; history.json scalar losses/veto strengths; cv_predictions.json heldout rows/logits/dense known-region diagnostic; report.json absolute gates and ablation; development_predictions.json calibration only; model.pt and no_audio.pt final reproducible heads; frozen_selection.json immutable candidate hashes. No fold/epoch checkpoints. '
        'Owner Ampixa/user; active consumer v82 admission and regression work. Review2026-11-10. NOT on HF; inherited research rights. Keep compact configs/results/final selected and explicit ablation heads; diagnostic arrays disposable24h after consumer closes. No new audio or encoders.',
        ['.venv-indicf5/bin/python -m scripts.train_nucleus_deletion'])
    torch.set_num_threads(2)
    def status(phase, **extra):
        path = OUT / 'status.partial'
        path.write_text(json.dumps({'phase': phase, 'pid': os.getpid(), 'updated_utc': datetime.now(timezone.utc).isoformat(), **extra}) + '\n')
        path.replace(OUT / 'status.json')
    size = len(protocol['phone_vocabulary'])
    outputs = {k: np.full(len(rows), np.nan) for k in ('acoustic', 'no_audio', 'dense')}
    history, folds = {}, []
    for number, (train, valid) in enumerate(cells(rows), 1):
        selected, validation = np.flatnonzero(train), np.flatnonzero(valid)
        entry = {'fold': number, 'validation': validation.tolist()}
        for acoustic, arm in ((True, 'acoustic'), (False, 'no_audio')):
            network, scaler, losses = fit(data, rows, selected, size, acoustic,
                lambda item: status('joint_cv', fold=number, arm=arm, **item))
            history[f'{number}_{arm}'] = losses
            outputs[arm][validation] = predict_cnn(network, scaler, data, validation)
            if acoustic:
                outputs['dense'][validation] = dense_diagnostic(network, scaler, data, validation)
                entry['vowel_veto_strength'] = losses[-1]['vowel_veto_strength']
        folds.append(entry)
        print(json.dumps({'fold': number, 'status': 'completed_both_arms'}), flush=True)
    if not all(np.isfinite(v).all() for v in outputs.values()):
        raise ValueError('OOF coverage incomplete/nonfinite')
    cv = {}
    for arm in ('acoustic', 'no_audio'):
        threshold, _ = operating_point(rows, outputs[arm])
        cv[arm] = assess(all_train, rows, outputs[arm], threshold)
    development, final, dzs = {}, {}, {}
    for acoustic, arm in ((True, 'acoustic'), (False, 'no_audio')):
        network, scaler, losses = fit(data, rows, np.arange(len(rows)), size, acoustic,
            lambda item: status('final_train_only', arm=arm, **item))
        history['final_' + arm] = losses
        dz = predict_cnn(network, scaler, devdata, np.arange(len(dev)))
        threshold, quantiles = operating_point(dev, dz)
        development[arm] = assess(all_dev, dev, dz, threshold)
        final[arm] = (network, scaler, threshold, quantiles)
        dzs[arm] = dz.tolist()
    ready = bool(cv['acoustic']['passed'] and development['acoustic']['passed'])
    benefit = cv['acoustic']['minimum_positive_recall'] - cv['no_audio']['minimum_positive_recall']
    short_gain = cv['acoustic']['duration_strata']['40_to_60ms']['minimum_positive_recall'] - cv['no_audio']['duration_strata']['40_to_60ms']['minimum_positive_recall']
    network, scaler, threshold, quantiles = final['acoustic']
    old = torch.load(ROOT / 'reports/acoustic_training_v81_recovery/model.pt', map_location='cpu', weights_only=False)
    state = {k: old[k] for k in ('recipe', 'phone_vocabulary', 'phone_configurations', 'code_sha256', 'resource_files', 'phone_inventory', 'secondary_phone_inventory', 'aligner_inventory', 'software')}
    state['code_sha256'] = {**state['code_sha256'], 'hindi_detector/nucleus_deletion.py': sha(ROOT / 'hindi_detector/nucleus_deletion.py')}
    state.update(schema=SCHEMA, weights=network.state_dict(), scaler=scaler, logit_threshold=threshold,
                 actual_hindi_error_accuracy=None, cause='internal_vowel_deletion', target_report_sha256=sha(targets / 'report.json'))
    torch.save(state, OUT / 'model.pt')
    bnet, bscale, bthreshold, _ = final['no_audio']
    torch.save({'weights': bnet.state_dict(), 'scaler': bscale, 'logit_threshold': bthreshold, 'purpose': 'Matched no-audio ablation, not deployed'}, OUT / 'no_audio.pt')
    pairs = {(r['pair_id'], r['family']): i for i, r in enumerate(rows)}
    sham = [float(outputs['dense'][i] - outputs['dense'][pairs[r['pair_id'], 'S_original']]) for i, r in enumerate(rows) if r['family'] == 'S0']
    result = {'status': 'complete_localized_supervision_fit', 'parameters': sum(p.numel() for p in network.parameters()),
        'training_sources': len({r['pair_id'] for r in rows}), 'training_lexemes': len({r['word_id'] for r in rows}),
        'training_rows': len(rows), 'development_sources': len({r['pair_id'] for r in dev}),
        'cv': cv, 'development_calibration': development, 'ready_for_new_independent_test': ready,
        'cv_minimum_recall_gain_over_no_audio': benefit, 'short_vowel_cv_gain_over_no_audio': short_gain,
        'mechanistic_gain_criterion_met': bool(benefit >= .05 and short_gain > 0),
        'vowel_veto_strength': history['final_acoustic'][-1]['vowel_veto_strength'],
        'oof_sham_minus_original_known_region_dense_logit': {'sources': len(sham), 'median': float(np.median(sham)), 'absolute_q95': float(np.quantile(np.abs(sham), .95))},
        'hint_fallback_rows_by_family': dict(Counter(r['family'] for r in rows if r.get('diagnostic_query_hints', {}).get('native_hint_fallback_windows', 0))),
        'model_sha256': sha(OUT / 'model.pt'), 'logit_threshold': threshold, 'calibration_quantiles': quantiles,
        'actual_hindi_error_accuracy': None, 'production_changed': False, 'new_test_opened': False}
    artifacts = {'history.json': history, 'report.json': result, 'cv_predictions.json': {'rows': rows, 'logits': {k: v.tolist() for k, v in outputs.items()}, 'folds': folds},
        'development_predictions.json': {'rows': dev, 'logits': dzs},
        'frozen_selection.json': {'created_utc': datetime.now(timezone.utc).isoformat(), 'model_sha256': sha(OUT / 'model.pt'), 'protocol_sha256': sha(OUT / 'protocol.json'), 'ready_for_new_test': ready, 'new_test_opened': False}}
    for name, value in artifacts.items():
        (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    status('complete', ready_for_new_independent_test=ready)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Localized acoustic fit failed explicitly; production unchanged')
        if OUT.exists():
            (OUT / 'failure.json').write_text(json.dumps({'created_utc': datetime.now(timezone.utc).isoformat(), 'traceback': traceback.format_exc()}) + '\n')
        raise
