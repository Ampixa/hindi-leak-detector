"""Fit one high-resolution acoustic candidate with strict group/voice CV."""
from __future__ import annotations

from collections import Counter
import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from hindi_detector.acoustic_deletion import AcousticDeletionNetwork, PROCESSING, RECIPE, SCHEMA, pair_indices
from hindi_detector.anchored_detector import cells
from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha

ROOT = Path(__file__).resolve().parents[1]
EPOCHS = 32
FAMILIES = ('S_original', 'S0', 'S1', 'S_world', 'S2', 'P1', 'P1w')


def source_weights(rows):
    masses = np.array([4. if r['family'] == 'S2' else 1. for r in rows])
    totals = Counter()
    groups = {}
    for r, mass in zip(rows, masses, strict=True):
        totals[r['pair_id'], r['label']] += mass
        groups.setdefault(r['phone_collision_group'], set()).add(r['pair_id'])
    w = np.array([m / totals[r['pair_id'], r['label']] / len(groups[r['phone_collision_group']])
                  for r, m in zip(rows, masses, strict=True)], np.float32)
    return w / w.mean()


def rank_loss(logits, rows, global_scales=None):
    pairs = pair_indices(rows)
    if not pairs:
        raise ValueError('Source-packed batch lacks required matched comparisons')
    a, b = (torch.tensor([p[j] for p in pairs], device=logits.device) for j in (0, 1))
    source_group_counts = {}
    for r in rows:
        source_group_counts.setdefault(r['phone_collision_group'], set()).add(r['pair_id'])
    values = ([global_scales[p] for p, _ in pairs] if global_scales is not None else
              [1 / len(source_group_counts[rows[p]['phone_collision_group']]) for p, _ in pairs])
    w = torch.tensor(values,
                     device=logits.device, dtype=logits.dtype)
    if global_scales is None:
        w = w / w.mean()
    return (F.softplus(-(logits[a] - logits[b])) * w).mean()


def normalize(phone, indices):
    mean = phone[indices].mean(0)
    scale = np.maximum(phone[indices].std(0), .05)
    return np.clip((phone - mean) / scale, -8, 8).astype(np.float32), {'mean': mean, 'scale': scale}


def fit_cnn(data, rows, selected, vocabulary_size, progress):
    torch.manual_seed(81)
    rng = np.random.default_rng(81)
    device = 'mps'
    if not torch.backends.mps.is_available():
        raise RuntimeError('Explicit k2 MPS training device unavailable')
    network = AcousticDeletionNetwork(vocabulary_size).to(device)
    phone, scaler = normalize(data['phone'], selected)
    arrays = {k: torch.from_numpy(v).to(device) for k, v in {
        'mel': data['mel'].astype(np.float32), 'context': data['context'],
        'phone': phone, 'labels': data['labels']}.items()}
    tr = [rows[i] for i in selected]
    weights = source_weights(tr)
    weight_lookup = {int(i): float(w) for i, w in zip(selected, weights, strict=True)}
    rank_sources = {}
    for row in tr:
        rank_sources.setdefault(row['phone_collision_group'], set()).add(row['pair_id'])
    rank_values = [1 / len(rank_sources[tr[a]['phone_collision_group']]) for a, _ in pair_indices(tr)]
    rank_normalizer = float(np.mean(rank_values))
    rank_lookup = {int(i): 1 / len(rank_sources[rows[i]['phone_collision_group']]) / rank_normalizer for i in selected}
    sources = {}
    for i in selected:
        sources.setdefault(rows[i]['pair_id'], []).append(int(i))
    identities = sorted(sources)
    optimizer = torch.optim.AdamW(network.parameters(), lr=.001, weight_decay=.01, foreach=False)
    history = []
    started = time.monotonic()
    for epoch in range(EPOCHS):
        network.train()
        order = rng.permutation(len(identities))
        losses = []
        for start in range(0, len(order), 14):
            at = [i for n in order[start:start + 14] for i in sources[identities[n]]]
            ids = torch.tensor(at, device=device)
            current = [rows[i] for i in at]
            optimizer.zero_grad(set_to_none=True)
            logits = network(arrays['mel'][ids], arrays['context'][ids], arrays['phone'][ids])
            w = torch.tensor([weight_lookup[i] for i in at], device=device)
            loss = (F.binary_cross_entropy_with_logits(logits, arrays['labels'][ids], reduction='none') * w).mean()
            loss = loss + .2 * rank_loss(logits, current, [rank_lookup[i] for i in at])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite acoustic objective')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        item = {'epoch': epoch + 1, 'loss': float(np.mean(losses)), 'elapsed_seconds': time.monotonic() - started}
        history.append(item)
        progress(item)
    network.eval().to('cpu')
    return network, scaler, history


def predict_cnn(network, scaler, data, indices):
    predictions = []
    phone = np.clip((data['phone'] - scaler['mean']) / scaler['scale'], -8, 8).astype(np.float32)
    with torch.inference_mode():
        for start in range(0, len(indices), 128):
            at = indices[start:start + 128]
            logits = network(torch.from_numpy(data['mel'][at].astype(np.float32)),
                torch.from_numpy(data['context'][at]), torch.from_numpy(phone[at]))
            predictions.extend(logits.numpy().tolist())
    if not np.isfinite(predictions).all():
        raise ValueError('Nonfinite acoustic validation output')
    return np.array(predictions)


def baseline_values(data, mode, vocabulary_size):
    if mode == 'phone':
        return data['phone']
    if mode == 'duration':
        context = np.eye(vocabulary_size, dtype=np.float32)[data['context']].reshape(len(data['context']), -1)
        return np.concatenate([data['duration'], context], axis=1)
    raise ValueError('Unknown fixed baseline mode')


def fit_phone(data, rows, selected, mode='phone', vocabulary_size=None):
    x, scaler = normalize(baseline_values(data, mode, vocabulary_size), selected)
    x = torch.from_numpy(x[selected].astype(np.float64))
    y = torch.from_numpy(data['labels'][selected].astype(np.float64))
    current = [rows[i] for i in selected]
    w = torch.from_numpy(source_weights(current).astype(np.float64))
    net = nn.Linear(x.shape[1], 1, dtype=torch.float64)
    nn.init.zeros_(net.weight); nn.init.zeros_(net.bias)
    optimizer = torch.optim.LBFGS(net.parameters(), max_iter=100, line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad(set_to_none=True)
        logits = net(x).squeeze(1)
        loss = (F.binary_cross_entropy_with_logits(logits, y, reduction='none') * w).mean()
        loss = loss + .2 * rank_loss(logits, current) + .01 * net.weight.square().sum() / 2
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite phone baseline objective')
        loss.backward()
        return loss
    optimizer.step(closure)
    return net.eval(), scaler


def predict_phone(network, scaler, data, indices, mode='phone', vocabulary_size=None):
    x = np.clip((baseline_values(data, mode, vocabulary_size)[indices] - scaler['mean']) / scaler['scale'], -8, 8).astype(np.float64)
    with torch.inference_mode():
        return network(torch.from_numpy(x)).squeeze(1).numpy()


def operating_point(rows, logits):
    quantiles = {}
    for family in FAMILIES[:5]:
        values = logits[np.array([r['family'] == family for r in rows])]
        if len(values) < 30:
            raise ValueError('Insufficient hard-family calibration observations: ' + family)
        quantiles[family] = float(np.quantile(values, .975 if family == 'S0' else .95, method='higher'))
    # Quantiles commute with sigmoid. Use logits to avoid saturated probabilities
    # destroying the distinction between the highest negative scores.
    return max(0., *quantiles.values()) + .00001, quantiles


def assess(all_rows, rows, logits, threshold, include_strata=True):
    lookup = {(r['pair_id'], r['family']): z for r, z in zip(rows, logits, strict=True)}
    families = {}
    for family in FAMILIES:
        current = [r for r in all_rows if r['family'] == family]
        values = [lookup.get((r['pair_id'], r['family'])) for r in current]
        measured = [v for v in values if v is not None]
        flagged = int(sum(v >= threshold for v in measured))
        families[family] = {'attempted': len(values), 'measured': len(measured), 'flagged': int(flagged),
            'flag_rate': flagged / len(values), 'abstention_rate': 1 - len(measured) / len(values)}
    recall = min(families[f]['flag_rate'] for f in ('P1', 'P1w'))
    gates = {'P1_recall': families['P1']['flag_rate'] >= .8, 'P1w_recall': families['P1w']['flag_rate'] >= .8,
        'S0_false_flag_target': families['S0']['flag_rate'] <= .025,
        'other_retained_false_flag_targets': all(families[f]['flag_rate'] <= .05 for f in FAMILIES[:5] if f != 'S0'),
        'abstentions': all(f['abstention_rate'] <= .05 for f in families.values())}
    strata = {}
    if include_strata:
        for low, high, name in [(40, 60, '40_to_60ms'), (60, 90, '60_to_90ms'), (90, 151, '90ms_and_up')]:
            relevant = [r for r in all_rows if r.get('diagnostic_retained_ms') is not None and low <= r['diagnostic_retained_ms'] < high]
            if not relevant:
                continue
            at = [i for i, r in enumerate(rows) if r.get('diagnostic_retained_ms') is not None and low <= r['diagnostic_retained_ms'] < high]
            stats = assess(relevant, [rows[i] for i in at], logits[at], threshold, False)
            stats.update(sources=len({r['pair_id'] for r in relevant}), groups=len({r['phone_collision_group'] for r in relevant}))
            strata[name] = stats
        short = strata.get('40_to_60ms')
        gates['short_vowel_40_to_60ms'] = bool(short and short['sources'] >= 20 and short['groups'] >= 10 and
            short['minimum_positive_recall'] >= .8 and short['families']['S2']['flag_rate'] <= .05)
    return {'families': families, 'duration_strata': strata, 'minimum_positive_recall': recall, 'gates': gates, 'passed': all(gates.values())}


def read_cache(directory, part, report):
    if sha(directory / (part + '.npz')) != report['partitions'][part]['array_sha256'] or sha(directory / (part + '.index.json')) != report['partitions'][part]['index_sha256']:
        raise ValueError('Acoustic cache bytes changed')
    all_rows = json.loads((directory / (part + '.index.json')).read_text())
    rows = [r for r in all_rows if r['status'] == 'measured']
    if [r['array_index'] for r in rows] != list(range(len(rows))):
        raise ValueError('Acoustic array/index order differs')
    with np.load(directory / (part + '.npz'), allow_pickle=False) as arrays:
        data = {k: arrays[k].copy() for k in ('mel', 'context', 'phone', 'labels', 'duration')}
    if len(data['mel']) != len(rows) or not np.array_equal(data['labels'], [r['label'] for r in rows]):
        raise ValueError('Measured acoustic labels and row identities differ')
    if any(not np.isfinite(v).all() for v in data.values()):
        raise ValueError('Nonfinite acoustic cache')
    return all_rows, rows, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/acoustic_training_v81')
    out = parser.parse_args().out
    directory = ROOT / 'reports/acoustic_features_v81_reviewed'
    if out.exists():
        raise FileExistsError(out)
    report = json.loads((directory / 'report.json').read_text())
    protocol = json.loads((directory / 'protocol.json').read_text())
    if report['status'] != 'complete_train_development_acoustic_cache' or sha(ROOT / 'hindi_detector/acoustic_deletion.py') != protocol['feature_code_sha256']:
        raise ValueError('Acoustic feature recipe incomplete or changed')
    train_all, rows, data = read_cache(directory, 'train', report)
    dev_all, dev, devdata = read_cache(directory, 'development', report)
    for field in ('word_id', 'voice', 'phone_collision_group'):
        if {r[field] for r in train_all} & {r[field] for r in dev_all}:
            raise ValueError('Train/development identity leakage: ' + field)
    if len({r['pair_id'] for r in rows}) < 100 or len({r['word_id'] for r in rows}) < 70:
        raise ValueError('Insufficient source/lexeme training support')
    write_group(out, {'protocol.json': {'schema': SCHEMA, 'recipe': RECIPE, 'epochs': EPOCHS, 'seed': 81,
        'architecture': 'One fixed66->24 temporal CNN three blocks,GroupNorm; global/two observed-focus pools + neighbor phones + local14 ->24hidden ->1.',
        'objective': 'Equal phonetic-group/source/class weighted BCE; S2 gets half the negative-class mass where available;0.2 explicit source-identity matched rank objective.',
        'cv': '12 joint phonetic-groupfold/voice cells from train only. No development early stopping or architecture sweep.',
        'baseline': 'Linear local14 and separate observed-word-duration+neighbor-phone baseline, fixedridge0.01, same source/class weights and matched rank objective, same CV.',
        'processing_base': PROCESSING,
        'processing_limits': 'P1/S0 both use same PCM splice; P1w/S_world/S2 share WORLD base. Removal versus continuous retiming still differ at joins; matching a base does not eliminate splice shortcuts. Real unseen recordings remain necessary.',
        'operating_rule': 'Logit threshold=max(0,S0q97.5,other retained-familyq95)+1e-5, higher quantile; sigmoid equivalent apart from tiny numeric margin. Declare before any fitting.',
        'admission': 'One pooled out-of-fold train threshold, same rule as development. Both trainCV and dev min(P1,P1w) recall>=.8 with S0<=.025,otherretained<=.05,abstention<=.05.40-60ms subset must also have minrecall>=.8 andS2falseflags<=.05 with>=20sources/10groups. CNN candidate benefit requires>=.05 trainCV minrecall improvement over both fixed baselines and positive improvement in40-60ms. Not an isolated proof of raw-acoustic contribution. No test opened here.',
        'fresh_test': 'Oldv71test spent; Swar and v80 remain development/discovery only. New independent test requires new pre-frozen generation after this fit and external hash copy.',
        'actual_hindi_error_accuracy': None, 'source_partitions': report['partitions']},
        'inputs.json': {'cache_protocol_sha256': sha(directory / 'protocol.json'), 'cache_report_sha256': sha(directory / 'report.json'),
            'feature_code_sha256': protocol['feature_code_sha256'], 'trainer_sha256': sha(Path(__file__))}},
        'Actual acoustic deletion training. protocol/inputs.json fixed architecture/objective/heldout rules and feature identities; status.json atomic live fold/epoch/PID; history.json scalar training losses without epoch checkpoints; '
        'cv_predictions.json held-out logits and source/family identities; report.json counts, training-onlyCV and dev calibration/admission, phone/duration comparisons and fallback/abstention strata; model.pt final acoustic candidate and phone normalization, phone_baseline.pt/duration_baseline.pt fixed ablation states; '
        'frozen_selection.json checksums before any new test. No old test or cross-system data read; no live-model change. Physical interventions only, not human Hindi-error gold. Owner Ampixa/user; review2026-11-10, current consumer new candidate/admission. '
        'NOT on HF; inherited source/MMS rights. Keep only final selected acoustic/explicit ablation states for reproduction, no historical epochs; cache/logs disposable24h after consumer closes.',
        ['.venv-indicf5/bin/python -m scripts.train_acoustic_deletion'])
    torch.set_num_threads(2)
    history = {}
    cv_cnn = np.full(len(rows), np.nan)
    cv_phone = np.full(len(rows), np.nan)
    cv_duration = np.full(len(rows), np.nan)
    vocabulary_size = len(protocol['phone_vocabulary'])
    membership = []
    def status(phase, **extra):
        p = out / 'status.partial'
        p.write_text(json.dumps({'phase': phase, 'pid': __import__('os').getpid(), 'updated_utc': datetime.now(timezone.utc).isoformat(), **extra}, indent=2) + '\n')
        p.replace(out / 'status.json')
    for number, (train, valid) in enumerate(cells(rows)):
        selected, validation = np.flatnonzero(train), np.flatnonzero(valid)
        status('training_joint_cv', fold=number + 1, folds=12, epoch=0)
        net, scaler, losses = fit_cnn(data, rows, selected, len(protocol['phone_vocabulary']),
            lambda item: status('training_joint_cv', fold=number + 1, folds=12, **item))
        cv_cnn[validation] = predict_cnn(net, scaler, data, validation)
        linear, lscaler = fit_phone(data, rows, selected)
        cv_phone[validation] = predict_phone(linear, lscaler, data, validation)
        duration, dscaler = fit_phone(data, rows, selected, 'duration', vocabulary_size)
        cv_duration[validation] = predict_phone(duration, dscaler, data, validation, 'duration', vocabulary_size)
        history['fold_' + str(number + 1)] = losses
        val_rows = [rows[i] for i in validation]
        pairs = pair_indices(val_rows)
        concordance = float(np.mean([float(cv_cnn[validation[a]] > cv_cnn[validation[b]]) + .5 * float(cv_cnn[validation[a]] == cv_cnn[validation[b]]) for a, b in pairs]))
        membership.append({'fold': number + 1, 'train_rows': len(selected), 'validation_rows': validation.tolist(), 'paired_concordance': concordance})
        print(json.dumps({'fold': number + 1, 'status': 'complete_joint_cv_fold', 'validation_rows': len(validation)}), flush=True)
    if not all(np.isfinite(v).all() for v in (cv_cnn, cv_phone, cv_duration)):
        raise ValueError('Cross-validation omitted measured rows')
    cnn_op, _ = operating_point(rows, cv_cnn)
    phone_op, _ = operating_point(rows, cv_phone)
    cv = assess(train_all, rows, cv_cnn, cnn_op)
    baseline_cv = assess(train_all, rows, cv_phone, phone_op)
    duration_op, _ = operating_point(rows, cv_duration)
    duration_cv = assess(train_all, rows, cv_duration, duration_op)
    status('training_final_train_only', epoch=0)
    selected = np.arange(len(rows))
    net, scaler, history['final'] = fit_cnn(data, rows, selected, len(protocol['phone_vocabulary']),
        lambda item: status('training_final_train_only', **item))
    linear, lscaler = fit_phone(data, rows, selected)
    duration, dscaler = fit_phone(data, rows, selected, 'duration', vocabulary_size)
    dz = predict_cnn(net, scaler, devdata, np.arange(len(dev)))
    bz = predict_phone(linear, lscaler, devdata, np.arange(len(dev)))
    duz = predict_phone(duration, dscaler, devdata, np.arange(len(dev)), 'duration', vocabulary_size)
    threshold, quantiles = operating_point(dev, dz)
    baseline_threshold, _ = operating_point(dev, bz)
    development = assess(dev_all, dev, dz, threshold)
    baseline_dev = assess(dev_all, dev, bz, baseline_threshold)
    duration_threshold, _ = operating_point(dev, duz)
    duration_dev = assess(dev_all, dev, duz, duration_threshold)
    benefit = cv['minimum_positive_recall'] - max(baseline_cv['minimum_positive_recall'], duration_cv['minimum_positive_recall'])
    short_gain = cv['duration_strata']['40_to_60ms']['minimum_positive_recall'] - max(baseline_cv['duration_strata']['40_to_60ms']['minimum_positive_recall'], duration_cv['duration_strata']['40_to_60ms']['minimum_positive_recall'])
    ready = cv['passed'] and development['passed'] and benefit >= .05 and short_gain > 0
    code = {name: sha(ROOT / name) for name in ['hindi_detector/acoustic_deletion.py', 'hindi_detector/anchored_detector.py',
        'hindi_detector/anchored_vowels.py', 'hindi_detector/exact_phone_ctc.py', 'hindi_detector/context_templates.py',
        'hindi_detector/consonant_context.py', 'hindi_detector/aspirate_context.py', 'hindi_detector/ctc.py',
        'hindi_detector/phone_lattice.py', 'hindi_detector/pronunciation_sites.py', 'hindi_detector/dual_runtime.py',
        'hindi_detector/word_evidence.py', 'hindi_detector/dual_g2p.py']}
    resources = {name: sha(ROOT / name) for name in ['data/swar_phone_plan_v18/ipa_map.json', 'data/swar_phone_plan_v18/inputs.json',
        'data/swar_phone_plan_v18/words.json', 'data/panphon_context_v58/inventory.json', 'models/phoneme_probe/inventory.json',
        'models/phoneme_probe/config.json', 'models/phoneme_probe_lv60_v20/inventory.json', 'models/phoneme_probe_lv60_v20/config.json', 'models/mms_fa/inventory.json']}
    source_runtime = json.loads((directory / 'train.inputs.json').read_text())['qualified_source_inputs']['runtime']
    state = {'schema': SCHEMA, 'weights': net.state_dict(), 'scaler': scaler, 'recipe': RECIPE,
        'phone_vocabulary': protocol['phone_vocabulary'], 'phone_configurations': protocol['phone_configurations'],
        'logit_threshold': threshold, 'code_sha256': code, 'resource_files': resources,
        'phone_inventory': source_runtime['phone_inventories'][0], 'secondary_phone_inventory': source_runtime['phone_inventories'][1],
        'aligner_inventory': source_runtime['aligner_inventory'], 'software': {'torch': torch.__version__, 'numpy': np.__version__},
        'actual_hindi_error_accuracy': None, 'cause': 'internal_vowel_deletion'}
    torch.save(state, out / 'model.pt')
    torch.save({'weights': linear.state_dict(), 'scaler': lscaler, 'logit_threshold': baseline_threshold, 'purpose': 'Fixed fair phone-only ablation, not deployed'}, out / 'phone_baseline.pt')
    torch.save({'weights': duration.state_dict(), 'scaler': dscaler, 'logit_threshold': duration_threshold, 'purpose': 'Fixed duration/context ablation, not deployed'}, out / 'duration_baseline.pt')
    result = {'status': 'complete_acoustic_fit_before_any_new_test', 'parameters': sum(p.numel() for p in net.parameters()),
        'training_sources': len({r['pair_id'] for r in rows}), 'training_lexemes': len({r['word_id'] for r in rows}),
        'training_measured_rows': len(rows), 'development_sources': len({r['pair_id'] for r in dev}),
        'cv': cv, 'phone_baseline_cv': baseline_cv, 'development_calibration': development, 'phone_baseline_development': baseline_dev,
        'duration_baseline_cv': duration_cv, 'duration_baseline_development': duration_dev,
        'acoustic_cv_min_recall_improvement': benefit, 'ready_for_new_independent_test': ready,
        'short_vowel_cv_min_recall_improvement': short_gain,
        'query_hint_fallback_rows_by_family': dict(Counter(r['family'] for r in rows if r.get('diagnostic_query_hints', {}).get('native_hint_fallback_windows', 0))),
        'logit_threshold': threshold, 'calibration_quantiles': quantiles, 'model_sha256': sha(out / 'model.pt'),
        'actual_hindi_error_accuracy': None, 'production_changed': False, 'new_test_opened': False}
    for name, value in [('history.json', history), ('report.json', result), ('cv_predictions.json', {'rows': rows, 'acoustic_logits': cv_cnn.tolist(), 'phone_logits': cv_phone.tolist(), 'membership': membership}),
        ('development_predictions.json', {'rows': dev, 'acoustic_logits': dz.tolist(), 'phone_logits': bz.tolist()})]:
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    (out / 'frozen_selection.json').write_text(json.dumps({'created_utc': datetime.now(timezone.utc).isoformat(),
        'model_sha256': sha(out / 'model.pt'), 'protocol_sha256': sha(out / 'protocol.json'), 'ready_for_new_test': ready,
        'new_test_opened': False, 'external_copy_required_before_new_test': True}, indent=2) + '\n')
    status('complete_acoustic_fit', ready_for_new_independent_test=ready)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Acoustic training failed explicitly; no fallback or live model switch')
        arguments = argparse.ArgumentParser()
        arguments.add_argument('--out', type=Path, default=ROOT / 'reports/acoustic_training_v81')
        p = arguments.parse_args().out
        if p.exists():
            (p / 'failure.json').write_text(json.dumps({'created_utc': datetime.now(timezone.utc).isoformat(), 'traceback': traceback.format_exc()}, indent=2) + '\n')
        raise
