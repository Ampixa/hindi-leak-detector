"""Fit full-region deletion controls; freeze low-data quality head unchanged."""
from __future__ import annotations

from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
from pathlib import Path

import numpy as np
import torch

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.dual_paired_detector import DualPairedNetwork, select_inputs
from hindi_detector.dual_paired_vowels import FEATURE_NAMES, supported
from hindi_detector.evaluation import binary_metrics
from hindi_detector.task_vowel_detector import decisions, load, score
from scripts.extract_corrected_training import read_rows
from scripts.train_task_vowels import head_prob

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [('cross_voice', 'full_context_controls_v60', 'full_context_content_v60'),
           ('self_backoff', 'full_context_self_v60b', 'full_context_self_content_v60b')]


def digest_weights(head):
    digest = hashlib.sha256()
    for name, tensor in sorted(head['weights'].items()):
        digest.update(name.encode())
        digest.update(tensor.cpu().numpy().tobytes())
    return digest.hexdigest()


def gather():
    records, exclusions, inputs = [], [], []
    for domain, control_name, content_name in SOURCES:
        directory, content = ROOT / 'reports' / control_name, ROOT / 'reports' / content_name
        screen = json.loads((content / 'result.json').read_text())
        receipts = {row['id']: row for row in screen['rows']}
        accepted = set(screen['accepted_ids'])
        if accepted != {key for key, row in receipts.items() if row['decision'] == 'pass'}:
            raise ValueError('Source content acceptance list differs from exact ASR decisions')
        source = read_rows(directory / 'items.jsonl')
        if len(source) % 2:
            raise ValueError('Physical data contains incomplete pairs')
        inputs.append({'directory': str(directory), 'items_sha256': sha(directory / 'items.jsonl'),
                       'content_sha256': sha(content / 'result.json')})
        for negative, positive in zip(source[::2], source[1::2], strict=True):
            if (negative['label'], positive['label']) != (0, 1) or any(negative[k] != positive[k] for k in ('word', 'job_id', 'reference_job_id', 'split', 'kind')):
                raise ValueError('Physical pair identities/labels differ')
            needed = {negative['job_id'], negative['reference_job_id']}
            if negative['donor_job_id'] is not None:
                needed.add(negative['donor_job_id'])
            if not needed <= accepted:
                exclusions.append({'job_id': negative['job_id'], 'word': negative['word'],
                    'reason': 'At least one canonical/reference/edited-donor source failed content pass',
                    'decisions': {key: receipts[key]['decision'] for key in needed}})
                continue
            if receipts[negative['job_id']]['sha256'] != negative['source_audio_sha256'] or receipts[negative['reference_job_id']]['sha256'] != negative['reference_audio_sha256']:
                raise ValueError('Content audit belongs to a different canonical/reference waveform')
            if negative['donor_job_id'] is not None and receipts[negative['donor_job_id']]['sha256'] != negative['donor']['sha256']:
                raise ValueError('Content audit belongs to a different donor waveform')
            for mode in negative['modes']:
                for row in (negative, positive):
                    if sha(directory / row['audio']) != row['audio_sha256']:
                        raise ValueError('Physical fitting waveform changed')
                    values = np.asarray(row['modes'][mode]['features'], np.float32)
                    if values.shape != (114,) or not np.isfinite(values).all():
                        raise ValueError('Physical context feature dimensions/values differ')
                    records.append({'word': row['word'], 'word_id': row['word_id'], 'split': row['split'], 'kind': row['kind'],
                        'label': row['label'], 'mode': mode, 'domain': domain, 'pair_id': row['job_id'],
                        'voice': row['voice'], 'reference_voice': row['reference_voice'], 'features': values})
    train_words = {row['word'] for row in records if row['split'] == 'train'}
    dev_words = {row['word'] for row in records if row['split'] == 'development'}
    if train_words & dev_words or {'हैन', 'किनकि'} & (train_words | dev_words):
        raise ValueError('Word families cross training/development or known user regression exclusion')
    return records, exclusions, inputs


def summary_metrics(state, networks, rows):
    result = {}
    for kind in ('vowel_deletion', 'vowel_quality'):
        subset = [row for row in rows if row['kind'] == kind]
        if not subset:
            continue
        x = np.stack([row['features'] for row in subset])
        y = np.array([row['label'] for row in subset])
        p = score(networks, state, x)
        ds = np.array(decisions(state, x, p))
        per_mode = {}
        for mode in sorted({row['mode'] for row in subset}):
            mask = np.array([row['mode'] == mode for row in subset])
            per_mode[mode] = binary_metrics(y[mask], (ds[mask] == 'strong_vowel_change').astype(float), .5)
        result[kind] = {'base_pairs': len({row['pair_id'] for row in subset}), 'word_families': len({row['word'] for row in subset}),
                        'per_mode': per_mode, 'mask_rows_are_independent': False}
    return result


def main():
    out = ROOT / 'reports/context_fallback_v61'
    if out.exists():
        raise FileExistsError(out)
    records, exclusions, inputs = gather()
    original_model = ROOT / 'reports/task_vowel_release_v31/model.pt'
    original_networks, original = load(original_model)
    write_group(out, {'protocol.json': {'inputs': inputs, 'old_head_sha256': sha(original_model),
        'recipe': 'Full observed vowel edits and same-source-voice sham; cross-conditioning-voice reference when qualified, separately tagged self backoff otherwise.',
        'selection': 'Per-kind train/dev only; equal word/reference-domain BCE, paired ranking. Context masks are repeated views, never new independent samples.',
        'candidate_modes': ['primary57', 'secondary57', 'dual114'], 'architectures': ['linear', 'mlp16'],
        'data_floor_for_refitting': 'At least30 training base pairs,8 development base pairs and6 development word families per cause; otherwise freeze existing cause head exactly.',
        'threshold': 'Refitted head max(.5, highest eligible development-negative score across all context masks+.00001). No reserved data are loaded.',
        'quality_policy': 'The observed small quality dataset cannot justify refitting; preserve the selected release quality head.',
        'evaluation': 'Existing v59 stress data are retrospective after method development. Newly rendered v62 unsupported-consonant words remain held out and unopened here.',
        'actual_hindi_error_accuracy': None}, 'source_exclusions.json': exclusions,
        'source_snapshot.json': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()}
                               for p in (Path(__file__), ROOT / 'hindi_detector/context_detector.py')]},
        'Research context fallback checkpoint on lexical-content-passed full-vowel physical '
        'changes. model.pt selected heads/train-only normalization/thresholds and code/class '
        'resource pins; quality head remains bit-exact to release31 if its data floor fails. '
        'protocol.json exact source/selection recipe; source_exclusions.json all failed-source '
        'pairs; selection.json dev candidates, history.json optimizer traces, report.json '
        'independent base-pair/family counts and per-mask diagnostics; fitting_rows.json exact '
        'word/fold/domain/mask membership without duplicate arrays. frozen_selection.json is '
        'written before any v62 score loading. No raw voice/source/word IDs as features. '
        'Not automatically promoted to production; no real Hindi-error labels. Current '
        'consumer broader unsupported-context admission, review2026-11-10. Retain selected '
        'checkpoint/compact evidence while consumed; no epoch checkpoints/encoder/WAV copy '
        'or HF upload. Source/MMS/PanPhon licences apply.',
        ['.venv-indicf5/bin/python -m scripts.train_context_fallback'])
    train = [row for row in records if row['split'] == 'train']
    development = [row for row in records if row['split'] == 'development']
    heads, networks, selection, history = {}, {}, {}, {}
    torch.set_num_threads(2)
    for kind in ('vowel_deletion', 'vowel_quality'):
        tr = [row for row in train if row['kind'] == kind]
        dr = [row for row in development if row['kind'] == kind]
        counts = {'train_pairs': len({row['pair_id'] for row in tr}), 'development_pairs': len({row['pair_id'] for row in dr}),
                  'train_words': len({row['word'] for row in tr}), 'development_words': len({row['word'] for row in dr})}
        if not tr or not dr:
            raise ValueError('No usable source-passed train/dev data for ' + kind)
        x, dx = [np.stack([row['features'] for row in part]) for part in (tr, dr)]
        y, dy = [np.array([row['label'] for row in part], np.float32) for part in (tr, dr)]
        original_head = copy.deepcopy(original['heads'][kind])
        floor = counts['train_pairs'] >= 30 and counts['development_pairs'] >= 8 and counts['development_words'] >= 6
        # Quality needs an independently sufficient training pool; never select
        # a small, apparently perfect head merely because fitting is available.
        if kind == 'vowel_quality' or not floor:
            heads[kind] = original_head
            networks[kind] = original_networks[kind]
            selection[kind] = {'selected': 'frozen_release_v31', 'counts': counts, 'refitted': False,
                               'weights_sha256': digest_weights(original_head), 'reason': 'Prespecified data adequacy floor/quality freeze'}
            continue
        frequencies = Counter((row['domain'], row['word']) for row in tr)
        words_per_domain = Counter(domain for domain, _ in frequencies)
        w = torch.tensor([1 / (frequencies[row['domain'], row['word']] * words_per_domain[row['domain']]) for row in tr], dtype=torch.float32)
        w /= w.mean()
        dev_frequencies = Counter((row['domain'], row['word']) for row in dr)
        dev_words_per_domain = Counter(domain for domain, _ in dev_frequencies)
        dw = torch.tensor([1 / (dev_frequencies[row['domain'], row['word']] * dev_words_per_domain[row['domain']]) for row in dr], dtype=torch.float32)
        dw /= dw.mean()
        yt, dyt = torch.from_numpy(y), torch.from_numpy(dy)
        candidates = []
        for mode in ('primary57', 'secondary57', 'dual114'):
            values, dvalues = select_inputs(x, mode), select_inputs(dx, mode)
            mean, scale = values.mean(0), np.maximum(values.std(0), .05)
            tx, dt = [torch.from_numpy(np.clip((value - mean) / scale, -8, 8).astype(np.float32).copy()) for value in (values, dvalues)]
            for architecture in ('linear', 'mlp16'):
                torch.manual_seed(20261109)
                network = DualPairedNetwork(architecture, mode)
                optimizer = torch.optim.AdamW(network.parameters(), lr=.01, weight_decay=.1)
                best, stale, trace = None, 0, []
                for epoch in range(400):
                    network.train()
                    optimizer.zero_grad(set_to_none=True)
                    logits = network(tx)
                    loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, yt, reduction='none') * w).mean()
                    loss += .25 * (torch.nn.functional.softplus(-(logits[1::2] - logits[::2])) * w[::2]).mean()
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite context fallback objective')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 5, error_if_nonfinite=True)
                    optimizer.step()
                    network.eval()
                    with torch.inference_mode():
                        value = float((torch.nn.functional.binary_cross_entropy_with_logits(network(dt), dyt, reduction='none') * dw).mean())
                    trace.append({'epoch': epoch + 1, 'loss': float(loss.detach()), 'dev_bce': value})
                    if best is None or value < best['loss'] - .0001:
                        best = {'loss': value, 'epoch': epoch + 1, 'weights': {k: v.detach().clone() for k, v in network.state_dict().items()}}
                        stale = 0
                    else:
                        stale += 1
                    if stale >= 40:
                        break
                network.load_state_dict(best['weights'], strict=True)
                head = {'feature_mode': mode, 'architecture': architecture, 'weights': best['weights'],
                    'mean': mean, 'scale': scale, 'best_epoch': best['epoch'], 'origin': 'full_region_context_fit_v61'}
                name = mode + '/' + architecture
                candidates.append((best['loss'], name, network.eval(), head))
                history[kind + '/' + name] = trace
        op = np.clip(head_prob(original_networks[kind], original_head, dx), 1e-7, 1 - 1e-7)
        old_loss = float(np.mean((-dy * np.log(op) - (1 - dy) * np.log1p(-op)) * dw.numpy()))
        candidates.append((old_loss, 'frozen_release_v31', original_networks[kind], original_head))
        loss, name, network, head = min(candidates, key=lambda value: (value[0], value[1]))
        p = head_prob(network, head, dx)
        negatives = p[dy == 0].copy()
        negatives[~supported(dx)[dy == 0]] = -1
        high = max(.5, float(negatives.max()) + .00001)
        head['operating_points'] = {'high': high, 'low': .2, 'development_base_negatives': counts['development_pairs'], 'context_views_not_independent': True}
        heads[kind], networks[kind] = head, network
        selection[kind] = {'selected': name, 'counts': counts, 'refitted': name != 'frozen_release_v31',
            'dev_bce': loss, 'candidates': [{'name': candidate[1], 'dev_bce': candidate[0]} for candidate in candidates], 'weights_sha256': digest_weights(head)}
    if digest_weights(heads['vowel_quality']) != digest_weights(original['heads']['vowel_quality']):
        raise ValueError('Frozen quality-head weights changed')
    state = {'schema': 'context_task_vowels_v61', 'heads': heads, 'feature_names': FEATURE_NAMES,
        'phone_inventory': original['phone_inventory'], 'secondary_phone_inventory': original['secondary_phone_inventory'],
        'class_resource_inventory_sha256': sha(ROOT / 'data/panphon_context_v58/inventory.json'),
        'code_sha256': {name: sha(ROOT / name) for name in [*original['code_sha256'], 'hindi_detector/context_detector.py', 'hindi_detector/consonant_context.py']},
        'original_release_sha256': sha(original_model), 'actual_hindi_error_accuracy': None,
        'policy': 'Use only as a separately validated fallback for explicitly unknown consonant contexts; no consonant identity judgment.'}
    torch.save(state, out / 'model.pt')
    frozen = {'model_sha256': sha(out / 'model.pt'), 'model_bytes': (out / 'model.pt').stat().st_size,
              'created_utc': datetime.now(timezone.utc).isoformat(), 'v62_scores_opened': False,
              'operating_points': {kind: head['operating_points'] for kind, head in heads.items()}}
    (out / 'frozen_selection.json').write_text(json.dumps(frozen, indent=2) + '\n')
    report = {'status': 'complete_context_candidate_frozen_before_v62', 'model_sha256': frozen['model_sha256'],
        'selection': selection, 'train': summary_metrics(state, networks, train), 'development': summary_metrics(state, networks, development),
        'source_content_excluded_pairs': len(exclusions), 'actual_hindi_error_accuracy': None, 'production_changed': False,
        'quality_head_bit_exact_release_v31': True, 'independent_base_pairs': len({row['pair_id'] for row in records}),
        'limits': ['Training context views reuse the same base edit and are not independent samples.',
                   'Same-voice backoff is easier than cross-voice deployment and stays explicitly tagged.',
                   'V62 new words have not been scored here; no real Hindi-error accuracy is established.']}
    membership = [{k: v for k, v in row.items() if k != 'features'} for row in records]
    for name, value in [('report.json', report), ('selection.json', selection), ('history.json', history), ('fitting_rows.json', membership)]:
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'model_sha256', 'selection', 'independent_base_pairs', 'source_content_excluded_pairs')}))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Full-region context fallback fitting failed explicitly; do not open new evaluation scores or promote')
        raise
