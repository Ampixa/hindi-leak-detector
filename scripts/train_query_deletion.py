"""Train a query-only full-region deletion candidate; no production promotion."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import numpy as np
import torch

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.evaluation import binary_metrics
from hindi_detector.query_deletion import MODES, QueryDeletion, inputs, score
from scripts.extract_corrected_training import read_rows
from scripts.train_context_fallback import gather

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {'cross_voice': 'full_context_controls_v60', 'self_backoff': 'full_context_self_v60b'}


def main():
    out = ROOT / 'reports/query_deletion_v66'
    if out.exists():
        raise FileExistsError(out)
    source = ROOT / 'reports/independent_full_controls_v64b'
    report = json.loads((source / 'report.json').read_text())
    if report['status'] != 'complete_independent_physical_evidence' or sha(source / 'items.jsonl') != report['items_sha256']:
        raise ValueError('Independent train/development evidence incomplete or changed')
    third = {(row['group'], row['job_id'], row['label']): row for row in read_rows(source / 'items.jsonl')}
    gathered, exclusions, pins = gather()
    records = [row for row in gathered if row['kind'] == 'vowel_deletion']
    for row in records:
        independent = third.get((GROUPS[row['domain']], row['pair_id'], row['label']))
        if independent is not None and independent['word_id'] != row['word_id']:
            raise ValueError('Third-encoder word identity mismatch')
        row['third_features'] = np.asarray(independent['features'] if independent else np.zeros(32), np.float32)
        row['third_available'] = independent is not None
    training = [row for row in records if row['split'] == 'train']
    development = [row for row in records if row['split'] == 'development']
    if len({row['pair_id'] for row in training}) < 30 or len({row['pair_id'] for row in development}) < 8:
        raise ValueError('Insufficient independent base pairs for query-only deletion fitting')
    write_group(out, {'protocol.json': {
        'inputs': pins, 'third_evidence_sha256': sha(source / 'items.jsonl'),
        'cause': 'Physical full-vowel deletion only; quality head unchanged in production.',
        'independent_samples': 'Base pairs and word families; four context masks reuse each pair.',
        'architecture_search': ['linear', 'mlp8'], 'feature_modes': MODES,
        'selection': 'Minimum equal-word development BCE; train-only normalization; one fixed seed.',
        'threshold': 'Maximum eligible development-sham score plus1e-5, minimum0.5. No heldout data read.',
        'scope': 'No source/voice/word IDs or reference-derived acoustic values are neural inputs.',
        'previously_opened_v62': 'Reserved for retrospective diagnosis only; a new test is required before admission.',
        'actual_hindi_error_accuracy': None, 'production_changed': False},
        'source_snapshot.json': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()}
            for p in (Path(__file__), ROOT / 'hindi_detector/query_deletion.py')]},
        'Query-only physical-deletion research model. model.pt selected small head, exact acoustic '
        'column list, train-only normalization, development threshold and code/data pins. '
        'selection.json six train/dev candidate losses, history.json optimizer traces, '
        'fitting_rows.json base-pair/word/conditioning/mask membership, frozen_selection.json '
        'freeze time/SHA before any later scoring; report.json train/dev counts and metrics. '
        'No raw encoder/audio/frame or epoch-checkpoint copies; quality and live detector unchanged. '
        'Current consumer solve reference-confidence false flags, owner Ampixa/user, review2026-11-10. '
        'Keep selected model/compact evidence while consumed; not on HF. Acoustic/source/PanPhon '
        'rights inherited, independent encoder GPL-3.0 code and absent checkpoint-specific terms, '
        'internal research only/no redistribution. Physical-edit accuracy is not real Hindi-error accuracy.',
        ['.venv-indicf5/bin/python -m scripts.train_query_deletion'])

    def matrices(rows):
        return (np.stack([row['features'] for row in rows]), np.stack([row['third_features'] for row in rows]),
                np.array([row['third_available'] for row in rows]))

    tr, dr = matrices(training), matrices(development)
    y, dy = [np.array([row['label'] for row in rows], np.float32) for rows in (training, development)]
    weights = []
    for rows in (training, development):
        frequency = Counter(row['word'] for row in rows)
        weight = np.array([1 / frequency[row['word']] for row in rows], np.float32)
        weights.append(torch.from_numpy(weight / weight.mean()))
    wt, wd = weights
    yt, dyt = torch.from_numpy(y), torch.from_numpy(dy)
    torch.set_num_threads(2)
    candidates, history = [], {}
    for mode in MODES:
        x, dx = inputs(*tr, mode), inputs(*dr, mode)
        mean, scale = x.mean(0), np.maximum(x.std(0), .05)
        tx, dt = [torch.from_numpy(np.clip((v - mean) / scale, -8, 8).astype(np.float32)) for v in (x, dx)]
        for architecture in ('linear', 'mlp8'):
            torch.manual_seed(20261109)
            network = QueryDeletion(mode, architecture)
            optimizer = torch.optim.AdamW(network.parameters(), lr=.01, weight_decay=.1)
            best, stale, trace = None, 0, []
            for epoch in range(500):
                optimizer.zero_grad(set_to_none=True)
                logits = network(tx)
                loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, yt, reduction='none') * wt).mean()
                loss += .25 * (torch.nn.functional.softplus(-(logits[1::2] - logits[::2])) * wt[::2]).mean()
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite query-only deletion objective')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 5, error_if_nonfinite=True)
                optimizer.step()
                with torch.inference_mode():
                    value = float((torch.nn.functional.binary_cross_entropy_with_logits(network(dt), dyt, reduction='none') * wd).mean())
                trace.append({'epoch': epoch + 1, 'loss': float(loss.detach()), 'development_bce': value})
                if best is None or value < best['loss'] - .0001:
                    best = {'loss': value, 'epoch': epoch + 1, 'weights': {key: tensor.detach().clone() for key, tensor in network.state_dict().items()}}
                    stale = 0
                else:
                    stale += 1
                if stale >= 40:
                    break
            network.load_state_dict(best['weights'], strict=True)
            state = {'schema': 'query_full_deletion_v66', 'feature_mode': mode, 'architecture': architecture,
                'feature_names': MODES[mode], 'weights': best['weights'], 'mean': mean, 'scale': scale,
                'best_epoch': best['epoch'], 'code_sha256': {'hindi_detector/query_deletion.py': sha(ROOT / 'hindi_detector/query_deletion.py')},
                'third_runtime_sha256': sha(source / 'runtime.json'), 'actual_hindi_error_accuracy': None,
                'requires_reference_audio': False, 'third_acoustic_encoder_required': mode != 'dual46'}
            candidates.append((best['loss'], mode + '/' + architecture, network.eval(), state))
            history[mode + '/' + architecture] = trace
    value, name, network, state = min(candidates, key=lambda candidate: (candidate[0], candidate[1]))
    probabilities = score(network, state, *dr)
    high = max(.5, float(probabilities[dy == 0].max()) + .00001)
    state['operating_points'] = {'high': high, 'low': .2}
    torch.save(state, out / 'model.pt')
    (out / 'frozen_selection.json').write_text(json.dumps({'created_utc': datetime.now(timezone.utc).isoformat(),
        'model_sha256': sha(out / 'model.pt'), 'new_reserved_scores_opened': False,
        'previous_v62_is_diagnosis_only': True, 'operating_points': state['operating_points']}, indent=2) + '\n')
    selection = [{'name': candidate[1], 'development_bce': candidate[0]} for candidate in candidates]
    metrics = {}
    for part, rows, data, labels in [('train', training, tr, y), ('development', development, dr, dy)]:
        probabilities = score(network, state, *data)
        metrics[part] = {'base_pairs': len({row['pair_id'] for row in rows}), 'word_families': len({row['word'] for row in rows}),
            'conditioning_groups': sorted({row['voice'] for row in rows}),
            'per_context': {mode: binary_metrics(labels[mask], probabilities[mask], high)
                for mode in sorted({row['mode'] for row in rows})
                for mask in [np.array([row['mode'] == mode for row in rows])]},
            'context_views_are_independent': False}
    report = {'status': 'complete_query_only_deletion_candidate_frozen', 'selected': name,
              'development_bce': value, 'model_sha256': sha(out / 'model.pt'), 'operating_points': state['operating_points'],
              'metrics': metrics, 'actual_hindi_error_accuracy': None, 'production_changed': False}
    for filename, data in [('selection.json', selection), ('history.json', history), ('report.json', report),
        ('fitting_rows.json', [{key: value for key, value in row.items() if key not in {'features', 'third_features'}} for row in records])]:
        (out / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Query-only deletion fitting failed explicitly; no production change')
        raise
