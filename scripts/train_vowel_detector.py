"""Fit/freeze an explicit vowel-change head before loading held-out controls."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import io
import hashlib
import json
import logging
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.evaluation import binary_metrics
from hindi_detector.pronunciation_sites import FEATURE_NAMES
from hindi_detector.vowel_detector import VowelSiteNetwork, decisions, score

ROOT = Path(__file__).resolve().parents[1]


def read_partition(base, partition, index):
    matrix_path = base/'training_partitions'/('arrays_'+partition+'.npz')
    inventory = json.loads((base/'training_partitions/inventory.json').read_text())
    expected = next(r for r in inventory if r['path'] == matrix_path.name)
    if sha(matrix_path) != expected['sha256']:raise ValueError('Frozen partition matrix changed')
    with np.load(matrix_path, allow_pickle=False) as data:
        values, labels = data['features'], data['labels']
    rows = [r for r in index if r['split'] == partition]
    if len(values) != len(rows) or not np.array_equal(labels, np.array([r['label'] for r in rows])):
        raise ValueError('Control partition matrix/metadata identity differs')
    if len(rows) % 2 or any((a['label'], b['label']) != (0, 1) or
            any(a[k] != b[k] for k in ('key', 'word', 'ordinal', 'unit_index', 'kind')) for a, b in zip(rows[::2], rows[1::2])):
        raise ValueError('Control source pairs are incomplete or reordered')
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite control features')
    return values, labels, rows


def measure(state, values, labels, rows, probabilities):
    output = {}
    decision = np.array(decisions(state, values, probabilities))
    for kind in ('vowel_deletion', 'vowel_quality'):
        mask = np.array([r['kind'] == kind for r in rows])
        y, p = labels[mask], probabilities[mask]
        if not len(y):
            continue
        op = state['operating_points'][kind]
        positive = decision[mask] == 'strong_vowel_change'
        output[kind] = {'balanced': binary_metrics(y, p, op['balanced']),
            'conservative_score': binary_metrics(y, p, op['high']),
            'final_gated_decision': binary_metrics(y, positive.astype(float), .5),
            'paired_ordering': float(np.mean(p[1::2] > p[::2])), 'operating_points': op,
            'uncertain_fraction': float(np.mean(decision[mask] == 'uncertain'))}
    return output


def main():
    base = ROOT/'reports/vowel_controls_v19'
    out = ROOT/'reports/vowel_detector_v19'
    if out.exists():
        raise FileExistsError(out)
    write_group(out, {'protocol.json': {'features': FEATURE_NAMES,
        'candidate_architectures': ['linear', 'mlp16'], 'selection': 'Minimum development BCE, seed44, no actual Swar regression tuning',
        'thresholds': 'Development-only by vowel-change kind; high raw-score threshold aims at5% sham flags, plus fixed directional evidence gate',
        'test_policy': 'Held-out control feature matrix read only after selected model/operating points saved',
        'label_scope': 'Known actual same-speaker vowel intervention, not independent real Hindi-error labels'},
        'source_snapshot.json': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()}
            for p in (Path(__file__), ROOT/'hindi_detector/vowel_detector.py', ROOT/'hindi_detector/pronunciation_sites.py', ROOT/'hindi_detector/phone_lattice.py')]},
        'Vowel-site neural model. model.pt includes selected weights/train-only normalization/development operating '
        'points and source/encoder pins. history.json optimizer traces for the two predeclared small models; no '
        'epoch checkpoints retained. report.json separates controlled held-out change detection from actual user '
        'regressions and real Hindi-error accuracy. predictions.json has held-out control decisions; actual_swar.json '
        'is produced after freeze from real source recordings. No raw voice, speaker or generator features. '
        'Review2026-11-09, current consumer actual Nepali TTS word analysis. Keep selected model/compact inputs '
        'and reports; remove redundant diagnostics24h after consumer closes. No HF upload; source/MMS research licences.',
        ['.venv-indicf5/bin/python -m scripts.train_vowel_detector',
         'Known same-speaker physical interventions, frozen speaker/word exclusions,18 acoustic path features'])

    def status(phase, **extra):
        p = out/'status.json'; temp = p.with_suffix('.partial')
        temp.write_text(json.dumps({'phase': phase, 'pid': os.getpid(),
            'updated_utc': datetime.now(timezone.utc).isoformat(), **extra}, indent=2)+'\n'); temp.replace(p)

    try:
        status('waiting_for_actual_vowel_controls')
        while True:
            previous = json.loads((base/'status.json').read_text())
            if previous['phase'] == 'failed_explicitly':
                raise RuntimeError('Actual same-speaker control generation failed')
            if previous['phase'] == 'complete_actual_vowel_controls':
                break
            os.kill(previous['pid'], 0); time.sleep(5)
        report = json.loads((base/'report.json').read_text())
        index = json.loads((base/'index.json').read_text())
        if sha(base/'index.json') != report['index_sha256']:
            raise ValueError('Frozen control index changed')
        train, y, train_rows = read_partition(base, 'train', index)
        dev, dy, dev_rows = read_partition(base, 'development', index)
        if len(train) < 64 or len(dev) < 32:
            raise ValueError('Insufficient independent control groups')
        if any(r['word'] in {'हैन', 'किनकि'} for r in train_rows+dev_rows):
            raise ValueError('Known regression lexeme reached fitting')
        mean = train.mean(0); scale = np.maximum(train.std(0), .05)
        norm = lambda values: torch.from_numpy(np.clip((values-mean)/scale, -8, 8).astype(np.float32))
        tx, dx = norm(train), norm(dev)
        ty, dyt = torch.from_numpy(y.astype(np.float32)), torch.from_numpy(dy.astype(np.float32))
        frequencies = Counter(r['word'] for r in train_rows)
        weights = torch.tensor([1/frequencies[r['word']] for r in train_rows], dtype=torch.float32)
        weights /= weights.mean()
        torch.set_num_threads(2)
        candidates, histories = [], {}
        for architecture in ('linear', 'mlp16'):
            torch.manual_seed(44)
            network = VowelSiteNetwork(architecture)
            optimizer = torch.optim.AdamW(network.parameters(), lr=.01, weight_decay=.02)
            best, stale, snapshot, best_epoch = float('inf'), 0, None, None
            history = []
            for epoch in range(400):
                network.train(); optimizer.zero_grad(set_to_none=True)
                logits = network(tx)
                bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, ty, reduction='none')
                loss = (bce*weights).mean() + .25*torch.nn.functional.softplus(-(logits[1::2]-logits[::2])).mean()
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite site optimization loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(network.parameters(), 5, error_if_nonfinite=True); optimizer.step()
                network.eval()
                with torch.inference_mode():
                    value = float(torch.nn.functional.binary_cross_entropy_with_logits(network(dx), dyt))
                history.append({'epoch': epoch+1, 'train_loss': float(loss.detach()), 'development_bce': value})
                if value < best-.0001:
                    best, stale, best_epoch = value, 0, epoch+1
                    buffer = io.BytesIO(); torch.save(network.state_dict(), buffer); snapshot = buffer.getvalue()
                else:
                    stale += 1
                if stale >= 35:
                    break
            network.load_state_dict(torch.load(io.BytesIO(snapshot), weights_only=True))
            candidates.append((best, architecture, network.eval(), best_epoch))
            histories[architecture] = history
        best, architecture, network, best_epoch = min(candidates, key=lambda r: r[0])
        state = {'schema': 'paired_vowel_site_v19', 'architecture': architecture, 'weights': network.state_dict(),
            'mean': mean, 'scale': scale, 'feature_names': FEATURE_NAMES, 'best_epoch': best_epoch,
            'operating_points': {}, 'labels': 'Known actual same-speaker vowel replacement/removal; no actual Hindi-error probability',
            'control_index_sha256': sha(base/'index.json'), 'control_runtime_sha256': sha(base/'runtime.json'),
            'phone_plan_sha256': sha(ROOT/'data/swar_phone_plan_v18/words.json'),
            'projection_code_sha256': sha(ROOT/'hindi_detector/phone_lattice.py'),
            'site_code_sha256': sha(ROOT/'hindi_detector/pronunciation_sites.py'),
            'actual_hindi_error_accuracy': None}
        dp = score(network, state, dev)
        for kind in ('vowel_deletion', 'vowel_quality'):
            mask = np.array([r['kind'] == kind for r in dev_rows]); negative = dp[mask & (dy == 0)]; positive = dp[mask & (dy == 1)]
            if min(len(negative), len(positive)) < 8:
                raise ValueError('Insufficient development pairs for ' + kind)
            high = float(np.nextafter(np.sort(negative)[max(0, len(negative)-int(.05*len(negative))-1)], np.float32(np.inf)))
            low = float(np.quantile(positive, .1))
            balanced = max((binary_metrics(dy[mask], dp[mask], float(t))['balanced_accuracy'], -abs(t-.5), float(t)) for t in np.linspace(.05,.95,91))[2]
            state['operating_points'][kind] = {'high': high, 'low': min(low, high), 'balanced': balanced}
        torch.save(state, out/'model.pt')
        digest = sha(out/'model.pt')
        (out/'history.json').write_text(json.dumps(histories, indent=2)+'\n')
        status('model_frozen_loading_reserved_controls', model_sha256=digest)
        test, test_y, test_rows = read_partition(base, 'test', index)
        tp = score(network, state, test)
        result = {'status': 'trained_source_independent_vowel_site_candidate', 'model_sha256': digest,
            'architecture': architecture, 'best_epoch': best_epoch, 'development_bce': best,
            'control_pairs': {p: sum(r['split']==p for r in index)//2 for p in ('train','development','test')},
            'control_lexemes': {p: len({r['word'] for r in index if r['split']==p}) for p in ('train','development','test')},
            'development': measure(state, dev, dy, dev_rows, dp), 'test': measure(state, test, test_y, test_rows, tp),
            'heldout_speakers': ['yt01','yt10','yt12'], 'actual_hindi_error_accuracy': None,
            'actual_user_regression_accuracy': None, 'limits': [
                'Controlled vowel corruption detection is not independent real Hindi-error accuracy.',
                'Same-speaker edits prevent generator/voice identity from supplying intervention labels, but splice artifacts and probe biases remain.',
                'Reference pronunciation and dual-G2P coverage constrain inference; unsupported sites abstain.',
                'Prosody/intonation error correctness is not established by this vowel head.']}
        masks = {'unseen_lexemes': np.array([int(hashlib.sha256(r['word'].encode()).hexdigest()[:8],16)%100>=85 for r in test_rows]),
                 'unseen_speakers': np.array([r['voice'] in {'yt01','yt10','yt12'} for r in test_rows])}
        for key, mask in masks.items():
            result[key] = measure(state, test[mask], test_y[mask], [r for r,m in zip(test_rows,mask) if m], tp[mask]) if mask.any() else None
        predictions = [{**r, 'controlled_change_score': float(p), 'decision': d} for r,p,d in zip(
            test_rows, tp, decisions(state, test, tp), strict=True)]
        (out/'predictions.json').write_text(json.dumps(predictions, ensure_ascii=False)+'\n')
        (out/'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
        if sha(out/'model.pt') != digest:
            raise ValueError('Selected model changed after reserved evaluation')
        status('complete_frozen_vowel_site_model', model_sha256=digest)
    except Exception:
        status('failed_explicitly', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Vowel site fitting failed explicitly')
        raise
