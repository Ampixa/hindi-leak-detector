"""Fit word-local paired candidates and reserve test until checkpoint freeze."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import io
import json
import logging
import os
from pathlib import Path
import time

import numpy as np
import torch

from hindi_detector.artifacts import write_group
from hindi_detector.evaluation import binary_metrics
from hindi_detector.paired_detector import BRANCHES, PairedWordNetwork, load_feature, paired_features, score_arrays, sha
from scripts.extract_bulk_words import read_complete

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)
PARTS = ('train', 'development', 'reserved_evaluation')


def dataset(groups: dict, directory: Path, projection: np.ndarray, contrast_words: set) -> list[dict]:
    samples = []
    for tid, receipts in groups.items():
        loaded = {(r['row']['system'], r['row']['speaker_id']): load_feature(directory, r) for r in receipts}
        voices = sorted({r['row']['speaker_id'] for r in receipts})
        if len(voices) != 2 or len(loaded) != 4:
            raise ValueError('Need exact two-source/two-conditioning quartets')
        for ordinal, voice in enumerate(voices):
            native = loaded['pocket_nepali_v1c', voice]
            hindi = loaded['pocket_hindi_parent', voice]
            reference = loaded['pocket_nepali_v1c', voices[1 - ordinal]]
            nf = paired_features(native, reference, projection)
            hf = paired_features(hindi, reference, projection)
            words = native['detail']['words']; focal = native['receipt']['focal_word_indices']
            if focal != hindi['receipt']['focal_word_indices']:
                raise ValueError('Focal indices disagree between same-text sources')
            eligible = np.array([i in focal or w['word'] not in contrast_words for i, w in enumerate(words)])
            for condition in ('native_control', 'feature_local_hindi_replacement', 'full_hindi_source_audit'):
                if condition == 'native_control':
                    vectors = tuple(v.copy() for v in nf)
                elif condition == 'feature_local_hindi_replacement':
                    vectors = tuple(v.copy() for v in nf)
                    for target, source in zip(vectors, hf):
                        target[focal] = source[focal]
                else:
                    vectors = hf
                labels = np.zeros(len(words), dtype=np.float32)
                if condition != 'native_control':
                    labels[focal] = 1
                mask = eligible.copy()
                if condition == 'full_hindi_source_audit':
                    mask[:] = False; mask[focal] = True
                samples.append({'text_id': tid, 'text': native['receipt']['row']['text'], 'text_kind': native['receipt']['row']['text_kind'],
                    'speaker_id': voice, 'reference_speaker_id': voices[1 - ordinal], 'condition': condition,
                    'phonetics': vectors[0], 'prosody': vectors[1], 'labels': labels, 'mask': mask,
                    'focal': focal, 'words': hindi['detail']['words'] if condition == 'full_hindi_source_audit' else words,
                    'query_clip_id': (native if condition == 'native_control' else hindi)['receipt']['clip_id'],
                    'reference_clip_id': reference['receipt']['clip_id'], 'native_clip_id': native['receipt']['clip_id'],
                    'hindi_clip_id': hindi['receipt']['clip_id']})
    return samples


def flatten(samples: list[dict], *, focal_only: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = [s for s in samples if s['condition'] != 'full_hindi_source_audit']
    masks = []
    for row in rows:
        mask = row['mask'].copy()
        if focal_only:
            mask[:] = False; mask[row['focal']] = True
        masks.append(mask)
    return tuple(np.concatenate([r[name][mask] for r, mask in zip(rows, masks)]) for name in ('phonetics', 'prosody', 'labels'))


def report_scores(network, state, samples: list[dict]) -> tuple[dict, list[dict]]:
    predictions = []
    for sample in samples:
        scores = score_arrays(network, state, sample['phonetics'], sample['prosody'])
        predictions.append({k: sample[k] for k in ('text_id', 'text', 'text_kind', 'speaker_id', 'reference_speaker_id', 'condition', 'focal',
                                                   'query_clip_id', 'reference_clip_id', 'native_clip_id', 'hindi_clip_id')})
        predictions[-1]['words'] = [{**{k: w[k] for k in ('word', 'start', 'end')},
            'source_label': int(sample['labels'][i]) if sample['mask'][i] else None,
            'scores': dict(zip(BRANCHES, map(float, scores[i]))), 'actual_hindi_error_probability': None}
            for i, w in enumerate(sample['words'])]
    result = {}
    for branch in BRANCHES:
        scopes = {}
        for scope, conditions, focal_only in (
            ('feature_localization', {'native_control', 'feature_local_hindi_replacement'}, False),
            ('focal_source', {'native_control', 'feature_local_hindi_replacement'}, True),
            ('full_source_focal_audit', {'native_control', 'full_hindi_source_audit'}, True)):
            pairs = [(w['source_label'], w['scores'][branch], row['text_id'], w['word']) for row in predictions if row['condition'] in conditions
                     for i, w in enumerate(row['words']) if w['source_label'] is not None and (not focal_only or i in row['focal'])]
            labels = np.array([p[0] for p in pairs]); scores = np.array([p[1] for p in pairs])
            fixed = binary_metrics(labels, scores)
            selected = binary_metrics(labels, scores, state['thresholds'][branch])
            clusters = defaultdict(list)
            for i, p in enumerate(pairs):
                clusters[p[3] if focal_only else p[2]].append(i)
            rng = np.random.default_rng(33); keys = list(clusters); draws = []
            for _ in range(500):
                indices = [i for k in rng.choice(keys, len(keys), replace=True) for i in clusters[k]]
                value = binary_metrics(labels[indices], scores[indices], state['thresholds'][branch])['balanced_accuracy']
                if value is not None:
                    draws.append(value)
            scopes[scope] = {'fixed_threshold_0_5': fixed, 'development_threshold': state['thresholds'][branch],
                'selected_threshold_metrics': selected, 'clusters': len(keys),
                'cluster_kind': 'focal_lexeme' if focal_only else 'text',
                'cluster_bootstrap_balanced_accuracy_ci95': np.quantile(draws, [.025, .975]).tolist() if draws else None}
        native = [w['scores'][branch] for row in predictions if row['condition'] == 'native_control' for w in row['words'] if w['source_label'] is not None]
        scopes['native_control_word_flag_rate'] = float(np.mean(np.array(native) >= state['thresholds'][branch]))
        result[branch] = scopes
    return result, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--wait-complete', action='store_true')
    parser.add_argument('--epochs', type=int, default=120)
    args = parser.parse_args()
    if args.out.exists() or not 1 <= args.epochs <= 500:
        raise ValueError('Fresh output and bounded positive epoch count required')
    torch.set_num_threads(2); torch.manual_seed(44); np.random.seed(44)
    if args.wait_complete:
        while True:
            state = json.loads((args.features / 'status.json').read_text())
            if state['phase'] == 'complete_bounded_word_extraction':
                break
            if datetime.now(timezone.utc) >= datetime.fromisoformat('2026-11-08T01:50:00+00:00'):
                LOG.warning('Bounded wait ended; freeze only already complete quartet receipts')
                break
            os.kill(state['pid'], 0)
            time.sleep(10)
    receipts = read_complete(args.features / 'items.jsonl')
    by_text = defaultdict(list)
    for row in receipts:
        by_text[row['text_id']].append(row)
    complete = {tid: rows for tid, rows in by_text.items() if len(rows) == 4 and all(r['status'] == 'ok_word_features' for r in rows)}
    partitions = {p: {tid: rows for tid, rows in complete.items() if rows[0]['row']['split'] == p} for p in PARTS}
    if min(map(len, partitions.values())) < 8:
        raise ValueError('At least eight independent texts required in each fixed partition')
    # Check provenance partitions before loading feature arrays, including held-out ones.
    sets = {}
    for part, groups in partitions.items():
        rows = [r['row'] for group in groups.values() for r in group]
        sets[part] = {'texts': {r['text'] for r in rows}, 'conditioning': {r['speaker_id'] for r in rows},
                      'focal_words': {w for r in rows for w in r['target_words']}}
    for a in PARTS:
        for b in PARTS:
            if a < b:
                for key in sets[a]:
                    if sets[a][key] & sets[b][key]:
                        raise ValueError(f'Partition leakage: {a}/{b}/{key}')
    priors_path = ROOT / 'data/pocket_bulk_plan_v11/g2p_priors.json'
    contrast_words = {r['word'] for r in json.loads(priors_path.read_text())}
    rng = np.random.default_rng(20261108)
    projection = (rng.integers(0, 2, size=(4096, 128), dtype=np.int8).astype(np.float32) * 2 - 1) / np.sqrt(128)
    source_paths = [Path(__file__), ROOT / 'hindi_detector/paired_detector.py', ROOT / 'hindi_detector/evaluation.py']
    frozen = {p: [r for group in groups.values() for r in group] for p, groups in partitions.items()}
    write_group(args.out, {'inputs.json': {'features': str(args.features.resolve()), 'feature_runtime_sha256': sha(args.features / 'runtime.json'),
        'prior_sha256': sha(priors_path), 'epochs': args.epochs, 'partition_texts': {p: len(v) for p, v in partitions.items()},
        'partition_focal_lexemes': {p: len(sets[p]['focal_words']) for p in PARTS}, 'frozen_receipts': frozen},
        'source_snapshot.json': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()} for p in source_paths]},
        'Paired reference-conditioned word-source candidate. model.pt is the single selected final checkpoint; history.json is compact train/dev fitting history, report.json and predictions.json are frozen test results, status.json phase/PID. No human accent labels. Synthetic localized feature replacements have known source identity but are not physical waveforms. Focal lexemes/texts/conditioning disjoint; incidental contrast words excluded from negative supervision. Random hidden projection is pinned inside model. Keep final model and metrics until superseded; no additional epoch checkpoints retained.',
        ['.venv-indicf5/bin/python -m scripts.train_bulk_detector --features GROUP --out NEW [--wait-complete]', 'Original WAV and feature identities in frozen_receipts; native alternative pronunciations remain valid.'])
    def status(phase, **extra):
        p = args.out / 'status.json'; temporary = p.with_suffix('.partial')
        temporary.write_text(json.dumps({'phase': phase, 'pid': os.getpid(), 'updated_utc': datetime.now(timezone.utc).isoformat(), **extra}, indent=2) + '\n'); temporary.replace(p)
    status('loading_train_development_only')
    train = dataset(partitions['train'], args.features, projection, contrast_words)
    dev = dataset(partitions['development'], args.features, projection, contrast_words)
    tx, ts, ty = flatten(train); dx, ds, dy = flatten(dev, focal_only=True)
    normalization = {}
    for name, values in (('phonetics', tx), ('prosody', ts)):
        normalization[name] = {'mean': values.mean(0), 'scale': np.maximum(values.std(0), .01)}
    normalize = lambda name, v: torch.from_numpy(np.clip((v - normalization[name]['mean']) / normalization[name]['scale'], -8, 8).astype(np.float32))
    tensors = [normalize('phonetics', tx), normalize('prosody', ts), torch.from_numpy(ty)]
    dev_tensors = [normalize('phonetics', dx), normalize('prosody', ds), torch.from_numpy(dy)]
    dimensions = {'phonetic_dimension': tx.shape[1], 'prosody_dimension': ts.shape[1]}
    network = PairedWordNetwork(**dimensions)
    optimizer = torch.optim.AdamW(network.parameters(), lr=.001, weight_decay=.02)
    pos_weight = torch.tensor(float((ty == 0).sum() / (ty == 1).sum()))
    history = []; best_loss = float('inf'); best_bytes = None; stale = 0
    for epoch in range(args.epochs):
        network.train(); order = torch.randperm(len(ty)); losses = []
        for indices in order.split(256):
            optimizer.zero_grad(set_to_none=True)
            logits = network(tensors[0][indices], tensors[1][indices])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, tensors[2][indices, None].expand_as(logits), pos_weight=pos_weight)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(network.parameters(), 5., error_if_nonfinite=True); optimizer.step()
            losses.append(float(loss.detach()))
        network.eval()
        with torch.inference_mode():
            dev_logits = network(dev_tensors[0], dev_tensors[1])
            dev_loss = float(torch.nn.functional.binary_cross_entropy_with_logits(dev_logits, dev_tensors[2][:, None].expand_as(dev_logits)))
        history.append({'epoch': epoch + 1, 'train_loss': float(np.mean(losses)), 'development_focal_bce': dev_loss})
        if dev_loss < best_loss - .0001:
            best_loss = dev_loss; stale = 0
            buffer = io.BytesIO(); torch.save(network.state_dict(), buffer); best_bytes = buffer.getvalue(); best_epoch = epoch + 1
        else:
            stale += 1
        status('fitting_train_selecting_development', epoch=epoch + 1, best_epoch=best_epoch, development_bce=dev_loss,
               training_words=len(ty), train_texts=len(partitions['train']))
        if stale >= 15:
            break
    network.load_state_dict(torch.load(io.BytesIO(best_bytes), weights_only=True))
    model_state = {'schema': 'paired_word_source_candidate_v11', 'state_dict': network.state_dict(), 'dimensions': dimensions,
        'normalization': normalization, 'projection': projection.astype(np.float32), 'best_epoch': best_epoch,
        'branches': BRANCHES, 'thresholds': {}, 'training_partition_sets': {p: {k: sorted(v) for k, v in sets[p].items()} for p in ('train', 'development')},
        'feature_runtime_sha256': sha(args.features / 'runtime.json'), 'labels': 'known Hindi-source feature replacement at G2P focal sites; native baseline source controls',
        'actual_hindi_accuracy': None, 'actual_prosody_accuracy': None}
    model_state['thresholds'] = dict.fromkeys(BRANCHES, .5)
    dev_scores = score_arrays(network, model_state, dx, ds)
    for i, branch in enumerate(BRANCHES):
        choices = [(binary_metrics(dy, dev_scores[:, i], float(t))['balanced_accuracy'], -abs(float(t) - .5), float(t)) for t in np.linspace(.05, .95, 91)]
        model_state['thresholds'][branch] = max(choices)[2]
    torch.save(model_state, args.out / 'model.pt')
    (args.out / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
    model_hash = sha(args.out / 'model.pt')
    status('checkpoint_frozen_before_reserved_feature_loading', model_sha256=model_hash, best_epoch=best_epoch)
    # First held-out array load is after immutable model/thresholds are written.
    test = dataset(partitions['reserved_evaluation'], args.features, projection, contrast_words)
    metrics, predictions = report_scores(network, model_state, test)
    report = {'status': 'trained_paired_word_source_candidate_not_validated_accent_detector', 'model_sha256': model_hash,
        'best_epoch': best_epoch, 'epochs_run': len(history), 'training_loss_word_rows': len(ty),
        'partition_texts': {p: len(v) for p, v in partitions.items()}, 'partition_audio_clips': {p: len(v) for p, v in frozen.items()},
        'partition_focal_lexemes': {p: len(sets[p]['focal_words']) for p in PARTS},
        'conditioning_groups': {p: len(sets[p]['conditioning']) for p in PARTS},
        'text_kind_counts': {p: dict(Counter(g[0]['row']['text_kind'] for g in v.values())) for p, v in partitions.items()},
        'test': metrics, 'actual_hindi_error_accuracy': None, 'actual_prosody_error_accuracy': None,
        'limits': ['Feature-local replacements do not simulate acoustic splice boundary effects or sentence-level intonation changes.',
                   'Source identity at a focal G2P site is weak supervision, not proof of an incorrectly realized phone.',
                   'No gold Hindi pronunciation/prosody errors; class scores are uncalibrated source evidence.',
                   'Same generator family and shared authored carrier templates; no claim of unseen-generator generalization.',
                   'Hidden speech features can retain voice/channel cues despite held-out conditioning groups.',
                   'Only ASR-passing complete quartets used; report generated, filtered and trained counts separately.']}
    (args.out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    (args.out / 'predictions.json').write_text(json.dumps(predictions, ensure_ascii=False, allow_nan=False) + '\n')
    if sha(args.out / 'model.pt') != model_hash:
        raise ValueError('Frozen model mutated during final evaluation')
    status('complete_frozen_test_evaluation', model_sha256=model_hash, partition_texts=report['partition_texts'])
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Paired word candidate fitting failed; no success substituted')
        raise
