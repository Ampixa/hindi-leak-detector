"""Stream complete content-passing text quartets into shared word encoders."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import time
import traceback

import numpy as np

from hindi_detector.artifacts import write_group
from hindi_detector.local_features import PROSODY_NAMES
from hindi_detector.word_evidence import WordEvidenceEngine
from scripts.calibrate_asr_gate import audio

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)
PARTS = ('train', 'development', 'reserved_evaluation')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_complete(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_bytes().split(b'\n')
    return [json.loads(line) for line in lines[:-1]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--asr', action='append', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--quotas', nargs=3, type=int, default=[512, 128, 128])
    parser.add_argument('--cutoff-utc', default='2026-11-08T01:45:00+00:00')
    args = parser.parse_args()
    cutoff = datetime.fromisoformat(args.cutoff_utc)
    if args.out.exists() or cutoff.tzinfo is None or min(args.quotas) < 1:
        raise ValueError('Need a fresh output, aware cutoff and positive quotas')
    plan_path = ROOT / 'data/pocket_bulk_plan_v11/selection.json'
    prior_path = ROOT / 'data/pocket_bulk_plan_v11/g2p_priors.json'
    priors = json.loads(prior_path.read_text())
    bank = [{'word': r['word'], 'nepali_g2p': {'ipa_candidate_units': r['nepali_ipa_variants'][0]},
             'hindi_g2p': {'units_with_stress_preserved': r['hindi_espeak_raw'].strip().split('|')}} for r in priors]
    plan = json.loads(plan_path.read_text())
    expected = defaultdict(set)
    for job in plan['jobs']:
        for system in ('pocket_hindi_parent', 'pocket_nepali_v1c'):
            expected[job['text_id']].add((system, job['speaker_id']))
    source_paths = [Path(__file__), ROOT / 'hindi_detector/word_evidence.py', ROOT / 'hindi_detector/local_features.py',
                    ROOT / 'hindi_detector/g2p_evidence.py', ROOT / 'hindi_detector/ctc.py']
    write_group(args.out, {'inputs.json': {'asr': [str(p.resolve()) for p in args.asr],
        'plan_sha256': sha(plan_path), 'prior_sha256': sha(prior_path), 'quotas': dict(zip(PARTS, args.quotas)),
        'cutoff_utc': cutoff.isoformat(), 'selection': 'first available complete four-pass focal-text quartet; quotas fixed before evaluation'},
        'source_snapshot.json': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()} for p in source_paths]},
        'Accepted same-text four-clip quartets; items.jsonl records each original row, alignment and encoder artifacts/hashes. Per-clip *.json contains word boundaries, phone/G2P and relative prosody; *.npz contains float16 hidden layers12/24 and float32 phone/prosody. status.json is live coverage, runtime.json pins models. Feature extraction may see held-out audio, model fitting must not. First native G2P variant is a diagnostic prior only; all alternatives remain in pinned corpus priors.',
        ['.venv-indicf5/bin/python -m scripts.extract_bulk_words --asr GROUP ... --out NEW', 'No original audio copied. Existing model licences apply; NOT uploaded to HF. Delete redundant per-clip details after final inference/report consumer is superseded.'])
    engine = WordEvidenceEngine(ROOT, g2p_word_bank=bank, device='mps', pitch_method='dio', hidden_layers=(12, 24))
    (args.out / 'runtime.json').write_text(json.dumps({'alignment': engine.aligner_inventory,
        'phone_encoder': engine.phone_inventory, 'vocabulary': engine.vocab, 'prosody_names': PROSODY_NAMES,
        'device': 'mps', 'pitch_method': 'dio', 'hidden_layers': [12, 24]}, indent=2) + '\n')
    done = set(); counts = Counter(); successes = 0; failures = 0
    def status(phase, **extra):
        p = args.out / 'status.json'; temporary = p.with_suffix('.partial')
        temporary.write_text(json.dumps({'phase': phase, 'pid': os.getpid(), 'updated_utc': datetime.now(timezone.utc).isoformat(),
            'complete_texts': dict(counts), 'successful_clips': successes, 'failed_clips': failures, **extra}, indent=2) + '\n')
        temporary.replace(p)
    status('extracting_content_passing_word_quartets')
    while datetime.now(timezone.utc) < cutoff:
        by_text = defaultdict(dict); seen = {}
        for group in args.asr:
            for row in read_complete(group / 'items.jsonl'):
                key = row['clip_id']
                if key in seen and seen[key]['sha256'] != row['sha256']:
                    raise ValueError('Conflicting duplicate ASR clip identity')
                seen[key] = row
        for row in seen.values():
            if row['target_words'] and row['content_quality']['decision'] == 'pass':
                by_text[row['text_id']][(row['system'], row['speaker_id'])] = row
        ready = []
        for tid, rows in by_text.items():
            if tid in done or set(rows) != expected[tid]:
                continue
            split = next(iter(rows.values()))['split']
            if counts[split] < dict(zip(PARTS, args.quotas))[split]:
                ready.append((tid, rows))
        if not ready:
            status('waiting_for_content_passing_quartets', available_focal_texts=len(by_text))
            if all(counts[p] >= q for p, q in zip(PARTS, args.quotas)):
                break
            time.sleep(5)
            continue
        # Balance progress across fixed partitions, without looking at model scores.
        ready.sort(key=lambda r: (counts[next(iter(r[1].values()))['split']] / dict(zip(PARTS, args.quotas))[next(iter(r[1].values()))['split']], r[0]))
        tid, rows = ready[0]; complete = True
        for row in rows.values():
            identity = row['clip_id']; started = time.monotonic()
            receipt = {'row': row, 'text_id': tid, 'clip_id': identity}
            try:
                samples = audio(Path(row['source_pack']), row)
                detail, arrays = engine.analyze(samples, row['text'])
                if any(not np.isfinite(v).all() for v in arrays.values()):
                    raise ValueError('Nonfinite retained feature including float16 conversion')
                active = [i for i, w in enumerate(detail['words']) if w['word'] in row['target_words']]
                if not active:
                    raise ValueError('Focal target missing from actual aligned words')
                detail.update(clip_id=identity, source_audio_sha256=row['sha256'], focal_word_indices=active)
                dp = args.out / (identity + '.json'); fp = args.out / (identity + '.npz')
                if dp.exists() or fp.exists():
                    raise FileExistsError('Feature identity already retained')
                dp.write_text(json.dumps(detail, ensure_ascii=False, allow_nan=False) + '\n')
                np.savez_compressed(fp, **arrays)
                receipt.update(status='ok_word_features', detail=dp.name, detail_sha256=sha(dp), detail_bytes=dp.stat().st_size,
                    features=fp.name, features_sha256=sha(fp), feature_bytes=fp.stat().st_size,
                    focal_word_indices=active, words=len(detail['words']))
                successes += 1
            except Exception:
                LOG.exception('Explicit word extraction exclusion %s', identity)
                receipt.update(status='failed_word_features', error=traceback.format_exc()); failures += 1; complete = False
            receipt['wall_seconds'] = time.monotonic() - started
            with (args.out / 'items.jsonl').open('a') as handle:
                handle.write(json.dumps(receipt, ensure_ascii=False, allow_nan=False) + '\n'); handle.flush()
        done.add(tid)
        if complete:
            counts[next(iter(rows.values()))['split']] += 1
        status('extracting_content_passing_word_quartets', attempted_texts=len(done))
        LOG.info('Words: %s complete texts, %s successful/%s failed clips', dict(counts), successes, failures)
    status('complete_bounded_word_extraction', attempted_texts=len(done))
    if not successes:
        raise RuntimeError('No usable word features extracted')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Bulk word extraction failed; inspect explicit journal before retry')
        raise
