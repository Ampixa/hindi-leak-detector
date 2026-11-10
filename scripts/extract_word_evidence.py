"""Localize existing paired speech into auditable word and prosody evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from hindi_detector.local_features import PROSODY_NAMES
from hindi_detector.review_server import read_items
from hindi_detector.word_evidence import WordEvidenceEngine

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--pack', type=Path, action='append', required=True)
    parser.add_argument('--sentence')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--g2p-bank', type=Path)
    args = parser.parse_args()
    if not (args.out / 'README.md').is_file() or (args.out / 'manifest.json').exists() or (args.out / 'items.jsonl').exists():
        raise ValueError('Need a documented new word-evidence output group')
    jobs = []
    for pack in args.pack:
        items, _ = read_items(pack)
        jobs.extend((pack, item) for item in items.values() if args.sentence is None or item['sentence_id'] == args.sentence)
    if not jobs or len({row['clip_id'] for _, row in jobs}) != len(jobs):
        raise ValueError('Need unique nonempty input clips')
    sources = []
    for path in [Path(__file__), ROOT / 'hindi_detector/word_evidence.py', ROOT / 'hindi_detector/local_features.py', ROOT / 'hindi_detector/g2p_evidence.py', ROOT / 'hindi_detector/ctc.py']:
        content = path.read_bytes()
        sources.append({'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(content).hexdigest(),
                        'source_utf8': content.decode('utf-8')})
    (args.out / 'source_snapshot.json').write_text(json.dumps(sources, indent=2) + '\n')
    bank = json.loads(args.g2p_bank.read_text()) if args.g2p_bank else None
    engine = WordEvidenceEngine(ROOT, g2p_word_bank=bank)
    records = []
    for pack, row in jobs:
        identity = row['clip_id']; path = pack / row['audio']
        metadata = {k: row[k] for k in ['clip_id', 'system', 'speaker_id', 'sentence_id', 'text', 'split']}
        metadata['audio_sha256'] = row['sha256']
        try:
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Source waveform identity mismatch')
            wave, sr = sf.read(path, dtype='float32', always_2d=True)
            if wave.shape[1] != 1 or not np.isfinite(wave).all():
                raise ValueError('Invalid source waveform')
            divisor = int(np.gcd(sr, 16000))
            samples = resample_poly(wave[:, 0], 16000 // divisor, sr // divisor).astype('float32') if sr != 16000 else wave[:, 0]
            result, arrays = engine.analyze(samples, row['text'])
            result.update(metadata)
            detail = args.out / (identity + '.json')
            features = args.out / (identity + '.npz')
            if detail.exists() or features.exists():
                raise FileExistsError('Retained word artifacts already exist')
            detail.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
            np.savez_compressed(features, **arrays, parent_audio_sha256=np.array(row['sha256']))
            metadata.update(status='ok_unreviewed_word_evidence', words=len(result['words']),
                            detail=detail.name, detail_sha256=hashlib.sha256(detail.read_bytes()).hexdigest(),
                            detail_bytes=detail.stat().st_size, features=features.name,
                            feature_sha256=hashlib.sha256(features.read_bytes()).hexdigest(), feature_bytes=features.stat().st_size)
        except Exception:
            metadata.update(status='failed_explicitly_excluded', error=traceback.format_exc())
            LOG.exception('Word evidence failed for %s; no substitute words/scores', identity)
        with (args.out / 'items.jsonl').open('a') as handle:
            handle.write(json.dumps(metadata, ensure_ascii=False, allow_nan=False) + '\n')
        records.append(metadata)
        LOG.info('Word evidence %s/%s %s %s', len(records), len(jobs), row['system'], metadata['status'])
    manifest = {'creator': 'Codex GPT-6 /root', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'driver_sha256': sources[0]['sha256'],
                'engine_sha256': sources[1]['sha256'], 'prosody_code_sha256': sources[2]['sha256'],
                'alignment_sha256': engine.aligner_inventory['sha256'],
                'phone_encoder_revision': engine.phone_inventory['revision'],
                'phone_vocabulary': engine.vocab, 'prosody_feature_names': PROSODY_NAMES,
                'status': 'word_evidence_extracted_not_trained_hindi_word_labels', 'items': records,
                'successes': sum(r['status'] == 'ok_unreviewed_word_evidence' for r in records),
                'failures': sum(r['status'] != 'ok_unreviewed_word_evidence' for r in records)}
    manifest['g2p_bank_sha256'] = hashlib.sha256(args.g2p_bank.read_bytes()).hexdigest() if args.g2p_bank else None
    manifest['additional_source_hashes'] = {row['path']: row['sha256'] for row in sources[3:]}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if not manifest['successes']:
        raise RuntimeError('Every word extraction failed; inspect explicit errors')
    print(json.dumps({k: manifest[k] for k in ['status', 'successes', 'failures']}))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Word evidence batch could not complete')
        raise
