"""Analyze a real WAV into timed, separate pronunciation and prosody evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from hindi_detector.artifacts import write_group
from hindi_detector.word_evidence import WordEvidenceEngine
from hindi_detector.word_model import predict_bag

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--text', required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Need a new analysis directory')
    wave, sr = sf.read(args.audio, dtype='float32', always_2d=True)
    if wave.shape[1] != 1 or not np.isfinite(wave).all():
        raise ValueError('Expected finite mono WAV; never silently mix speakers/channels')
    divisor = int(np.gcd(sr, 16000))
    samples = resample_poly(wave[:, 0], 16000 // divisor, sr // divisor).astype('float32') if sr != 16000 else wave[:, 0]
    models = json.loads(args.models.read_text())
    if set(models) != {'phonetics', 'prosody', 'combined'} or any(m.get('reference_recipe') for m in models.values()):
        raise ValueError('Expected three independent word branches; reference heads require a separate preferred reading')
    engine = WordEvidenceEngine(ROOT)
    if any(m['phone_encoder_revision'] != engine.phone_inventory['revision'] for m in models.values()):
        raise ValueError('Word checkpoint and phone encoder mismatch')
    detail, arrays = engine.analyze(samples, args.text)
    bag = {'words': detail['words'], 'phonetics': arrays['phonetics'], 'prosody': arrays['prosody'],
           'phone_mask': np.array([w['phone_evidence_available'] for w in detail['words']], dtype=bool)}
    scores = {}
    for branch, model in models.items():
        score, words = predict_bag(model, bag)
        scores[branch] = score
        for item, value in zip(detail['words'], words, strict=True):
            item[branch + '_source_evidence'] = value
    result = {'status': 'timed_weak_word_evidence_not_validated_hindi_classification',
              'audio_sha256': hashlib.sha256(args.audio.read_bytes()).hexdigest(),
              'models_sha256': hashlib.sha256(args.models.read_bytes()).hexdigest(),
              'bag_source_evidence': scores, 'validated_word_hindi_accuracy': None,
              'detail': detail, 'score_meaning': 'Evidence for the IndicF5 source bag under weak training assumptions; not a calibrated Hindi probability'}
    files = [Path(__file__), ROOT / 'hindi_detector/word_evidence.py', ROOT / 'hindi_detector/local_features.py', ROOT / 'hindi_detector/word_model.py']
    sources = [{'path': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'source_utf8': p.read_text()} for p in files]
    write_group(args.out, {'analysis.json': result, 'source_snapshot.json': sources},
                'Direct WAV word/prosody inference; exposes timing and separate evidence without inventing word gold.',
                [str(args.audio.resolve()), str(args.models.resolve()),
                 'python -m scripts.analyze_word_audio --audio mono.wav --text ... --models reports/wordlab_models_v3/models.json --out NEW'])
    print(json.dumps({'words': len(detail['words']), 'bag_source_evidence': scores, 'validated_word_hindi_accuracy': None}))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Direct word audio analysis failed; no substitute timings or scores')
        raise
