"""Score a mono WAV with the actual trained source-proxy head and pinned encoder."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, AutoModelForCTC

from hindi_detector.artifacts import write_group
from hindi_detector.evaluation import ENCODER_REVISION, agreement_decisions, predict_linear
from hindi_detector.proxy import LABEL_SOURCE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--head', type=Path, required=True)
    parser.add_argument('--encoder', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--agreement-head', type=Path)
    args = parser.parse_args()
    head = json.loads(args.head.read_text())
    if head['label_source'] != LABEL_SOURCE or head['encoder_revision'] != ENCODER_REVISION:
        raise ValueError('Unsupported head/encoder or missing explicit proxy status')
    inventory = json.loads((args.encoder / 'inventory.json').read_text())
    if inventory['revision'] != ENCODER_REVISION or inventory['model'] != 'facebook/wav2vec2-xlsr-53-espeak-cv-ft':
        raise ValueError('Wrong pinned encoder')
    for row in inventory['files']:
        path = (args.encoder / row['path']).resolve(strict=True)
        if not path.is_relative_to(args.encoder.resolve()):
            raise ValueError('Encoder inventory path outside model directory')
        if path.stat().st_size != row['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Encoder file verification failed')
    wave, sr = sf.read(args.audio, dtype='float32', always_2d=True)
    if wave.shape[1] != 1 or len(wave) < sr // 5 or not np.isfinite(wave).all():
        raise ValueError('Need finite mono audio at least 0.2 seconds long')
    if not np.any(wave):
        raise ValueError('Silent input does not establish pronunciation')
    if sr != 16000:
        divisor = int(np.gcd(sr, 16000))
        samples = resample_poly(wave[:, 0], 16000 // divisor, sr // divisor).astype('float32')
    else:
        samples = wave[:, 0]
    torch.set_num_threads(2)
    extractor = AutoFeatureExtractor.from_pretrained(args.encoder, local_files_only=True)
    encoder = AutoModelForCTC.from_pretrained(args.encoder, local_files_only=True).eval()
    with torch.inference_mode():
        output = encoder(**extractor(samples, sampling_rate=16000, return_tensors='pt'), output_hidden_states=True)
    pools = []
    for layer in [12, 24]:
        values = output.hidden_states[layer][0].cpu().numpy()
        pools.extend([values.mean(0), values.std(0)])
    features = np.concatenate(pools).astype('float32')
    score = float(predict_linear(head, features[None])[0])
    result = {'status': head['status'], 'indicf5_like_score': score,
              'score_meaning': head['score_meaning'], 'label_source': LABEL_SOURCE,
              'audio_sha256': hashlib.sha256(args.audio.read_bytes()).hexdigest(),
              'head_sha256': hashlib.sha256(args.head.read_bytes()).hexdigest(),
              'encoder_revision': ENCODER_REVISION, 'seconds': len(wave) / sr,
              'word_schwa_evidence': None, 'verified_leakage_probability': None}
    if args.agreement_head:
        other = json.loads(args.agreement_head.read_text())
        if other['encoder_revision'] != ENCODER_REVISION or other['label_source'] != LABEL_SOURCE:
            raise ValueError('Agreement head has incompatible encoder/supervision')
        other_score = float(predict_linear(other, features[None])[0])
        decision = int(agreement_decisions(np.array([score]), np.array([other_score]))[0])
        result.update(agreement_head_score=other_score,
                      agreement_head_sha256=hashlib.sha256(args.agreement_head.read_bytes()).hexdigest(),
                      decision={-1: 'uncertain', 0: 'nepali_like_proxy', 1: 'hindi_like_proxy'}[decision],
                      decision_rule='Both fixed-0.5 source-proxy heads must agree; otherwise abstain')
    write_group(args.out, {'score.json': result}, 'End-to-end WAV score from the actual source-proxy model; uncalibrated.',
                [str(args.audio.resolve()), str(args.head.resolve()), str(args.encoder.resolve()),
                 'python -m scripts.score_source_proxy --audio ... --head ... --encoder ... --out NEW'])
    print(json.dumps(result))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Proxy WAV scoring failed; no substitute score')
        raise
