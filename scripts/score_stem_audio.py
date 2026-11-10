"""Offline v83 word-deletion research inference on an arbitrary text/WAV pair."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch

from hindi_detector.acoustic_deletion import WordAcoustics
from hindi_detector.anchored_inference import AnchoredInference
from hindi_detector.corrected_synthesis import sha
from hindi_detector.stem_deletion import StemVowelNetwork, SCHEMA, interpolate_stem

ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA = 'ae98b0048e0103e8f5c136b07d3bd68d842bb0efe2eb3a9ab2d3c64abe3d858a'
LOG = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', required=True, type=Path)
    parser.add_argument('--text-file', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f'Refusing to overwrite output: {args.out}')
    text = args.text_file.read_text().strip()
    if not text:
        raise ValueError('Input text is empty')
    samples, rate = sf.read(args.audio, dtype='float32')
    if samples.ndim != 1 or not np.isfinite(samples).all() or len(samples) == 0:
        raise ValueError('Provide finite, nonempty mono WAV audio')
    input_rate = rate
    if rate != 16000:
        from math import gcd
        divisor = gcd(rate, 16000)
        samples = resample_poly(samples, 16000 // divisor, rate // divisor).astype(np.float32)
        rate = 16000
    model = ROOT / 'reports/stem_training_v83/model.pt'
    if sha(model) != MODEL_SHA:
        raise ValueError('The pinned development head differs from the demonstrated v83 model')
    state = torch.load(model, map_location='cpu', weights_only=False)
    if state['schema'] != SCHEMA:
        raise ValueError('Unexpected model schema')
    for name, digest in {**state['code_sha256'], **state['resource_files']}.items():
        if sha(ROOT / name) != digest:
            raise ValueError('Frozen inference input differs: ' + name)
    started = time.perf_counter()
    network = StemVowelNetwork(len(state['phone_vocabulary']))
    network.load_state_dict(state['weights'], strict=True)
    network.eval()
    frontend = WordAcoustics(state['phone_vocabulary'], state['phone_configurations'])
    runtime = AnchoredInference(ROOT, ROOT / 'reports/anchored_training_v71/model.pt')
    if any(runtime.state[k] != state[k] for k in ('phone_inventory', 'secondary_phone_inventory', 'aligner_inventory')):
        raise ValueError('Adapter and candidate use different encoders/aligner')
    if runtime.engine.phone_inventory != state['stem_recipe']['checkpoint_inventory']:
        raise ValueError('Acoustic encoder differs from frozen candidate')
    evidence = runtime.analyze(samples, text)
    stem_layers = torch.nn.Sequential(*list(runtime.engine.phone.wav2vec2.feature_extractor.conv_layers)[:5]).eval()
    with torch.inference_mode():
        values = runtime.engine.extractor(samples, sampling_rate=16000, return_tensors='pt').input_values.to(runtime.engine.device)
        raw = stem_layers(values[:, None])[0].cpu().numpy()
    words = []
    for ordinal, word in enumerate(evidence['words']):
        record = {'word': word['word'], 'ordinal': ordinal, 'start': word['start'], 'end': word['end'],
                  'sites': [], 'abstentions': list(word['unavailable_reasons'])}
        for site in word['sites']:
            try:
                features, context, phone, geometry = frontend.extract(samples, (word['start'], word['end']), site['evidence'], site['native_phones'], site['native_phone_index'], site['features'])
                stem = interpolate_stem(raw, (word['start'], word['end']), len(samples)).astype(np.float16).astype(np.float32)
                features = np.concatenate([stem, features[64:66]], axis=0)
                phone = np.clip((phone - state['scaler']['mean']) / state['scaler']['scale'], -8, 8).astype(np.float32)
                with torch.inference_mode():
                    logit = float(network(torch.from_numpy(features[None]), torch.from_numpy(context[None]), torch.from_numpy(phone[None])).item())
                record['sites'].append({'phone_index': site['native_phone_index'], 'logit': logit,
                                        'candidate': logit >= state['logit_threshold'], 'query_geometry': geometry})
            except ValueError as error:
                LOG.warning('Abstaining on %s site %s: %s', word['word'], site['native_phone_index'], error)
                record['abstentions'].append(str(error))
        record['tested'] = bool(record['sites'])
        record['candidate'] = any(s['candidate'] for s in record['sites'])
        words.append(record)
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'creator': 'Codex /root',
              'status': 'offline_development_candidate_inference', 'text': text,
              'audio_sha256': sha(args.audio), 'input_sample_rate': input_rate,
              'analysis_sample_rate': rate, 'model_sha256': MODEL_SHA,
              'script_sha256': sha(Path(__file__)), 'words': words,
              'processing_seconds_including_load': time.perf_counter() - started,
              'actual_hindi_error_probability': None, 'streaming': False,
              'limits': ['Internal-vowel deletion candidate only, not general Hindi influence or speaker identity.',
                         'No tested contrast does not establish correct pronunciation.',
                         'No independent accuracy or sentence-prosody claim.']}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.out), 'candidates': [w['word'] for w in words if w['candidate']],
                      'tested_words': sum(w['tested'] for w in words)}, ensure_ascii=False))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Offline inference failed; no substitute result was produced')
        raise
