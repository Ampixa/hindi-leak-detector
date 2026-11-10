"""Frozen features for full speech and short views; no retained waveform copies."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, AutoModelForCTC

from hindi_detector.evaluation import ENCODER_REVISION
from hindi_detector.proxy import load_pack
from hindi_detector.review_server import read_items

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--pack', type=Path, action='append', required=True)
    parser.add_argument('--existing', type=Path, action='append')
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.existing and len(args.existing) != len(args.pack):
        raise ValueError('Each pack requires its matching feature directory')
    if not (args.out / 'README.md').is_file() or (args.out / 'manifest.json').exists():
        raise ValueError('Need a documented new feature output')
    driver = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    model_path = ROOT / 'models/phoneme_probe'
    inventory = json.loads((model_path / 'inventory.json').read_text())
    if inventory['revision'] != ENCODER_REVISION:
        raise ValueError('Wrong encoder revision')
    for item in inventory['files']:
        p = model_path / item['path']
        if p.stat().st_size != item['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Encoder checksum mismatch')
    torch.set_num_threads(2)
    extractor = AutoFeatureExtractor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCTC.from_pretrained(model_path, local_files_only=True).eval()
    records = []
    for pack_index, pack in enumerate(args.pack):
        items, _ = read_items(pack)
        cached = {}
        if args.existing:
            old_rows, old_x = load_pack(args.existing[pack_index], pack)
            cached = {r['clip_id']: x for r, x in zip(old_rows, old_x, strict=True)}
        for clip_id, row in sorted(items.items()):
            path = pack / row['audio']
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Source WAV identity mismatch')
            wave, sr = sf.read(path, dtype='float32', always_2d=True)
            if wave.shape[1] != 1 or not np.isfinite(wave).all():
                raise ValueError('Invalid waveform')
            divisor = int(np.gcd(sr, 16000))
            samples = resample_poly(wave[:, 0], 16000 // divisor, sr // divisor).astype('float32') if sr != 16000 else wave[:, 0]
            windows = [('full', 0, len(samples))]
            views = [('short_1p5', 1.5, .5)] if args.probe else [('short_1p5', 1.5, .3), ('short_2p5', 2.5, .7)]
            for name, seconds, center in views:
                length = round(seconds * 16000)
                if len(samples) <= length:
                    raise ValueError('Expected source longer than every short view')
                start = max(0, min(len(samples) - length, round(len(samples) * center - length / 2)))
                windows.append((name, start, start + length))
            for variant, start, end in windows:
                identifier = clip_id + '_' + variant
                destination = args.out / (identifier + '.npz')
                if destination.exists():
                    raise FileExistsError('Existing window cache; inspect partial run before retry')
                if variant == 'full' and clip_id in cached:
                    vector = cached[clip_id]
                else:
                    inputs = extractor(samples[start:end], sampling_rate=16000, return_tensors='pt')
                    with torch.inference_mode():
                        output = model(**inputs, output_hidden_states=True)
                    pools = []
                    for layer in [12, 24]:
                        values = output.hidden_states[layer][0].cpu().numpy()
                        pools.extend([values.mean(0), values.std(0)])
                    vector = np.concatenate(pools).astype('float32')
                if vector.shape != (4096,) or not np.isfinite(vector).all():
                    raise ValueError('Invalid window feature vector')
                np.savez_compressed(destination, speech=vector, parent_audio_sha256=np.array(row['sha256']),
                                    driver_sha256=np.array(driver), encoder_revision=np.array(ENCODER_REVISION),
                                    start_sample=np.array(start), end_sample=np.array(end))
                records.append({'id': identifier, 'parent_clip_id': clip_id, 'variant': variant,
                                'path': destination.name, 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                                'size_bytes': destination.stat().st_size, 'audio_sha256': row['sha256'],
                                'speaker_id': row['speaker_id'], 'sentence_id': row['sentence_id'],
                                'text_sha256': hashlib.sha256(re.sub(r'[\W_]+', '', unicodedata.normalize('NFC', row['text'])).encode()).hexdigest(),
                                'system': {'audio8_dpo_R3': 'audio8_r3'}.get(row['system'], row['system']),
                                'split': row['split'], 'start_sample': start, 'end_sample': end,
                                'sample_rate': 16000, 'seconds': (end - start) / 16000})
            LOG.info('Encoded %s, %s total views', clip_id, len(records))
    manifest = {'schema_version': 1, 'creator': 'Codex GPT-6 /root',
                'created_utc': datetime.now(timezone.utc).isoformat(), 'driver_sha256': driver,
                'encoder_revision': ENCODER_REVISION, 'dimension': 4096,
                'status': 'complete_features_only', 'items': records,
                'source_selections': [{'path': str(p.resolve()), 'sha256': hashlib.sha256((p / 'selection.json').read_bytes()).hexdigest()} for p in args.pack]}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'views': len(records)}))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Window feature extraction failed; inspect partial cache, never substitute features')
        raise
