"""Cache pinned pretrained5ms query features; no edit geometry in extraction."""
from __future__ import annotations

import json
import logging
from pathlib import Path
import time

import numpy as np
import torch
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2FeatureEncoder

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.stem_deletion import interpolate_stem
from scripts.calibrate_asr_gate import audio
from scripts.train_anchored_detector import read_partition

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / 'reports/stem_features_v83'
    base = ROOT / 'reports/acoustic_features_v81_reviewed'
    if out.exists():
        raise FileExistsError(out)
    checkpoint = ROOT / 'models/phoneme_probe'
    inventory = json.loads((checkpoint / 'inventory.json').read_text())
    for member in inventory['files']:
        path = checkpoint / member['path']
        if path.stat().st_size != member['bytes'] or sha(path) != member['sha256']:
            raise ValueError('Pinned speech checkpoint changed')
    config = Wav2Vec2Config.from_pretrained(checkpoint, local_files_only=True)
    if list(config.conv_kernel[:5]) != [10, 3, 3, 3, 3] or list(config.conv_stride[:5]) != [5, 2, 2, 2, 2] or config.feat_extract_norm != 'layer' or list(config.conv_dim[:5]) != [512] * 5:
        raise ValueError('Unverified pretrained frontend geometry')
    stem = Wav2Vec2FeatureEncoder(config)
    state = torch.load(checkpoint / 'pytorch_model.bin', map_location='cpu', weights_only=True, mmap=True)
    prefix = 'wav2vec2.feature_extractor.'
    stem.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
    del state
    layers = torch.nn.Sequential(*list(stem.conv_layers)[:5]).eval().to('mps')
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(checkpoint, local_files_only=True)
    if extractor.sampling_rate != 16000:
        raise ValueError('Incorrect pinned extractor sample rate')
    torch.set_num_threads(2)
    write_group(out, {'protocol.json': {'schema': 'query_stem_features_v83', 'checkpoint_inventory': inventory,
        'input_cache_report_sha256': sha(base / 'report.json'), 'code_sha256': sha(ROOT / 'hindi_detector/stem_deletion.py'),
        'builder_sha256': sha(Path(__file__)), 'feature_recipe': 'Pinned XLSR waveform normalization; first5frozen conv layers,512channels,RF160samples,stride80samples. Whole query waveform only. Sample onto identical v81 centered-STFT/interpolation physical128-bin grid. Append unchanged two observed hint channels; float16 storage,float32 fit.',
        'partition_limit': 'train/development only; no old test,Swar,cross-system v80 used in fitting cache',
        'boundary': 'Only query waveform, observed production word boundaries and cached query hints used. Edit/source-region/mapped-word/source identity never input to extractor or network.'}},
        'Pretrained high-resolution query feature cache. PART.npz contains512acoustic channels only; PART.receipt.json input row/cache/source hashes and feature checksum; report/status.json completeness/progress. Hint/context/phone/labels remain immutable v81 cache and auxiliary labels remain separately v82 targets. No encoder/WAV copies. '
        'Codex GPT-6 /root, owner Ampixa/user. Active consumer v83 fixed controlled fit; review2026-11-10. NOT on HF, inherited checkpoint/source research rights. Cache disposable24h after fit/reproduction consumer closes; compact receipt/protocol retained.',
        ['.venv-indicf5/bin/python -m scripts.prepare_stem_features'])
    result = {}
    for part in ('train', 'development'):
        index = json.loads((base / (part + '.index.json')).read_text())
        index = [r for r in index if r['status'] == 'measured']
        rows, _, pins = read_partition(part)
        lookup = {(r['pair_id'], r['family']): r for r in rows}
        values = np.empty((len(index), 512, 128), np.float16)
        directory = ROOT / ('reports/native_main_controls_' + part + '_v71')
        started = time.monotonic()
        for number, record in enumerate(index):
            row = lookup[record['pair_id'], record['family']]
            if record['array_index'] != number or row['audio_sha256'] != record['audio_sha256']:
                raise ValueError('Observed feature source/input row differs')
            wave = audio(directory, {'audio': row['audio'], 'sha256': row['audio_sha256']})
            bounds = (row['production_word']['start'], row['production_word']['end'])
            with torch.inference_mode():
                x = extractor(wave, sampling_rate=16000, return_tensors='pt').input_values.to('mps')
                raw = layers(x[:, None])[0].cpu().numpy()
            values[number] = interpolate_stem(raw, bounds, len(wave))
            if not np.isfinite(values[number]).all():
                raise ValueError('Nonfinite frozen frontend feature')
            if number % 100 == 0:
                status = {'partition': part, 'processed': number + 1, 'expected': len(index), 'elapsed_seconds': time.monotonic() - started}
                (out / 'status.json').write_text(json.dumps(status) + '\n')
                print(json.dumps(status), flush=True)
        path = out / (part + '.npz')
        np.savez_compressed(path, acoustic=values)
        result[part] = {'rows': len(index), 'sha256': sha(path), 'input_index_sha256': sha(base / (part + '.index.json'))}
        (out / (part + '.receipt.json')).write_text(json.dumps({'source_inputs': pins, **result[part]}, indent=2) + '\n')
    (out / 'report.json').write_text(json.dumps({'status': 'complete_pretrained_stem_cache', 'partitions': result}, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Frozen early frontend extraction failed explicitly; no substitute checkpoint')
        raise
