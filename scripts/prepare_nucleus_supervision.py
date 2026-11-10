"""Add train/dev loss-only vowel-region targets to immutable v81 acoustics."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.nucleus_deletion import supervision
from scripts.train_anchored_detector import read_partition

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT / 'reports/acoustic_features_v81_reviewed'
    out = ROOT / 'reports/nucleus_targets_v82_reviewed'
    if out.exists():
        raise FileExistsError(out)
    write_group(out, {'protocol.json': {'input_cache_protocol_sha256': sha(base / 'protocol.json'),
        'target_builder_sha256': sha(ROOT / 'hindi_detector/nucleus_deletion.py'),
        'supervision': 'Loss only: retained original/WORLD/sham nuclei have positive central-region targets; removed nuclei have negative40ms splice-neighborhood targets. S1 coordinates shifted only in target labels after off-target cut.',
        'unknown_frames': 'All other frames have zero loss weight, not a negative phoneme label.',
        'boundary_caveat': 'Coordinates describe known physical manipulations and source-probe regions, not human-verified phoneme boundaries.',
        'input_boundary': 'No edit/source-region/target/mask arrays enter network.forward; same query-only v81 mel/context/local14 inputs. Targets and masks consumed only by auxiliary loss.',
        'partitions': ['train', 'development'], 'test_read': False}},
        'Auxiliary training-only vowel-region labels. PART.npz target/mask arrays128bins aligned exactly to immutable measured v81 rows; PART.inputs.json row/cache/source receipt hashes and class counts; report.json completeness. '
        'No duplicated mel/WAV/model encoder assets. Owner Ampixa/user; review2026-11-10, active consumer localized training v82. NOT on HF, inherited research rights. Keep compact label provenance and selected head; targets disposable24h after reproducibility consumer closes.',
        ['.venv-indicf5/bin/python -m scripts.prepare_nucleus_supervision'])
    report = {}
    for part in ('train', 'development'):
        cache = json.loads((base / (part + '.index.json')).read_text())
        cache = [r for r in cache if r['status'] == 'measured']
        rows, _, inputs = read_partition(part)
        lookup = {(r['pair_id'], r['family']): r for r in rows}
        directory = ROOT / ('reports/native_main_controls_' + part + '_v71')
        original_lengths = {r['pair_id']: sf.info(directory / r['audio']).frames
                            for r in rows if r['family'] == 'S_original'}
        targets, masks = [], []
        for index, record in enumerate(cache):
            if record['array_index'] != index:
                raise ValueError('Auxiliary target/input row order differs')
            row = lookup[record['pair_id'], record['family']]
            if row['audio_sha256'] != record['audio_sha256']:
                raise ValueError('Known intervention label belongs to a different waveform')
            path = directory / row['audio']
            if sha(path) != record['audio_sha256']:
                raise ValueError('Actual auxiliary waveform checksum differs')
            info = sf.info(path)
            if info.samplerate != 16000 or info.channels != 1:
                raise ValueError('Auxiliary coordinates require original16k mono control')
            start, end = row['production_word']['start'], row['production_word']['end']
            left = max(0, int(np.floor((start - .04) * 16000)))
            right = min(info.frames, int(np.ceil((end + .04) * 16000)))
            expected = record['diagnostic_query_hints']
            if abs((right - left) / 16000 - expected['observed_crop_seconds']) > 1e-12 or abs(end - start - expected['observed_word_seconds']) > 1e-12:
                raise ValueError('Loss target and immutable acoustic crop geometry differ')
            if row['family'] in {'S0', 'S_world'} and info.frames != original_lengths[row['pair_id']]:
                raise ValueError('Retained control waveform length differs from original')
            target, mask = supervision(row, info.frames)
            targets.append(target); masks.append(mask)
        path = out / (part + '.npz')
        np.savez_compressed(path, targets=np.stack(targets), masks=np.stack(masks))
        report[part] = {'rows': len(targets), 'sha256': sha(path), 'input_index_sha256': sha(base / (part + '.index.json'))}
        (out / (part + '.inputs.json')).write_text(json.dumps({'source_inputs': inputs, **report[part]}, ensure_ascii=False, indent=2) + '\n')
    (out / 'report.json').write_text(json.dumps({'status': 'complete_loss_only_supervision', 'partitions': report}, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Loss-only nucleus target construction failed explicitly; no target geometry enters inference')
        raise
