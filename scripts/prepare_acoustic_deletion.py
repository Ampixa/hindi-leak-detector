"""Cache train/development-only observable word acoustics, never oracle edits."""
from __future__ import annotations

from collections import Counter
import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time

import numpy as np
import torch

from hindi_detector.acoustic_deletion import RECIPE, WordAcoustics, frame_count
from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from scripts.calibrate_asr_gate import audio
from scripts.extract_corrected_training import read_rows
from scripts.train_anchored_detector import read_partition

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/acoustic_features_v81')
    out = parser.parse_args().out
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(2)
    configurations = [json.loads((ROOT / 'models' / p / 'config.json').read_text()) for p in ('phoneme_probe', 'phoneme_probe_lv60_v20')]
    # Validate convolution-clock geometry against retained actual model outputs,
    # rather than assuming a20ms frame rate or a particular frontend layout.
    source = ROOT / 'reports/native_main_features_v71'
    receipts = read_rows(source / 'items.jsonl')
    geometry_checks = []
    for row in receipts[::max(1, len(receipts) // 8)][:8]:
        detail = json.loads((source / row['detail']).read_text())
        if sha(source / row['detail']) != row['detail_sha256'] or sha(source / row['features']) != row['features_sha256']:
            raise ValueError('Observed geometry fixture bytes changed')
        samples = round(detail['seconds'] * 16000)
        with np.load(source / row['features'], allow_pickle=False) as arrays:
            counts = [len(arrays[k]) for k in ('phone_log_probs', 'secondary_phone_log_probs')]
        if counts != [frame_count(samples, c) for c in configurations]:
            raise ValueError('Observed encoder output disagrees with verified convolution clock')
        geometry_checks.append({'source_id': row['id'], 'samples': samples, 'observed_frames': counts})
    ipa = json.loads((ROOT / 'data/swar_phone_plan_v18/ipa_map.json').read_text())
    vocabulary = {phone: i for i, phone in enumerate(sorted(ipa))}
    frontend = WordAcoustics(vocabulary, configurations)
    words = {w['word_id']: w for w in json.loads((ROOT / 'data/native_main_v71/selection.json').read_text())['words']}
    write_group(out, {'protocol.json': {'recipe': RECIPE, 'phone_vocabulary': vocabulary, 'phone_configurations': configurations,
        'geometry_checks': geometry_checks, 'partitions': ['train', 'development'],
        'never_read': ['main_v71 test controls', 'Swar recordings/labels', 'cross-system v80 audio/results/labels'],
        'physical_labels': 'Known interventions only, not independent human phonetic/Hindi-error gold.',
        'feature_code_sha256': sha(ROOT / 'hindi_detector/acoustic_deletion.py'), 'script_sha256': sha(Path(__file__)),
        'query_evidence_extractor_sha256': sha(ROOT / 'hindi_detector/anchored_vowels.py')}},
        'High-resolution query-word spectral training cache. PART.npz contains float32mel66x128, int64context neighbor IDs, float32local14, float32labels and duration for an explicit separate baseline only; PART.index.json metadata/attempted statuses/source hashes. Source intervention retained-ms enters diagnostic strata only, never acoustic/phone/duration inputs. PART.inputs.json pinned control/content receipts and runtime. '
        'report.json measured/attempted counts and array SHA, status.json atomic progress. Features use only actual edited query waveform, production word boundaries, query phone traces and intended G2P triphone; never edit/mapped/source-region geometry, source or speaker IDs, or reference audio. '
        'No new WAV/encoder/checkpoint copies. Owner Ampixa/user; review2026-11-10. Active consumer acoustic deletion training v81, NOT on HF; inherited source/MMS/PanPhon rights. Cache disposable24h after fitting/reproducibility consumer closes; compact configs/hash manifest retained.',
        ['.venv-indicf5/bin/python -m scripts.prepare_acoustic_deletion'])
    reports = {}
    for part in ('train', 'development'):
        rows, rejected, inputs = read_partition(part)
        records, mel, context, phone, labels, duration = [], [], [], [], [], []
        retained = {r['pair_id']: r['edit']['replacement_samples'] / 16 for r in rows if r['family'] == 'S2'}
        directory = ROOT / ('reports/native_main_controls_' + part + '_v71')
        started = time.monotonic()
        for ordinal, row in enumerate(rows):
            record = {k: row[k] for k in ('pair_id', 'job_id', 'word_id', 'word', 'voice', 'phone_collision_group', 'split', 'family', 'label', 'status', 'audio_sha256')}
            record['array_index'] = None
            record['diagnostic_retained_ms'] = retained.get(row['pair_id'])
            if row['status'] == 'measured':
                try:
                    samples = audio(directory, {'audio': row['audio'], 'sha256': row['audio_sha256']})
                    spec = words[row['word_id']]
                    a, b, c, diagnostic = frontend.extract(samples, (row['production_word']['start'], row['production_word']['end']),
                        row['evidence'], spec['native_project_phones'], spec['site_index'], row['features'])
                    record['array_index'] = len(mel)
                    record['diagnostic_query_hints'] = diagnostic
                    mel.append(a); context.append(b); phone.append(c); labels.append(row['label'])
                    duration.append([diagnostic['observed_word_seconds']])
                except ValueError as error:
                    logging.warning('Retained acoustic abstention %s/%s: %s', row['pair_id'], row['family'], error)
                    record.update(status='abstain', reason=str(error))
            records.append(record)
            if ordinal % 100 == 0:
                (out / 'status.json').write_text(json.dumps({'phase': 'preparing_query_acoustics', 'partition': part, 'processed': ordinal + 1, 'expected': len(rows), 'updated_utc': datetime.now(timezone.utc).isoformat()}) + '\n')
        path = out / (part + '.npz')
        np.savez_compressed(path, mel=np.stack(mel), context=np.stack(context), phone=np.stack(phone), labels=np.array(labels, np.float32), duration=np.array(duration, np.float32))
        (out / (part + '.index.json')).write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
        (out / (part + '.inputs.json')).write_text(json.dumps({'qualified_source_inputs': inputs, 'unedited_content_excluded_rows': len(rejected)}, ensure_ascii=False, indent=2) + '\n')
        reports[part] = {'attempted_rows': len(records), 'measured_rows': len(mel), 'sources': len({r['pair_id'] for r in rows}),
            'lexemes': len({r['word_id'] for r in rows}), 'outcomes': dict(Counter(r['family'] + '/' + r['status'] for r in records)),
            'array_sha256': sha(path), 'index_sha256': sha(out / (part + '.index.json')), 'elapsed_seconds': time.monotonic() - started}
        print(part, json.dumps(reports[part]), flush=True)
    (out / 'report.json').write_text(json.dumps({'status': 'complete_train_development_acoustic_cache', 'partitions': reports}, indent=2) + '\n')
    (out / 'status.json').write_text(json.dumps({'phase': 'complete_acoustic_cache', 'updated_utc': datetime.now(timezone.utc).isoformat()}) + '\n')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Acoustic feature preparation failed explicitly; no silent feature substitution')
        raise
