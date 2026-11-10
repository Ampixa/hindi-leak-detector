"""Measure a persistent post-utterance detector on existing development clips."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time

import numpy as np

from hindi_detector.anchored_inference import AnchoredInference
from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from scripts.calibrate_asr_gate import audio

ROOT = Path(__file__).resolve().parents[1]
CASES = {'real_yt_yt01_008': ('किनकि', 'user-confirmed occurrence'),
         'real_yt_yt03_019': ('नेकपा', 'user-confirmed occurrence; पिढको listener-similar diagnostic'),
         'real_yt_yt03_018': ('तलको', 'user-confirmed occurrence'),
         'real_yt_yt05_040': ('आइसके', 'alignment-sensitive unconfirmed diagnostic')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    source = ROOT / 'data/swar_speech_v18'
    selection = {r['id']: r for r in json.loads((source / 'selection.json').read_text())}
    inventory = json.loads((source / 'inventory.json').read_text())
    assets = {(r['id'], r['kind']): r for r in inventory['assets']}
    if not set(CASES) <= set(selection):
        raise ValueError('Development benchmark case identity absent from actual selection')
    model = ROOT / 'reports/anchored_training_v71/model.pt'
    write_group(args.out, {'inputs.json': {'model_sha256': sha(model), 'selection_sha256': sha(source / 'selection.json'),
        'asset_inventory_sha256': sha(source / 'inventory.json'), 'cases': CASES, 'script_sha256': sha(Path(__file__)),
        'scope': 'Development regressions and timing only; no independent accuracy, no streaming, no production switch.'}},
        'Persistent runtime measurement on original Swar/real recordings. items.jsonl contains raw source hashes, exact text, word/site evidence, coverage and processing times; report.json contains cold initialization and sequential completed-utterance timing. '
        'No WAV/encoder copies. These cases influenced prior development and include listener feedback; they are not held-out test data. Owner Ampixa/user; review2026-11-10, current HindiBench prototype consumer. '
        'NOT on HF; source/MMS research usage restrictions inherited. Retain compact results while benchmark consumes them; no automatic accent/prosody accuracy claim.',
        ['.venv-indicf5/bin/python -m scripts.benchmark_anchored_inference --out NEW'])
    started = time.perf_counter()
    runtime = AnchoredInference(ROOT, model)
    cold = time.perf_counter() - started
    rows = []
    for identifier, (target, feedback) in CASES.items():
        for kind in ('swar', 'real'):
            asset = assets[identifier, kind]
            wave = audio(source, {'audio': asset['path'], 'sha256': asset['sha256']})
            result = runtime.analyze(wave, selection[identifier]['text'])
            row = {'id': identifier, 'kind': kind, 'target': target, 'feedback_scope': feedback,
                   'audio_sha256': asset['sha256'], **result}
            rows.append(row)
            with (args.out / 'items.jsonl').open('a') as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            print(json.dumps({'id': identifier, 'kind': kind, 'timing': result['timing_seconds'],
                'coverage': result['coverage'], 'target_words': [w for w in result['words'] if w['word'] in {target, 'पिढको'}]}, ensure_ascii=False), flush=True)
    times = [r['timing_seconds']['total'] for r in rows]
    rtfs = [r['timing_seconds']['real_time_factor'] for r in rows]
    report = {'status': 'complete_post_utterance_runtime_benchmark', 'model_sha256': sha(model), 'cold_initialization_seconds': cold,
              'completed_utterances': len(rows), 'processing_seconds_median': float(np.median(times)),
              'processing_seconds_p95': float(np.quantile(times, .95)), 'real_time_factor_median': float(np.median(rtfs)),
              'real_time_factor_p95': float(np.quantile(rtfs, .95)), 'streaming': False,
              'actual_hindi_error_accuracy': None, 'production_changed': False,
              'includes': 'Cold initialization hashes and loads all three encoders. Per-utterance timing includes alignment, both phone encoders, existing prosody extraction, G2P and local CTC/head; excludes file read/resampling. First utterance may include warmup. No network/upload time.',
              'items_sha256': sha(args.out / 'items.jsonl')}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Post-utterance benchmark failed explicitly; no streaming claim or promotion')
        raise
