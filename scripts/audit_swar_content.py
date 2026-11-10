"""Audit actual review/query content separately from all pronunciation scores."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import traceback

from hindi_detector.artifacts import write_group
from hindi_detector.asr_gate import CpuContentGate
from hindi_detector.corrected_synthesis import sha
from scripts.calibrate_asr_gate import audio
from scripts.check_query_content import check
from scripts.extract_corrected_training import read_rows

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / 'reports/swar_content_audit_v41'
    if out.exists():
        raise FileExistsError(out)
    features = ROOT / 'reports/swar_corpus_features_v18'
    pack = ROOT / 'data/swar_speech_v18'
    rows = read_rows(features / 'items.jsonl')
    write_group(out, {'protocol.json': {'input_receipts_sha256': sha(features / 'items.jsonl'),
        'clips': len(rows), 'selection': 'All258 already scored actual recordings, no score-based filtering',
        'model_changes': False, 'content_decision': 'Original frozen Native Zipformer/IndicConformer recipe; Hindi Conformer retained as additional audit, never accent gold'},
        'source_snapshot.json': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()}
            for p in (Path(__file__), ROOT / 'scripts/check_query_content.py', ROOT / 'hindi_detector/asr_gate.py')]},
        'Input-content audit of all actual86 same-text triples. items.jsonl exact WAV/text, '
        'three recognizer hypotheses and WER/CER/roman-CER with pass/quarantine/reject; '
        'runtime.json verified recognizer/source identities; report.json source counts and '
        'known examples. No pronunciation/source labels, new waveform or model copies; no '
        'training or detector thresholds changed. Current consumer misalignment/garbling '
        'guard and uncertainty; review2026-11-10. Keep compact content receipts with release, '
        'transient logs expire24h after consumer closes. NOT on HF; inherited source rights.',
        ['.venv-asr-gate/bin/python -m scripts.audit_swar_content'])

    def status(phase, **values):
        p = out / 'status.json'
        temp = p.with_suffix('.partial')
        temp.write_text(json.dumps({'phase': phase, 'pid': os.getpid(), 'updated_utc': datetime.now(timezone.utc).isoformat(), **values}, indent=2) + '\n')
        temp.replace(p)

    try:
        status('loading_verified_content_recognizers')
        gate = CpuContentGate(ROOT, threads=2)
        (out / 'runtime.json').write_text(json.dumps(gate.runtime, indent=2) + '\n')
        results = []
        for row in rows:
            result = check(gate, audio(pack, {'audio': row['path'], 'sha256': row['sha256']}), row['text'])
            record = {k: row[k] for k in ('id', 'kind', 'text', 'path', 'sha256')}
            record.update(result)
            results.append(record)
            with (out / 'items.jsonl').open('a') as handle:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            status('checking_actual_recorded_content', completed=len(results), planned=len(rows))
        report = {'status': 'complete_actual_content_audit', 'clips': len(rows),
            'decisions': dict(Counter(row['kind'] + '/' + row['decision'] for row in results)),
            'known_development_cases': [{k: row[k] for k in ('id', 'kind', 'decision')} for row in results
                if row['id'] in {'real_yt_yt01_005', 'real_yt_yt01_008'}],
            'actual_hindi_error_accuracy': None,
            'limits': ['ASR agreement checks intended lexical content, not word-level pronunciation or Hindi influence.',
                       'English loans, transcript spelling and code switching can cause ASR quarantine even for intelligible speech.']}
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        status(report['status'], completed=len(results))
        print(json.dumps(report))
    except Exception:
        status('failed_explicitly', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Actual recorded-content audit failed explicitly')
        raise
