"""Lexical-content checks on the Swar v84 renders with the same CpuContentGate as v80."""
from __future__ import annotations

from collections import Counter
import json
import logging
from pathlib import Path

from hindi_detector.artifacts import write_group
from hindi_detector.asr_gate import CpuContentGate
from hindi_detector.corrected_synthesis import sha
from scripts.calibrate_asr_gate import audio
from scripts.check_query_content import check
from scripts.extract_corrected_training import read_rows

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'data/cross_system_sentences_v80'
PACK = ROOT / 'data/cross_system_swar_v84'
OUT = ROOT / 'reports/cross_system_swar_content_v84'
SYSTEM = 'swar_podvoices_ema'


def main():
    jobs = {j['id']: j for j in json.loads((PLAN / 'selection.json').read_text())['jobs']}
    if json.loads((PACK / 'run.json').read_text())['status'] != 'complete_corrected_audio':
        raise RuntimeError('Swar generation not complete')
    rows = read_rows(PACK / 'items.jsonl')
    if sorted(r['id'] for r in rows) != sorted(jobs):
        raise ValueError('Swar receipts differ from plan')
    write_group(OUT, {'inputs.json': {'selection_sha256': sha(PLAN / 'selection.json'), 'items_sha256': sha(PACK / 'items.jsonl'), 'script_sha256': sha(Path(__file__))}},
        'Lexical-content checks (Nepali Zipformer, Nepali IndicConformer, Hindi audit) on every Swar v84 render, same gate as reports/cross_system_content_v80. '
        'items.jsonl per-waveform hypotheses/decision; report.json counts. Created by Claude (claude-opus-5-5) Claude Code session. No pronunciation labels. Status active; consumer reports/cross_system_swar_v84; review 2026-11-17.',
        ['.venv-asr-gate/bin/python -m scripts.screen_swar_cross_system'])
    gate = CpuContentGate(ROOT, threads=2)
    (OUT / 'runtime.json').write_text(json.dumps(gate.runtime, indent=2) + '\n')
    outcomes = Counter()
    for row in rows:
        job = jobs[row['id']]
        if any(row[k] != job[k] for k in ('text', 'speaker_id', 'reference_sha256')):
            raise ValueError('Receipt differs from frozen plan: ' + row['id'])
        result = {'system': SYSTEM, 'job_id': row['id'], 'text_id': job['text_id'], 'text': job['text'], 'generation_status': row['status'], 'realized_error_label': None}
        if row['status'] == 'ok_unreviewed':
            result.update(sha256=row['sha256'], content=check(gate, audio(PACK, row), row['text']))
        else:
            result.update(content=None, error=row['error'])
        with (OUT / 'items.jsonl').open('a') as stream:
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
        outcomes[result['content']['decision'] if result['content'] else 'generation_failure'] += 1
        logging.info('%s %s', row['id'], dict(outcomes))
    (OUT / 'report.json').write_text(json.dumps({'outcomes': dict(outcomes), 'items_sha256': sha(OUT / 'items.jsonl'), 'actual_hindi_error_accuracy': None}, indent=2) + '\n')
    print(json.dumps(dict(outcomes)))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Swar content screening failed explicitly')
        raise
