"""Follow the frozen v80 render journals with persistent lexical ASR checks."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import time
import traceback

from hindi_detector.artifacts import write_group
from hindi_detector.asr_gate import CpuContentGate
from hindi_detector.corrected_synthesis import sha
from scripts.calibrate_asr_gate import audio
from scripts.check_query_content import check
from scripts.extract_corrected_training import read_rows

ROOT = Path(__file__).resolve().parents[1]


def main():
    plan = ROOT / 'data/cross_system_sentences_v80'
    out = ROOT / 'reports/cross_system_content_v80'
    if out.exists():
        raise FileExistsError(out)
    jobs = {j['id']: j for j in json.loads((plan / 'selection.json').read_text())['jobs']}
    systems = {'pocket_teacher_250000_24l': plan / 'teacher', 'indicf5_stock': plan / 'indicf5'}
    write_group(out, {'inputs.json': {'selection_sha256': sha(plan / 'selection.json'), 'script_sha256': sha(Path(__file__)),
        'source_roots': {k: str(v) for k, v in systems.items()}, 'expected_rows': len(jobs) * 2}},
        'Lexical-content checks on every fresh v80 output using persistent Nepali Zipformer, Nepali IndicConformer and Hindi audit. items.jsonl exact waveform/text/source/hash and all recognizer hypotheses or generation failure; '
        'runtime.json pins deployed ASR; status.json atomic counts/PID; report.json full denominators; failure.json explicit failure if stopped. No phonetic gold, filtering of generation receipts or model fitting. Owner Ampixa/user; review2026-11-10. '
        'NOT on HF, inherited rights; current consumer cross-system sentence scoring. Retain compact checks while audio is consumed, no WAV/model copies here.',
        ['.venv-asr-gate/bin/python -m scripts.screen_cross_system_sentences'])
    gate = CpuContentGate(ROOT, threads=2)
    (out / 'runtime.json').write_text(json.dumps(gate.runtime, indent=2) + '\n')
    seen, records = set(), []
    deadline = time.monotonic() + 1800
    while len(seen) < len(jobs) * len(systems):
        fresh = 0
        for system, directory in systems.items():
            for row in read_rows(directory / 'items.jsonl'):
                identity = (system, row['id'])
                if identity in seen:
                    continue
                job = jobs[row['id']]
                if row['system'] != system or any(row[k] != job[k] for k in ('text', 'speaker_id', 'reference_sha256')):
                    raise ValueError('Actual fresh generation receipt differs from frozen plan')
                result = {'system': system, 'job_id': row['id'], 'text_id': job['text_id'], 'text': job['text'],
                    'generation_status': row['status'], 'realized_error_label': None}
                if row['status'] == 'ok_unreviewed':
                    started = time.monotonic()
                    result.update(sha256=row['sha256'], content=check(gate, audio(directory, row), row['text']),
                                  asr_seconds=time.monotonic() - started)
                elif row['status'] == 'failed_waveform_quality':
                    result.update(content=None, error=row['error'])
                else:
                    raise ValueError('Unknown generation state: ' + row['status'])
                with (out / 'items.jsonl').open('a') as stream:
                    stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                seen.add(identity); records.append(result); fresh += 1
        status = {'phase': 'complete_content_checks' if len(seen) == len(jobs) * 2 else 'following_actual_generation',
            'pid': os.getpid(), 'updated_utc': datetime.now(timezone.utc).isoformat(), 'processed': len(seen), 'expected': len(jobs) * 2,
            'outcomes': dict(Counter(r['system'] + '/' + (r['content']['decision'] if r['content'] else 'generation_failure') for r in records))}
        temporary = out / 'status.partial'
        temporary.write_text(json.dumps(status, indent=2) + '\n'); temporary.replace(out / 'status.json')
        for directory in systems.values():
            path = directory / 'run.json'
            if path.exists() and json.loads(path.read_text())['status'] == 'failed_explicitly':
                raise RuntimeError('A generation source failed explicitly; inspect before any resume: ' + str(path))
        if time.monotonic() > deadline:
            raise TimeoutError('Generation/content pipeline did not finish within30minutes; partial receipts retained')
        if not fresh and len(seen) < len(jobs) * 2:
            time.sleep(3)
    (out / 'report.json').write_text(json.dumps({**status, 'items_sha256': sha(out / 'items.jsonl'),
        'actual_hindi_error_accuracy': None}, indent=2) + '\n')
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Cross-system content stream failed explicitly; no invented pronunciation label')
        directory = ROOT / 'reports/cross_system_content_v80'
        if directory.exists():
            (directory / 'failure.json').write_text(json.dumps({'error': traceback.format_exc(), 'created_utc': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')
        raise
