"""Strictly screen new generated Native references with the isolated CPU ASR gate."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from hindi_detector.artifacts import write_group
from hindi_detector.asr_gate import CpuContentGate
from hindi_detector.corrected_synthesis import sha
from hindi_detector.teacher_reference_cache import verified_paths
from scripts.calibrate_asr_gate import audio

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--job-ids', type=Path, help='Explicit verified subset of source job IDs; no waveform copy')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    jobs = json.loads((args.plan / 'selection.json').read_text())['jobs']
    if args.job_ids is not None:
        requested = json.loads(args.job_ids.read_text())
        if (not isinstance(requested, list) or not requested or any(not isinstance(key, str) for key in requested)
                or len(requested) != len(set(requested)) or not set(requested) <= {job['id'] for job in jobs}):
            raise ValueError('Explicit content-screening subset has missing/duplicate/unknown job IDs')
        jobs = [job for job in jobs if job['id'] in set(requested)]
    paths = verified_paths(args.plan, jobs)
    gate = CpuContentGate(ROOT, threads=2)
    rows = []
    for job, path in zip(jobs, paths, strict=True):
        digest = sha(path)
        result = gate.score(audio(path.parent, {'audio': path.name, 'sha256': digest}),
                            job['text'], 'pocket_teacher_250000_24l')
        rows.append({'id': job['id'], 'text': job['text'], 'sha256': digest, **result})
    # Quarantined content is not a sufficiently trusted new-text reference.
    write_group(args.out, {'result.json': {'accepted_ids': [row['id'] for row in rows if row['decision'] == 'pass'],
        'rows': rows, 'runtime': gate.runtime, 'pronunciation_gold': False},
        'source_snapshot.json': [{'path': str(path.relative_to(ROOT)), 'sha256': sha(path),
            'source_utf8': path.read_text()} for path in (Path(__file__), ROOT / 'hindi_detector/asr_gate.py')]},
        'New-text Native reference lexical-content screening. Only content pass is accepted; '
        'quarantine/reject are retained and excluded. Exact hypotheses, WER/CER, model identities '
        'and source SHA retained. Content pass is never a Hindi/accent label. Current consumer '
        'generic reference inference; compact receipts retained with active reference cache.',
        [str(args.plan), '.venv-asr-gate/bin/python -m scripts.screen_teacher_reference --plan PLAN --out NEW'])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Generated Native reference content screening failed explicitly')
        raise
