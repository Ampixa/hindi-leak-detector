"""Follow owned synthesis journals and score content without waiting for all audio."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import time

from hindi_detector.artifacts import write_group
from hindi_detector.asr_gate import CpuContentGate, RECIPE
from scripts.calibrate_asr_gate import audio

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--producer', action='append', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--calibration-texts', type=int)
    parser.add_argument('--cutoff-utc', default='2026-11-08T01:00:00+00:00')
    args = parser.parse_args()
    if args.out.exists() or args.calibration_texts is not None and args.calibration_texts < 1:
        raise ValueError('Need a fresh scored-content group and positive optional calibration limit')
    cutoff = datetime.fromisoformat(args.cutoff_utc)
    if cutoff.tzinfo is None:
        raise ValueError('ASR cutoff needs a timezone')
    for producer in args.producer:
        if not (producer / 'plan.json').is_file() or not (producer / 'README.md').is_file():
            raise ValueError('Only actual documented owned synthesis producers are accepted')
    files = [Path(__file__), ROOT / 'hindi_detector/asr_gate.py', ROOT / 'hindi_detector/content_quality.py', ROOT / 'scripts/calibrate_asr_gate.py']
    write_group(args.out, {'inputs.json': {'producers': [str(p.resolve()) for p in args.producer],
        'plan_hashes': [sha(p / 'plan.json') for p in args.producer], 'recipe': RECIPE,
        'calibration_texts': args.calibration_texts, 'cutoff_utc': cutoff.isoformat()},
        'source_snapshot.json': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()} for p in files]},
        'Score real paired TTS with pinned deployed Zipformer and Hindi/Nepali IndicConformer. items.jsonl retains actual transcripts, WER/CER/roman-CER, pass/reject/quarantine, SHA/timing. status.json is live PID/progress, runtime.json exact models. Content decisions are not Hindi-error labels; no input WAV deletion.',
        ['Pinned isolated .venv-asr-gate runtime', 'Explicit producers/plans; no reference text prompts sent to recognizers',
         'Script-invariant roman CER is only a content check; original marks and acoustic targets remain unchanged.'])
    gate = CpuContentGate(ROOT, threads=2)
    (args.out / 'runtime.json').write_text(json.dumps(gate.runtime, indent=2) + '\n')
    offsets = dict.fromkeys(args.producer, 0); pending_bytes = dict.fromkeys(args.producer, b'')
    seen = set(); chosen_texts = set(); counts = Counter(); systems = Counter(); processed = 0

    def status(phase: str, **extra):
        path = args.out / 'status.json'; temporary = path.with_suffix('.partial')
        temporary.write_text(json.dumps({'phase': phase, 'pid': os.getpid(), 'processed': processed, 'decisions': dict(counts),
            'systems': dict(systems), 'updated_utc': datetime.now(timezone.utc).isoformat(), **extra}, indent=2) + '\n')
        temporary.replace(path)

    status('scoring_streaming_paired_tts')
    finished = False
    while not finished:
        fresh = 0; producer_states = []
        for producer in args.producer:
            path = producer / 'items.jsonl'
            if path.exists():
                with path.open('rb') as handle:
                    if path.stat().st_size < offsets[producer]:
                        raise ValueError('Producer journal was truncated')
                    handle.seek(offsets[producer]); body = pending_bytes[producer] + handle.read(); offsets[producer] = handle.tell()
                lines = body.split(b'\n'); pending_bytes[producer] = lines.pop()
                for line in lines:
                    row = json.loads(line)
                    if row['status'] == 'failed_waveform_quality':
                        continue
                    if row['status'] != 'ok_unreviewed' or row['system'] not in {'pocket_hindi_parent', 'pocket_nepali_v1c'}:
                        raise ValueError('Unexpected actual synthesis receipt')
                    if row['clip_id'] in seen:
                        raise ValueError('Duplicate source clip identity in ASR stream')
                    seen.add(row['clip_id'])
                    if args.calibration_texts is not None:
                        if row['split'] != 'train':
                            continue
                        if row['text_id'] not in chosen_texts:
                            if len(chosen_texts) >= args.calibration_texts:
                                continue
                            chosen_texts.add(row['text_id'])
                    samples = audio(producer, row); started = time.monotonic()
                    result = gate.score(samples, row['text'], row['system'])
                    record = {**row, 'source_pack': str(producer.resolve()), 'content_quality': result,
                              'asr_wall_seconds': time.monotonic() - started, 'status': 'content_scored_not_pronunciation_gold'}
                    with (args.out / 'items.jsonl').open('a') as handle:
                        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n'); handle.flush()
                    processed += 1; fresh += 1; counts[result['decision']] += 1; systems[row['system']] += 1
                    if processed % 32 == 0:
                        status('scoring_streaming_paired_tts'); LOG.info('Content scored %s clips; %s', processed, dict(counts))
                    if datetime.now(timezone.utc) >= cutoff:
                        finished = True; break
                if finished:
                    break
            state = json.loads((producer / 'run.json').read_text()); producer_states.append(state)
        if finished:
            status('cutoff_partial_content_scoring'); break
        if args.calibration_texts is not None and len(chosen_texts) >= args.calibration_texts:
            # Calibration selects early training texts from immutable producer order;
            # all four expected source/conditioning readings must have arrived.
            expected = len(chosen_texts) * 4
            if processed >= expected:
                status('training_only_calibration_complete'); break
        terminal = {'complete_paired_audio_with_explicit_quality_rejections', 'deadline_cutoff_partial_generation'}
        if not fresh and all(s['status'] in terminal for s in producer_states):
            if any(pending_bytes.values()):
                raise ValueError('Terminal producer journal ends with an incomplete record')
            status('complete_content_scoring_with_explicit_decisions'); break
        if not fresh:
            live = []
            for producer, state in zip(args.producer, producer_states, strict=True):
                if state['status'] in terminal:
                    continue
                if state['status'].startswith('failed'):
                    raise RuntimeError('Owned synthesis producer failed; inspect actual traceback')
                observed = subprocess.run(['ps', '-p', str(state['pid']), '-o', 'command='], capture_output=True, text=True)
                if observed.returncode != 0 or 'scripts.run_bulk_pocket' not in observed.stdout or str(producer.resolve()) not in observed.stdout:
                    raise RuntimeError('Producer handle is not live and no terminal success is recorded')
                live.append(state['pid'])
            status('verified_wait_for_live_synthesis', verified_live_producers=live)
            time.sleep(5)
    write_group(args.out / 'result', {'report.json': {'processed': processed, 'decisions': dict(counts), 'systems': dict(systems),
                 'calibration_unique_texts': len(chosen_texts), 'recipe': RECIPE, 'actual_hindi_accuracy': None}},
                'Actual content-quality decisions and coverage; original TTS audio/phonetic hypotheses retained separately. No word-error gold supplied by ASR.', ['Parent items.jsonl/runtime.json/source snapshot'])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Streaming content scoring failed; inspect partial outputs before retry')
        raise
