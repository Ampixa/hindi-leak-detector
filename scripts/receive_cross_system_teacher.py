"""Resume only missing WAVs onto k2 capacity, preserving original engine lock."""
from __future__ import annotations

import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shlex
import shutil
import subprocess

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from scripts.extract_corrected_training import read_rows

ROOT = Path(__file__).resolve().parents[1]


def main():
    plan = ROOT / 'data/cross_system_sentences_v80'
    audio = plan / 'teacher'
    out = ROOT / 'reports/cross_system_teacher_stream_v80'
    if out.exists():
        raise FileExistsError(out)
    ssh = ['ssh', '-4', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
        '-o', 'UserKnownHostsFile=' + str(ROOT / 'runtime/l40_known_hosts_v14'), '-o', 'StrictHostKeyChecking=yes']
    remote = '/home/ampixa/hindi-detector-teacher-f5-v14'
    subprocess.run(['rsync', '-a', '--exclude=*.partial', '-e', shlex.join(ssh),
        'ampixa@khalti-gpu:' + remote + '/data/cross_system_sentences_v80/', str(audio) + '/'], check=True, timeout=90)
    previous = json.loads((audio / 'run.json').read_text())
    if previous['status'] != 'failed_explicitly' or 'Preserve20GiB free' not in previous['error']:
        raise ValueError('Streaming transport recovery only accepts inspected capacity failure')
    selection = json.loads((plan / 'selection.json').read_text())
    jobs = {j['id']: j for j in selection['jobs']}
    rows = read_rows(audio / 'items.jsonl')
    done = {r['id']: r for r in rows}
    if len(done) != len(rows) or any(r['status'] != 'ok_unreviewed' or sha(audio / r['audio']) != r['sha256'] for r in rows):
        raise ValueError('Previously rendered outputs changed')
    budget = (len(jobs) - len(done)) * 24000 * 30 * 2 + 16 * 1024**2
    if shutil.disk_usage(audio).free < 20 * 1024**3 + budget:
        raise RuntimeError('k2 destination does not preserve20GiB plus remaining worst-case audio budget')
    write_group(out, {'previous_failure.json': previous, 'inputs.json': {'selection_sha256': sha(plan / 'selection.json'),
        'previous_journal_sha256': sha(audio / 'items.jsonl'), 'previous_outputs': len(done), 'remaining_outputs': len(jobs) - len(done),
        'destination_free_bytes': shutil.disk_usage(audio).free, 'remaining_worst_case_bytes': budget,
        'receiver_sha256': sha(Path(__file__)), 'worker_sha256': sha(ROOT / 'scripts/stream_cross_system_teacher.py')}},
        'Explicit transport recovery after repeated L40 destination-reserve failures. Approved unchanged CUDA teacher reads existing pinned model/prompts; WAV encoding stays in memory and travels over SSH to k2, where the original20GiB destination guard remains. '
        'No reserve lowered and no storage deletion. previous_failure.json exact old failure; inputs.json hashes/budget; worker.log remote stderr; launch.json owned SSH process; runtime.json exact unchanged engine identity; parity.json verifies first old WAV byte-for-byte before any new output; report.json full verified result. '
        'No checkpoint/source replacement. Teacher audio remains under original plan with per-row transport provenance and streamed completion in run.json; original lock remains and is not misrepresented as covering the new transport code. Owner Ampixa/user; review2026-11-10; current consumer fresh cross-system listening. NOT on HF, inherited rights; logs disposable24h after consumer closes.',
        ['.venv-indicf5/bin/python -m scripts.receive_cross_system_teacher'])
    command = ['env', 'PYTHONDONTWRITEBYTECODE=1', '/home/ampixa/pocket_tts_stage/.venv/bin/python', '-u',
        '-m', 'scripts.stream_cross_system_teacher', '--plan', remote + '/plans/cross_system_sentences_v80', '--skip-json', json.dumps(sorted(done))]
    with (out / 'worker.log').open('xb') as log:
        child = subprocess.Popen(ssh + ['ampixa@khalti-gpu', 'cd ' + shlex.quote(remote) + ' && ' + shlex.join(command)],
            stdout=subprocess.PIPE, stderr=log, text=True)
        (out / 'launch.json').write_text(json.dumps({'pid': child.pid, 'command': child.args,
            'created_utc': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')
        parity, complete = False, False
        try:
            for line in child.stdout:
                result = json.loads(line)
                if result['type'] == 'ready':
                    if result['runtime'] != json.loads((audio / 'runtime.json').read_text()) or result['plan_sha256'] != sha(plan / 'selection.json'):
                        raise ValueError('Streamed engine/runtime differs from original approved run')
                    (out / 'runtime.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
                    continue
                if result['type'] == 'complete':
                    complete = True
                    continue
                if result['type'] not in {'wave', 'quality_rejection'}:
                    raise ValueError('Unrecognized transport message')
                if result['parity_only']:
                    if result['id'] != selection['jobs'][0]['id'] or result['sha256'] != done[result['id']]['sha256']:
                        raise ValueError('Approved teacher/seed/PCM parity check failed')
                    parity = True
                    (out / 'parity.json').write_text(json.dumps({k: v for k, v in result.items() if k != 'wave_b64'}, indent=2) + '\n')
                    continue
                if not parity or result['id'] in done:
                    raise ValueError('Parity missing or output already rendered')
                job = jobs[result['id']]
                row = {k: job[k] for k in ('id', 'text', 'text_id', 'text_kind', 'target_words', 'speaker_id', 'split', 'reference_sha256', 'source_reference_sha256')}
                row.update(system='pocket_teacher_250000_24l', clip_id=hashlib.sha256(('pocket_teacher_250000_24l:' + job['id']).encode()).hexdigest()[:24],
                    automatic_leakage_label=None, generation_transport='approved_teacher_memory_pipe_v80',
                    transport_provenance='reports/cross_system_teacher_stream_v80')
                if result['type'] == 'wave':
                    body = base64.b64decode(result['wave_b64'], validate=True)
                    if len(body) != result['bytes'] or hashlib.sha256(body).hexdigest() != result['sha256']:
                        raise ValueError('Transport waveform bytes changed')
                    if shutil.disk_usage(audio).free < 20 * 1024**3 + len(body):
                        raise RuntimeError('Preserve20GiB on actual k2 destination')
                    path = audio / 'audio' / (job['id'] + '.wav')
                    with path.open('xb') as stream:
                        stream.write(body)
                    row.update({k: result[k] for k in ('sha256', 'bytes', 'seconds', 'generation_detail', 'clipping_fraction_before_pcm', 'generation_wall_seconds')})
                    row.update(status='ok_unreviewed', audio=str(path.relative_to(audio)))
                else:
                    row.update(status='failed_waveform_quality', error=result['error'])
                with (audio / 'items.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'); stream.flush(); os.fsync(stream.fileno())
                done[row['id']] = row
            code = child.wait(timeout=30)
            if code or not complete or set(done) != set(jobs):
                raise RuntimeError('Teacher stream did not finish all actual jobs: ' + str(code))
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=30)
    final = {'status': 'complete_corrected_audio', 'pid': os.getpid(), 'system': 'pocket_teacher_250000_24l',
        'updated_utc': datetime.now(timezone.utc).isoformat(), 'planned_clips': len(jobs), 'processed_clips': len(done),
        'counts': dict(Counter(r['status'] for r in done.values())), 'transport_recovery': str(out.relative_to(ROOT)),
        'approved_teacher_unchanged': True, 'old_destination_guard_not_lowered': True}
    (audio / 'run.json').write_text(json.dumps(final, indent=2) + '\n')
    (out / 'report.json').write_text(json.dumps({**final, 'journal_sha256': sha(audio / 'items.jsonl'),
        'all_audio_hashes_verified': all(r['status'] != 'ok_unreviewed' or sha(audio / r['audio']) == r['sha256'] for r in done.values())}, indent=2) + '\n')
    with (audio / 'README.md').open('a') as stream:
        stream.write('\nRemaining generated audio transported from unchanged L40 teacher directly to k2 memory/SSH receiver; original20GiB output reserve remains checked on actual k2 destination. See reports/cross_system_teacher_stream_v80 for parity/runtime/source provenance. Native original partial L40 mirror is incomplete; full verified corpus is this k2 directory. Owner Ampixa/user; review2026-11-10.\n')
    print(json.dumps(final), flush=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Teacher memory transport failed explicitly; retain partial receipts and do not retry silently')
        raise
