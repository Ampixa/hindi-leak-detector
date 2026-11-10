"""Owned CPU continuation: completed Pocket matrix -> word evidence -> fitting."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'reports/pocket_pipeline_v4'
LOG = logging.getLogger(__name__)


def save(value):
    value.update(updated_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid())
    path = STATE / 'status.json'; temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n'); temporary.replace(path)


def main():
    os.nice(10)
    if not (STATE / 'README.md').is_file() or (STATE / 'source_snapshot.json').exists():
        raise ValueError('Need a documented fresh continuation; inspect an existing run before retry')
    source_names = ['scripts/continue_pocket_pipeline.py', 'scripts/run_pocket_matrix.py', 'scripts/extract_word_evidence.py',
                    'scripts/train_crossed_words.py', 'hindi_detector/word_evidence.py', 'hindi_detector/local_features.py',
                    'hindi_detector/word_model.py', 'hindi_detector/reference_words.py', 'hindi_detector/crossed_words.py']
    snapshots = []
    for name in source_names:
        data = (ROOT / name).read_bytes()
        snapshots.append({'path': name, 'sha256': hashlib.sha256(data).hexdigest(), 'source_utf8': data.decode()})
    (STATE / 'source_snapshot.json').write_text(json.dumps(snapshots, indent=2) + '\n')

    def unchanged():
        for member in snapshots:
            if hashlib.sha256((ROOT / member['path']).read_bytes()).hexdigest() != member['sha256']:
                raise ValueError('Pipeline source changed; inspect before starting the next phase: ' + member['path'])

    pocket = ROOT / 'data/pocket_wordmatrix_v4/pocket'
    while True:
        unchanged()
        run = json.loads((pocket / 'run.json').read_text())
        if run['status'] == 'complete_unreviewed_audio':
            if run['completed'] != 1024:
                raise ValueError('Pocket run declared complete with missing outputs')
            break
        if run['status'] != 'running':
            raise RuntimeError('Pocket generation is not healthy: ' + run['status'])
        pid = int((pocket / 'pid').read_text())
        live = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True)
        if live.returncode or 'scripts.run_pocket_matrix' not in live.stdout:
            raise RuntimeError('Pocket generation worker disappeared; inspect partial receipts')
        save({'status': 'waiting_for_pocket_generation', 'generated_clips': run['completed'], 'planned_clips': 1024})
        time.sleep(15)
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2'}
    python = str(ROOT / '.venv-indicf5/bin/python')
    phases = [
        ('extracting_word_evidence', 'extract.log', ['-m', 'scripts.extract_word_evidence', '--pack', 'data/pocket_wordmatrix_v4', '--out', 'reports/wordlab_pocket_matrix_v4']),
        ('training_crossed_reference_heads', 'fit.log', ['-m', 'scripts.train_crossed_words', '--f5-evidence', 'reports/wordlab_matrix_f5_v3', '--pocket-evidence', 'reports/wordlab_pocket_matrix_v4', '--plan', 'data/pocket_wordmatrix_v4/plan.json', '--out', 'reports/pocket_crossed_models_v4'])]
    for status, filename, command in phases:
        unchanged(); save({'status': status, 'generated_clips': 1024})
        with (STATE / filename).open('w') as log:
            subprocess.run([python, *command], cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
    save({'status': 'candidate_trained_reliability_unvalidated', 'generated_clips': 1024,
          'report': 'reports/pocket_crossed_models_v4/report.json',
          'reliable_automatic_detector': False,
          'remaining': 'Independent native false-alarm and actual word/prosody-error validation; source role accuracy is insufficient.'})
    LOG.info('Pocket reference candidate trained; reliability remains unvalidated')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Pocket continuation failed; inspect status/logs and exact source snapshot before retry')
        save({'status': 'failed_inspect_logs_before_retry'})
        raise
