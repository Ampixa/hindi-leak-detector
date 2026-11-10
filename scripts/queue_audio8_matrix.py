"""Owned one-shot worker waits for spare GPU compute before matrix synthesis."""
from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/home/ampixa/hindi-detector-wordmatrix-v3')
LOG = logging.getLogger(__name__)


def save(state: dict) -> None:
    state.update(pid=os.getpid(), updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    path = ROOT / 'queue_status.json'
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(state, indent=2) + '\n'); temporary.replace(path)


def main() -> None:
    os.nice(15)
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    if manifest['experiment_id'] != 'wordmatrix_v3':
        raise ValueError('Unexpected GPU experiment')
    idle = 0
    while idle < 3:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        if len(output.strip().splitlines()) != 1:
            raise ValueError('Expected the inspected single-GPU host')
        free_mb, utilization = [int(v.strip()) for v in output.strip().split(',')]
        disk_free = shutil.disk_usage(ROOT).free
        eligible = free_mb >= 14 * 1024 and utilization <= 30 and disk_free >= 18 * 1024**3
        idle = idle + 1 if eligible else 0
        state = {'status': 'waiting_for_spare_gpu', 'free_gpu_mib': free_mb, 'gpu_utilization_percent': utilization,
                 'free_disk_bytes': disk_free, 'idle_samples': idle, 'required_samples': 3,
                 'policy': '15-second observations; wait for low utilization, sufficient memory/disk; never stop another job'}
        save(state); LOG.info('GPU queue %s', json.dumps(state))
        if idle < 3:
            time.sleep(15)
    save({'status': 'synthesizing', 'policy': 'Native worker additionally enforces its existing memory/disk/RAM caps'})
    with (ROOT / 'outputs/synthesis.log').open('a') as log:
        subprocess.run([sys.executable, '-u', str(ROOT / 'script.py'), '--manifest', str(ROOT / 'manifest.json'),
                        '--out', str(ROOT / 'outputs'), '--model-dir', str(ROOT / 'models/audio8')],
                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
    save({'status': 'synthesis_completed'})


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Owned Audio8 matrix worker failed; inspect receipts before resuming')
        save({'status': 'failed_inspect_log_and_partial_receipts'})
        raise
