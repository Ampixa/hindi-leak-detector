"""Owned serial rhythm experiment; never touches the live detector service."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import subprocess
import traceback

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recovery-v52', action='store_true')
    args = parser.parse_args()
    out = ROOT / ('reports/rhythm_launch_v52' if args.recovery_v52 else 'reports/rhythm_launch_v51')
    if out.exists():
        raise FileExistsError(out)
    write_group(out, {'inputs.json': {'driver_sha256': sha(Path(__file__)),
        'builder_sha256': sha(ROOT / 'scripts/build_rhythm_interventions.py'),
        'trainer_sha256': sha(ROOT / 'scripts/train_prosody_transfer.py')}},
        'Owned serial rhythm process receipts and logs. status.json actual PID/phase/failure; '
        'build_launch.json/build.log physical-control child, train_launch.json/train.log frozen '
        'candidate fitting/evaluation child. No source/render/service PID is terminated. '
        'Consumer causal word-rhythm admission test; review2026-11-10; logs disposable24h '
        'after closure; no model/audio copies/HF upload.',
        ['.venv-indicf5/bin/python -u -m scripts.run_rhythm_experiment'])

    def status(phase, **extra):
        temp = out / 'status.partial'
        temp.write_text(json.dumps({'phase': phase, 'pid': os.getpid(),
            'updated_utc': datetime.now(timezone.utc).isoformat(), **extra}, indent=2) + '\n')
        temp.replace(out / 'status.json')

    try:
        for name, module, arguments in [('build', 'scripts.build_rhythm_interventions', ['--recovery-v52'] if args.recovery_v52 else []),
                                         ('train', 'scripts.train_prosody_transfer', ['--rhythm-v52'] if args.recovery_v52 else ['--rhythm'])]:
            command = [str(ROOT / '.venv-indicf5/bin/python'), '-u', '-m', module, *arguments]
            with (out / (name + '.log')).open('xb') as log:
                process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                receipt = {'pid': process.pid, 'command': command, 'created_utc': datetime.now(timezone.utc).isoformat()}
                (out / (name + '_launch.json')).write_text(json.dumps(receipt, indent=2) + '\n')
                status(name, child=receipt)
                code = process.wait()
                if code != 0:
                    raise RuntimeError(name + ' child failed explicitly with exit' + str(code))
        status('complete_rhythm_experiment')
    except Exception:
        status('failed_explicitly', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Owned rhythm experiment stopped explicitly; no automatic rerun or threshold tuning')
        raise
