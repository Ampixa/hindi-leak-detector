"""Render approved teacher to a pipe; generated WAVs never touch L40 disk."""
from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf

from hindi_detector.corrected_synthesis import QualityError, TeacherEngine, require_pair, sha, validate_wave


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--skip-json', required=True)
    args = parser.parse_args()
    pair = json.loads((args.plan / 'model_pair.json').read_text())
    require_pair(pair)
    selection = json.loads((args.plan / 'selection.json').read_text())
    skip = json.loads(args.skip_json)
    jobs = selection['jobs']
    if selection['role'] != 'fresh_cross_system_discovery_v80_no_fitting' or not set(skip) <= {j['id'] for j in jobs}:
        raise ValueError('Unapproved streaming plan or skip identities')
    for job in jobs:
        path = (args.plan / job['reference_audio']).resolve(strict=True)
        if not path.is_relative_to((args.plan / 'references').resolve()) or sha(path) != job['reference_sha256']:
            raise ValueError('Streamed teacher conditioning changed')
        job['reference_audio'] = str(path)
    with contextlib.redirect_stdout(sys.stderr):
        engine = TeacherEngine(pair['systems']['pocket_teacher_250000_24l'])
    print(json.dumps({'type': 'ready', 'pid': os.getpid(), 'runtime': engine.runtime,
        'plan_sha256': sha(args.plan / 'selection.json'), 'source_sha256': sha(Path(__file__))}), flush=True)
    # The first old job is an explicit CUDA/seed/PCM parity check. The controller
    # discards that transport copy after checking its already-retained SHA.
    selected = [(jobs[0], True)] + [(j, False) for j in jobs if j['id'] not in skip]
    for job, parity in selected:
        start = time.monotonic()
        try:
            with contextlib.redirect_stdout(sys.stderr):
                wave, rate, detail = engine.generate(job)
            validate_wave(wave, rate)
            buffer = io.BytesIO()
            sf.write(buffer, wave, rate, format='WAV', subtype='PCM_16')
            body = buffer.getvalue()
            result = {'type': 'wave', 'id': job['id'], 'parity_only': parity, 'sha256': hashlib.sha256(body).hexdigest(),
                'bytes': len(body), 'seconds': len(wave) / rate, 'generation_detail': detail,
                'clipping_fraction_before_pcm': float(np.mean(np.abs(wave) >= 1)),
                'generation_wall_seconds': time.monotonic() - start, 'wave_b64': base64.b64encode(body).decode()}
        except QualityError as error:
            if parity:
                raise RuntimeError('Parity waveform rejected') from error
            result = {'type': 'quality_rejection', 'id': job['id'], 'parity_only': False, 'error': str(error)}
        print(json.dumps(result, allow_nan=False), flush=True)
    print(json.dumps({'type': 'complete', 'new_jobs': sum(j['id'] not in skip for j in jobs)}), flush=True)


if __name__ == '__main__':
    main()
