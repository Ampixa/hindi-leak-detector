"""Persistent isolated-runtime content gate for the owned feature worker."""
from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

from hindi_detector.asr_gate import CpuContentGate
from scripts.calibrate_asr_gate import audio


def main():
    root=Path(__file__).resolve().parents[1];gate=CpuContentGate(root,threads=2)
    print(json.dumps({'status':'ready','runtime':gate.runtime}),flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        pack=Path(request['pack']).resolve(strict=True)
        if not pack.is_relative_to(root/'reports/structured_evidence_v15'):
            raise ValueError('RPC accepts only the owned corrected physical-control group')
        result=gate.score(audio(pack,request['row']),request['row']['text'],request['row']['system'])
        print(json.dumps({'id':request['row']['clip_id'],'content_quality':result},ensure_ascii=False,allow_nan=False),flush=True)


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Owned content RPC failed explicitly');raise
