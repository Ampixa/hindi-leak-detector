"""Pinned content evidence for the generated same-text teacher references."""
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
from scripts.extract_corrected_training import read_rows

ROOT=Path(__file__).resolve().parents[1]


def main():
    plan=ROOT/'data/swar_teacher_v21';out=ROOT/'reports/swar_teacher_content_v21'
    if out.exists():raise FileExistsError(out)
    write_group(out,{'inputs.json':{'plan_sha256':sha(plan/'selection.json'),
        'purpose':'Lexical evidence only; does not certify individual vowel correctness'},
        'source_snapshot.json':{'sha256':sha(Path(__file__)),'source_utf8':Path(__file__).read_text()}},
        'Separate-runtime content screen of172 teacher references. items.jsonl preserves source rows, all '
        'pinned ASR hypotheses/errors and content decision; runtime.json exact gate, report/status counts. '
        'Global ASR quarantine can be caused by code-switching and does not alone decide a focal vowel; '
        'word phonetic suitability is independently measured. Rejected references are not used for strong '
        'consensus. Current consumer automatic native-reference evaluation; review2026-11-09, retain compact '
        'metrics/pins, no waveform/model copy/HF upload; inherited licences.',
        ['.venv-asr-gate/bin/python -m scripts.screen_swar_teacher_refs'])

    def status(phase,**extra):
        p=out/'status.json';t=p.with_suffix('.partial');t.write_text(json.dumps({'phase':phase,'pid':os.getpid(),
            'updated_utc':datetime.now(timezone.utc).isoformat(),**extra},indent=2)+'\n');t.replace(p)

    try:
        status('waiting_for_complete_best_teacher_audio')
        while True:
            previous=json.loads((ROOT/'reports/swar_teacher_validation_v21/status.json').read_text())
            if previous['phase']=='failed_explicitly':raise RuntimeError('Best-teacher generation failed')
            if previous['phase']=='complete_best_teacher_reference_audio':break
            os.kill(previous['pid'],0);time.sleep(5)
        runtime=json.loads((plan/'teacher/runtime.json').read_text())
        if runtime['loaded_layers']!=24 or runtime['weights_sha256']!='2b4636e3aed0e4f2d035630e504be95fc58f494b690a851a91bfe0dd8299516f':raise ValueError('Wrong reference teacher')
        gate=CpuContentGate(ROOT,threads=2);(out/'runtime.json').write_text(json.dumps(gate.runtime,indent=2)+'\n')
        counts=Counter();rows=read_rows(plan/'teacher/items.jsonl')
        for index,row in enumerate(rows):
            if row['status']!='ok_unreviewed':continue
            result=gate.score(audio(plan/'teacher',row),row['text'],row['system'])
            record={**row,'content_quality':result,'source_pack':str(plan/'teacher')}
            with (out/'items.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
            counts[result['decision']]+=1;status('screening_generated_references',completed=index+1,total=len(rows),decisions=dict(counts))
        (out/'report.json').write_text(json.dumps({'status':'complete_reference_content_evidence','decisions':dict(counts),'planned':len(rows),'actual_pronunciation_gold':None},indent=2)+'\n')
        status('complete_teacher_reference_content',decisions=dict(counts))
    except Exception:
        status('failed_explicitly',error=traceback.format_exc());raise


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Teacher reference content screen failed explicitly');raise
