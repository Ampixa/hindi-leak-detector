"""One explicit engine per worker, immutable input/model lock, durable receipts."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import soundfile as sf

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import SYSTEMS, IndicF5Engine, TeacherEngine, QualityError, require_pair, sha, validate_wave

LOG=logging.getLogger(__name__)


def save(path,value):
    temp=path.with_suffix('.partial');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--system',choices=SYSTEMS,required=True);parser.add_argument('--limit',type=int)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    pair=json.loads((args.plan/'model_pair.json').read_text());require_pair(pair)
    selection=json.loads((args.plan/'selection.json').read_text())
    if selection['experiment_id']!='teacher_indicf5_v14':raise ValueError('Wrong corrected text plan')
    jobs=[]
    for raw in selection['jobs']:
        reference=(args.plan/raw['reference_audio']).resolve(strict=True)
        if not reference.is_relative_to(args.plan.resolve()/'references'):
            raise ValueError('Shared reference path outside the corrected plan')
        jobs.append({**raw,'reference_audio':str(reference)})
    limit=len(jobs) if args.limit is None else args.limit
    if not 1<=limit<=len(jobs) or len({r['id'] for r in jobs})!=len(jobs):raise ValueError('Bad synthesis job identities/limit')
    for job in jobs:
        if not job['id'].replace('_','').isalnum() or job['seed']!=0:raise ValueError('Unexpected job namespace/seed')
    source=Path(__file__);module=source.parents[1]/'hindi_detector/corrected_synthesis.py'
    lock={'selection_sha256':sha(args.plan/'selection.json'),'model_pair_sha256':sha(args.plan/'model_pair.json'),
        'system':args.system,'driver_sha256':sha(source),'engine_sha256':sha(module)}
    if args.resume:
        if json.loads((args.out/'lock.json').read_text())!=lock:raise ValueError('Resume code/model/jobs differ')
    else:
        if args.out.exists():raise FileExistsError(args.out)
        write_group(args.out,{'lock.json':lock,'source_snapshot.json':[{'path':str(p),'sha256':sha(p),'source_utf8':p.read_text()} for p in (source,module)]},
            'Requested corrected synthesis engine. runtime.json exact model/inference source, run.json live PID/counts/failures; items.jsonl every WAV/source/text/reference SHA/bytes/duration/timing or explicit quality rejection; audio/*.wav generated24kPCM16; selection.json publishes registry. No old Pocket models or source-as-error labels. Keep generated audio for current corrected comparison; review by2026-11-09, delete within24h after consumer finishes unless documented current use.',
            ['Native explicit model adapter, same768 frozen texts/two shared reference groups per text','python -m scripts.run_corrected_synthesis --plan PLAN --out NEW --system '+args.system+' [--limit8]; resume only unchanged lock'])
        (args.out/'audio').mkdir()
    receipts=args.out/'items.jsonl';records=[json.loads(s) for s in receipts.read_text().splitlines()] if receipts.exists() else []
    existing={r['id']:r for r in records}
    if len(existing)!=len(records):raise ValueError('Duplicate corrected receipts')
    lookup={j['id']:j for j in jobs}
    for identifier,row in existing.items():
        job=lookup[identifier]
        if row['system']!=args.system or any(row[k]!=job[k] for k in ('text','split','reference_sha256','speaker_id')):raise ValueError('Receipt differs from plan')
        if row['status']=='ok_unreviewed' and sha(args.out/row['audio'])!=row['sha256']:raise ValueError('Retained corrected audio changed')
    def status(phase,**extra):
        save(args.out/'run.json',{'status':phase,'pid':os.getpid(),'system':args.system,'updated_utc':datetime.now(timezone.utc).isoformat(),
            'planned_clips':len(jobs),'processed_clips':len(existing),'counts':dict(Counter(r['status'] for r in existing.values())),**extra})
    try:
        references={j['reference_audio']:j['reference_sha256'] for j in jobs}
        for p,d in references.items():
            if sha(Path(p))!=d:raise ValueError('Shared reference prompt changed')
        status('loading_explicit_verified_model')
        engine=(TeacherEngine if args.system==SYSTEMS[0] else IndicF5Engine)(pair['systems'][args.system])
        save(args.out/'runtime.json',engine.runtime)
        for job in jobs[:limit]:
            if job['id'] in existing:continue
            if shutil.disk_usage(args.out).free<20*1024**3:raise RuntimeError('Preserve20GiB free; no cleanup of another workload')
            status('generating_corrected_pair',current_job=job['id']);start=time.monotonic()
            row={k:job[k] for k in ('id','text','text_id','text_kind','target_words','speaker_id','split','reference_sha256','source_reference_sha256')}
            row.update(system=args.system,clip_id=hashlib.sha256((args.system+':'+job['id']).encode()).hexdigest()[:24],automatic_leakage_label=None)
            try:
                wave,rate,detail=engine.generate(job);validate_wave(wave,rate)
                path=args.out/'audio'/(job['id']+'.wav')
                if path.exists():raise FileExistsError('Unreceipted corrected output requires inspection')
                sf.write(path,wave,rate,subtype='PCM_16')
                row.update(status='ok_unreviewed',audio=str(path.relative_to(args.out)),sha256=sha(path),bytes=path.stat().st_size,
                    seconds=len(wave)/rate,clipping_fraction_before_pcm=float(np.mean(np.abs(wave)>=1)),generation_detail=detail)
            except QualityError:
                LOG.exception('Explicit corrected waveform-quality rejection %s',job['id']);row.update(status='failed_waveform_quality',error=traceback.format_exc())
            row['generation_wall_seconds']=time.monotonic()-start
            with receipts.open('a') as handle:
                handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno())
            existing[job['id']]=row
            status('generating_corrected_pair');LOG.info('%s %s/%s',args.system,len(existing),len(jobs))
        clips=[{**r,'sentence_id':r['id']} for r in existing.values() if r['status']=='ok_unreviewed']
        save(args.out/'selection.json',{'experiment_id':'teacher_indicf5_v14','jobs':jobs,'clips':clips})
        status('complete_corrected_audio' if len(existing)==len(jobs) else 'smoke_complete_ready_for_verified_resume')
    except Exception:
        status('failed_explicitly',error=traceback.format_exc());raise


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        LOG.exception('Corrected synthesis stopped; no silent fallback or restart');raise
