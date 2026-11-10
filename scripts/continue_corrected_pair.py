"""Qualify the actual requested pair, resume locked bulk jobs, and screen content."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shlex
import subprocess
import time
import traceback

from hindi_detector.asr_gate import CpuContentGate, RECIPE
from hindi_detector.corrected_synthesis import SYSTEMS, require_pair, sha
from scripts.calibrate_asr_gate import audio

ROOT=Path(__file__).resolve().parents[1]
LOG=logging.getLogger(__name__)
REMOTE='/home/ampixa/hindi-detector-teacher-f5-v14'


def complete_rows(path):
    return [json.loads(line) for line in path.read_bytes().split(b'\n')[:-1]] if path.is_file() else []


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if not (args.out/'README.md').is_file() or (args.out/'status.json').exists():raise ValueError('Need documented fresh controller output')
    pair=json.loads((args.plan/'model_pair.json').read_text());require_pair(pair)
    ssh=['ssh','-4','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','UserKnownHostsFile='+str(ROOT/'runtime/l40_known_hosts_v14'),'-o','StrictHostKeyChecking=yes']
    def remote(command):
        return subprocess.check_output(ssh+['ampixa@khalti-gpu',command],text=True,timeout=60)
    def remote_json(relative):
        return json.loads(remote('cat '+shlex.quote(REMOTE+'/'+relative)))
    def status(phase,**extra):
        p=args.out/'status.json';temp=p.with_suffix('.partial')
        temp.write_text(json.dumps({'phase':phase,'pid':os.getpid(),'updated_utc':datetime.now(timezone.utc).isoformat(),
            'systems':list(SYSTEMS),**extra},ensure_ascii=False,indent=2)+'\n');temp.replace(p)
    paths={SYSTEMS[0]:args.plan/'teacher',SYSTEMS[1]:args.plan/'indicf5'}
    seen=set();counts=defaultdict(Counter)
    def sync_teacher():
        subprocess.run(['rsync','-a','--exclude=*.partial','-e',shlex.join(ssh),
            'ampixa@khalti-gpu:'+REMOTE+'/data/teacher/',str(paths[SYSTEMS[0]])+'/'],check=True,timeout=90)
    def verify_live(state,system):
        pid=str(state['pid'])
        command=remote('ps -p '+pid+' -o args=') if system==SYSTEMS[0] else subprocess.check_output(['ps','-p',pid,'-o','command='],text=True)
        if 'scripts.run_corrected_synthesis' not in command or system not in command:raise RuntimeError('Synthesis PID/command no longer matches owned job')
    def score_available(gate):
        for system in SYSTEMS:
            directory=paths[system]
            for row in complete_rows(directory/'items.jsonl'):
                if row['clip_id'] in seen:continue
                if row['system']!=system:raise ValueError('Wrong source in corrected audio registry')
                if row['status']=='failed_waveform_quality':
                    seen.add(row['clip_id']);counts[system]['waveform_reject']+=1;continue
                if row['status']!='ok_unreviewed':raise ValueError('Unknown synthesis receipt status')
                result=gate.score(audio(directory,row),row['text'],system)
                with (args.out/'content_items.jsonl').open('a') as handle:
                    handle.write(json.dumps({**row,'source_pack':str(directory.resolve()),'content_quality':result,
                        'status':'content_scored_not_pronunciation_gold'},ensure_ascii=False,allow_nan=False)+'\n');handle.flush()
                seen.add(row['clip_id']);counts[system][result['decision']]+=1
        return {s:dict(c) for s,c in counts.items()}
    try:
        (args.out/'controller_source_snapshot.json').write_text(json.dumps([{'path':str(p.relative_to(ROOT)),
            'sha256':sha(p),'source_utf8':p.read_text()} for p in (Path(__file__),ROOT/'hindi_detector/asr_gate.py',ROOT/'hindi_detector/content_quality.py')],indent=2)+'\n')
        status('waiting_for_explicit_model_smoke')
        while True:
            f5=json.loads((paths[SYSTEMS[1]]/'run.json').read_text());teacher=remote_json('data/teacher/run.json')
            states={SYSTEMS[0]:teacher,SYSTEMS[1]:f5}
            if any(s['status']=='failed_explicitly' for s in states.values()):raise RuntimeError('Corrected synthesis smoke failed; inspect actual model log')
            if all(s['status']=='smoke_complete_ready_for_verified_resume' for s in states.values()):break
            for system,state in states.items():
                if state['status']!='smoke_complete_ready_for_verified_resume':verify_live(state,system)
            status('waiting_for_explicit_model_smoke',generation=states);time.sleep(10)
        sync_teacher()
        for system in SYSTEMS:
            directory=paths[system];runtime=json.loads((directory/'runtime.json').read_text());rows=complete_rows(directory/'items.jsonl')
            if runtime['system']!=system or runtime['weights_sha256']!=pair['systems'][system]['sha256'] or len(rows)!=8:
                raise ValueError('Actual smoke model identity or clip count differs')
            if any(r['split']!='train' or r['status']!='ok_unreviewed' for r in rows):raise ValueError('Training-only8-clip finite-audio smoke did not pass')
            if system==SYSTEMS[0] and runtime['loaded_layers']!=24:raise ValueError('Teacher smoke loaded wrong depth')
        status('screening_actual_corrected_smoke')
        gate=CpuContentGate(ROOT,threads=2)
        (args.out/'asr_runtime.json').write_text(json.dumps(gate.runtime,indent=2)+'\n')
        measured=score_available(gate)
        (args.out/'smoke_report.json').write_text(json.dumps({'status':'actual_model_pair_smoke_complete','clip_counts':{s:8 for s in SYSTEMS},
            'content_decisions':measured,'recipe':RECIPE,'automatic_hindi_accuracy':None,
            'model_pair_sha256':sha(args.plan/'model_pair.json')},indent=2)+'\n')
        if any(counts[s]['pass']<2 for s in SYSTEMS):raise RuntimeError('Fewer than2/8 content-passing smoke clips in a source; bulk qualification stopped')
        for system in SYSTEMS:
            if system==SYSTEMS[1]:
                initial=json.loads((args.out/'indicf5_smoke_launch.json').read_text());cmd=initial['command'][:-2]+['--resume']
                with (args.out/'indicf5_bulk.log').open('xb') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                handle={'pid':p.pid,'command':cmd,'host':'k2'}
            else:
                initial=remote_json('teacher_smoke_launch.json');cmd=initial['command'][:-2]+['--resume']
                code='import pathlib,subprocess,json,datetime;root=pathlib.Path('+repr(REMOTE)+');cmd='+repr(cmd)+";f=(root/'teacher_bulk.log').open('xb');p=subprocess.Popen(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);(root/'teacher_bulk_launch.json').write_text(json.dumps({'pid':p.pid,'command':cmd},indent=2));print(p.pid)"
                pid=int(remote('python3 -c '+shlex.quote(code)).strip());handle={'pid':pid,'command':cmd,'host':'khalti-gpu'}
            (args.out/(system+'_bulk_launch.json')).write_text(json.dumps(handle,indent=2)+'\n')
            for _ in range(60):
                started=remote_json('data/teacher/run.json') if system==SYSTEMS[0] else json.loads((paths[system]/'run.json').read_text())
                if started['pid']==handle['pid']:break
                verify_live({'pid':handle['pid']},system);time.sleep(1)
            else:raise TimeoutError('Resumed worker did not publish its own PID; no old smoke status accepted')
        while True:
            sync_teacher();states={s:json.loads((p/'run.json').read_text()) for s,p in paths.items()}
            measured=score_available(gate)
            if any(s['status']=='failed_explicitly' for s in states.values()):raise RuntimeError('Corrected bulk worker failed; explicit partial outputs retained')
            done=all(s['status']=='complete_corrected_audio' for s in states.values())
            status('complete_corrected_generation_and_content_checks' if done else 'generating_and_screening_corrected_pair',generation=states,content_decisions=measured)
            if done:
                (args.out/'report.json').write_text(json.dumps({'generation':states,'content_decisions':measured,
                    'planned_clips':3072,'actual_hindi_accuracy':None,'new_detector_trained':False,
                    'note':'Correct TTS sources and content screen complete. No generator-identity labels presented as actual pronunciation errors.'},indent=2)+'\n');break
            for system,state in states.items():
                if state['status']!='complete_corrected_audio':verify_live(state,system)
            time.sleep(20)
    except Exception:
        status('failed_explicitly',error=traceback.format_exc(),content_decisions={s:dict(c) for s,c in counts.items()});raise


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        LOG.exception('Corrected-pair controller stopped explicitly; no blind retry');raise
