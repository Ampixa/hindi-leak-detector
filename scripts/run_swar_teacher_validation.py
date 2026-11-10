"""Smoke and render best-teacher references without touching another workload."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shlex
import subprocess
import time
import traceback

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from scripts.extract_corrected_training import read_rows

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/home/ampixa/hindi-detector-teacher-f5-v14'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,default=ROOT/'data/swar_teacher_v21')
    parser.add_argument('--out',type=Path,default=ROOT/'reports/swar_teacher_validation_v21')
    parser.add_argument('--namespace',default='swar_teacher_v21')
    parser.add_argument('--smoke',type=int,default=6)
    parser.add_argument('--resume-unlaunched',action='store_true')
    options=parser.parse_args()
    plan=options.plan;out=options.out;namespace=options.namespace
    if not namespace.replace('_','').isalnum() or not namespace.isascii():raise ValueError('Unsafe owned namespace')
    selection=json.loads((plan/'selection.json').read_text());expected_clips=len(selection['jobs'])
    contexts=len({r['text_id'] for r in selection['jobs']})
    if not 1<=options.smoke<=expected_clips:raise ValueError('Invalid smoke count')
    if out.exists() and not options.resume_unlaunched:
        raise FileExistsError(out)
    if options.resume_unlaunched:
        previous=json.loads((out/'status.json').read_text())
        if previous['phase']!='failed_explicitly' or "'list' object has no attribute 'smoke'" not in previous.get('error','') or (out/'smoke_launch.json').exists():
            raise ValueError('Only inspected pre-launch failure may resume')
        if json.loads((out/'inputs.json').read_text())['plan_sha256']!=sha(plan/'selection.json'):
            raise ValueError('Unlaunched frozen plan changed')
        write_group(out/'launch_recovery',{'previous_failure.json':previous,'source_snapshot.json':{'sha256':sha(Path(__file__)),'source_utf8':Path(__file__).read_text()}},
            'Inspected CLI/local command variable shadowing prevented any synthesis launch. No model or waveform was generated; frozen input plan unchanged. Review2026-11-09, preserve compact recovery decision.',
            ['--resume-unlaunched; no previous smoke process handle exists'])
    if not options.resume_unlaunched:
        write_group(out, {'inputs.json': {'plan_sha256':sha(plan/'selection.json'),
            'model_pair_sha256':sha(plan/'model_pair.json'),'reference_inventory_sha256':sha(plan/'reference_inventory.json'),
            'reference_generation_uses_detector_weights':False,
            'purpose':'Generated-reference generalization only, not classifier fitting or pronunciation gold'},
            'source_snapshot.json': {'sha256':sha(Path(__file__)),'source_utf8':Path(__file__).read_text()}},
            'Owned best-teacher rendering controller. status.json records actual process/phase; '
            'smoke_launch.json/bulk_launch.json pin remote command/PID. report.json counts actual outputs '
            'for the exact frozen selection. Remote directory is an explicitly documented mirror, no model-weight copy. '
            'Content/phonetic reference suitability is a later separate audit, not assumed from finite WAVs. '
            'Review2026-11-09, current consumer automatic-reference detector validation; logs disposable24h after '
            'consumer closes. No HF upload/publication; source model/corpus licences apply.',
            ['.venv-indicf5/bin/python -m scripts.run_swar_teacher_validation',
             'Original correct24-layer teacher CUDA adapter, unchanged validated source and exact shared prompt bytes'])
    ssh = ['ssh','-4','-o','BatchMode=yes','-o','ConnectTimeout=8',
        '-o','UserKnownHostsFile='+str(ROOT/'runtime/l40_known_hosts_v14'),'-o','StrictHostKeyChecking=yes']
    remote_plan = REMOTE+'/plans/'+namespace
    remote_audio = REMOTE+'/data/'+namespace

    def command(value):
        return subprocess.check_output(ssh+['ampixa@khalti-gpu',value],text=True,timeout=60)

    def read_remote(relative):
        return json.loads(command('cat '+shlex.quote(REMOTE+'/'+relative)))

    def status(phase, **extra):
        p=out/'status.json';temp=p.with_suffix('.partial')
        temp.write_text(json.dumps({'phase':phase,'pid':os.getpid(),
            'updated_utc':datetime.now(timezone.utc).isoformat(),**extra},indent=2)+'\n');temp.replace(p)

    def sync():
        subprocess.run(['rsync','-a','--exclude=*.partial','-e',shlex.join(ssh),
            'ampixa@khalti-gpu:'+remote_audio+'/',str(plan/'teacher')+'/'],check=True,timeout=90)

    def launch(resume):
        args=['/home/ampixa/pocket_tts_stage/.venv/bin/python','-u','-m','scripts.run_corrected_synthesis',
              '--plan',remote_plan,'--out',remote_audio,'--system','pocket_teacher_250000_24l']
        args += ['--resume'] if resume else ['--limit',str(options.smoke)]
        name=namespace+('_bulk' if resume else '_smoke')
        code=('import pathlib,subprocess,json,datetime;root=pathlib.Path('+repr(REMOTE)+');cmd='+repr(args)+
              ';f=(root/'+repr(name+'.log')+').open("xb");p=subprocess.Popen(cmd,cwd=root,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
              'record={"pid":p.pid,"command":cmd,"created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()};'
              '(root/'+repr(name+'_launch.json')+').write_text(json.dumps(record,indent=2));'
              'print(json.dumps(record))')
        record=json.loads(command('python3 -c '+shlex.quote(code)))
        (out/('bulk_launch.json' if resume else 'smoke_launch.json')).write_text(json.dumps(record,indent=2)+'\n')
        return record

    def wait_for_run(handle, terminal):
        while True:
            exists=command('test -f '+shlex.quote(remote_audio+'/run.json')+' && echo yes || echo no').strip()
            state=read_remote('data/'+namespace+'/run.json') if exists=='yes' else None
            if state is not None and state['pid']==handle['pid']:
                if state['status']=='failed_explicitly':
                    raise RuntimeError('Actual requested teacher failed: '+state.get('error','unknown'))
                status('rendering_best_teacher_reference_audio',generation=state)
                if state['status']==terminal:
                    return state
            live=command('ps -p '+str(handle['pid'])+' -o args=')
            if 'scripts.run_corrected_synthesis' not in live or namespace not in live:
                raise RuntimeError('Owned teacher PID/command differs; no automatic restart')
            time.sleep(10)

    try:
        status('copying_frozen_reference_plan')
        command('mkdir -p '+shlex.quote(remote_plan))
        subprocess.run(['rsync','-a','-e',shlex.join(ssh),str(plan)+'/',
            'ampixa@khalti-gpu:'+remote_plan+'/'],check=True,timeout=90)
        handle=launch(False)
        # When smoke covers the entire small plan, the synthesis runner correctly
        # writes complete_corrected_audio rather than the partial-smoke status.
        smoke_terminal='complete_corrected_audio' if options.smoke==expected_clips else 'smoke_complete_ready_for_verified_resume'
        smoke_final=wait_for_run(handle,smoke_terminal)
        sync()
        runtime=json.loads((plan/'teacher/runtime.json').read_text())
        smoke=read_rows(plan/'teacher/items.jsonl')
        if runtime['loaded_layers']!=24 or runtime['weights_sha256']!='2b4636e3aed0e4f2d035630e504be95fc58f494b690a851a91bfe0dd8299516f':
            raise ValueError('Wrong teacher source/depth in actual reference smoke')
        if len(smoke)!=options.smoke or any(r['status']!='ok_unreviewed' for r in smoke):
            raise ValueError('Expected finite original-source smoke outputs did not complete')
        for row in smoke:
            if sha(plan/'teacher'/row['audio'])!=row['sha256']:
                raise ValueError('Copied teacher smoke bytes differ')
        (out/'smoke_report.json').write_text(json.dumps({'clips':options.smoke,'layers':24,
            'weights_sha256':runtime['weights_sha256'],'finite_waveforms':True,'pronunciation_gold':False},indent=2)+'\n')
        if options.smoke==expected_clips:
            final=smoke_final
        else:
            handle=launch(True)
            final=wait_for_run(handle,'complete_corrected_audio')
            sync()
        rows=read_rows(plan/'teacher/items.jsonl')
        if len(rows)!=expected_clips:
            raise ValueError('Completed teacher receipt count differs from frozen plan')
        report={'status':'complete_best_teacher_reference_rendering','generation':final,
            'contexts':contexts,'outputs':len(rows),'finite_outputs':sum(r['status']=='ok_unreviewed' for r in rows),
            'weights_sha256':runtime['weights_sha256'],'content_and_pronunciation_validated':False}
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        amendment='\nActual '+namespace+' run:'+str(expected_clips)+' best-teacher requests for'+str(contexts)+' word/text contexts, not the legacy768-text training plan mentioned in the generic runner caption. Current consumer reference-generalization audit, no fitting or error-gold inference. Mirror L40 '+remote_audio+' and k2 '+str(plan/'teacher')+'. Codex /root, owner Ampixa/user, review2026-11-09; remove redundant generated copies after consumer closes.\n'
        with (plan/'teacher/README.md').open('a') as f:f.write(amendment)
        code='import pathlib;p=pathlib.Path('+repr(remote_audio+'/README.md')+');p.write_text(p.read_text()+'+repr(amendment)+')'
        command('python3 -c '+shlex.quote(code))
        status('complete_best_teacher_reference_audio',outputs=len(rows))
    except Exception:
        status('failed_explicitly',error=traceback.format_exc());raise


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Best-teacher reference validation rendering failed explicitly');raise
