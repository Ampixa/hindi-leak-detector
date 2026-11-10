"""Stream verified24-layer renderings into one/two frozen acoustic frame caches."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shlex
import subprocess
import time
import traceback

import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForCTC

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.word_evidence import WordEvidenceEngine
from scripts.calibrate_asr_gate import audio
from scripts.extract_corrected_training import read_rows

ROOT=Path(__file__).resolve().parents[1]
REMOTE='/home/ampixa/hindi-detector-teacher-f5-v14'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--controller',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--namespace',required=True)
    parser.add_argument('--secondary',action='store_true')
    parser.add_argument('--resume-copy-race',action='store_true',help='Resume an inspected missing-file rsync race; verify every committed receipt')
    args=parser.parse_args()
    if args.out.exists() and not args.resume_copy_race:raise FileExistsError(args.out)
    plan=args.plan.resolve(strict=True)
    if not plan.is_relative_to(ROOT/'data') or not args.namespace.replace('_','').isalnum() or not args.namespace.isascii():
        raise ValueError('Unsafe owned plan/namespace')
    selection=json.loads((plan/'selection.json').read_text());jobs={r['id']:r for r in selection['jobs']}
    if args.resume_copy_race:
        previous=json.loads((args.out/'status.json').read_text())
        if previous['phase']!='failed_explicitly' or 'FileNotFoundError' not in previous.get('error',''):
            raise ValueError('Resume is restricted to the inspected pre-copy missing-file failure')
        inputs=json.loads((args.out/'inputs.json').read_text())
        if inputs['plan_sha256']!=sha(plan/'selection.json') or inputs['secondary_model']!=args.secondary or inputs['namespace']!=args.namespace:
            raise ValueError('Resumed extraction inputs changed')
        recovery=args.out/'copy_race_recovery.json'
        if recovery.exists():raise FileExistsError(recovery)
        recovery.write_text(json.dumps({'previous_status':previous,'fixed_source_sha256':sha(Path(__file__)),
            'reason':'rsync file listing can predate the copied manifest; defer missing active-source audio until next sync'},indent=2)+'\n')
        with (args.out/'README.md').open('a') as f:f.write('\ncopy_race_recovery.json preserves the inspected failure and fixed source hash; committed evidence is verified and retained.\n')
    else:
      write_group(args.out,{'inputs.json':{'plan_sha256':sha(plan/'selection.json'),'controller':str(args.controller),
        'namespace':args.namespace,'requested_jobs':len(jobs),'secondary_model':args.secondary,
        'policy':'Generated classes are requests, not observed phonetic gold. Canonical analysis text is explicit and distinct from edited synthesis text.'},
        'source_snapshot.json':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'source_utf8':p.read_text()}
            for p in (Path(__file__),ROOT/'hindi_detector/word_evidence.py')]},
        'Streaming rendered speech evidence. items.jsonl pins actual source waveform and per-clip detail/NPZ '
        'SHA or explicit failure. details/*.json preserve canonical analysis text, original generation text, '
        'word times and measured phone/prosody; features/*.npz contain primary phone_log_probs and optional '
        'secondary_phone_log_probs. runtime.json pins models/vocab/actual source runtime; no hidden voice '
        'embeddings or source-as-error labels. source audio remains in documented plan/teacher mirror. '
        'Current consumer phonetic reference/minimal-pair evaluation, review2026-11-09; remove redundant '
        'raw frames after consumer closes, retain model/metrics/input pins. No HF upload; inherited licences '
        'including shared MMS research restriction.',
        ['.venv-indicf5/bin/python -m scripts.extract_rendered_vowels --plan PLAN --controller CONTROLLER --namespace NAME --out NEW [--secondary]'])
    for name in ('details','features'):(args.out/name).mkdir(exist_ok=args.resume_copy_race)
    ssh=['ssh','-4','-o','BatchMode=yes','-o','ConnectTimeout=8',
         '-o','UserKnownHostsFile='+str(ROOT/'runtime/l40_known_hosts_v14'),'-o','StrictHostKeyChecking=yes']

    def sync():
        subprocess.run(['rsync','-a','--exclude=*.partial','-e',shlex.join(ssh),
            'ampixa@khalti-gpu:'+REMOTE+'/data/'+args.namespace+'/',str(plan/'teacher')+'/'],check=True,timeout=90)

    def status(phase,**extra):
        p=args.out/'status.json';temp=p.with_suffix('.partial');temp.write_text(json.dumps({'phase':phase,'pid':os.getpid(),
            'updated_utc':datetime.now(timezone.utc).isoformat(),**extra},indent=2)+'\n');temp.replace(p)

    try:
        status('waiting_for_first_verified_rendering')
        while True:
            controller=json.loads((args.controller/'status.json').read_text())
            if controller['phase']=='failed_explicitly':raise RuntimeError('Owned renderer failed before feature extraction')
            if 'generation' in controller or controller['phase']=='complete_best_teacher_reference_audio':
                sync()
                if (plan/'teacher/runtime.json').is_file():break
            os.kill(controller['pid'],0);time.sleep(5)
        synthesis=json.loads((plan/'teacher/runtime.json').read_text())
        if synthesis['loaded_layers']!=24 or synthesis['weights_sha256']!='2b4636e3aed0e4f2d035630e504be95fc58f494b690a851a91bfe0dd8299516f':
            raise ValueError('Actual requested renderer source/depth differs')
        engine=WordEvidenceEngine(ROOT,device='mps',pitch_method='dio',temporal_bins=20,retain_frame_evidence=True)
        secondary=None;extractor=None;second_inventory=None;second_vocab=None
        if args.secondary:
            directory=ROOT/'models/phoneme_probe_lv60_v20'
            second_inventory=json.loads((directory/'inventory.json').read_text())
            for member in second_inventory['files']:
                if sha(directory/member['path'])!=member['sha256']:raise ValueError('Secondary model changed')
            second_vocab=json.loads((directory/'vocab.json').read_text())
            extractor=AutoFeatureExtractor.from_pretrained(directory,local_files_only=True)
            secondary=AutoModelForCTC.from_pretrained(directory,local_files_only=True).eval().to('mps')
            if secondary.config.pad_token_id!=0 or extractor.sampling_rate!=16000:raise ValueError('Secondary CTC contract differs')
        runtime={'primary_phone':engine.phone_inventory,'secondary_phone':second_inventory,
            'vocabulary':engine.vocab,'secondary_vocabulary':second_vocab,'alignment':engine.aligner_inventory,
            'device':'mps','source_synthesis_runtime_sha256':sha(plan/'teacher/runtime.json'),
            'word_evidence_source_sha256':sha(ROOT/'hindi_detector/word_evidence.py')}
        if args.resume_copy_race and runtime!=json.loads((args.out/'runtime.json').read_text()):
            raise ValueError('Resumed acoustic runtime differs')
        if not args.resume_copy_race:(args.out/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
        done=set();counts=Counter()
        if args.resume_copy_race:
            for row in read_rows(args.out/'items.jsonl'):
                if row['id'] in done:raise ValueError('Duplicate committed receipt')
                for name in ('detail','features'):
                    if name in row and sha(args.out/row[name])!=row[name+'_sha256']:
                        raise ValueError('Committed evidence changed')
                done.add(row['id']);counts[row['status']]+=1
        while True:
            sync();rows=read_rows(plan/'teacher/items.jsonl')
            for row in rows:
                if row['id'] in done:continue
                if row['id'] not in jobs or row['system']!='pocket_teacher_250000_24l':raise ValueError('Unexpected source job')
                job=jobs[row['id']]
                if row['text']!=job['text'] or row['reference_sha256']!=job['reference_sha256']:raise ValueError('Rendered job/input differs')
                record={**row,'job':job}
                if row['status']!='ok_unreviewed':
                    record.update(status='input_waveform_rejected',reason=row.get('error','Explicit synthesis rejection'))
                else:
                    if not (plan/'teacher'/row['audio']).is_file():
                        current=json.loads((args.controller/'status.json').read_text())
                        if current['phase']=='complete_best_teacher_reference_audio':
                            sync()
                            if not (plan/'teacher'/row['audio']).is_file():raise FileNotFoundError(row['audio'])
                        else:
                            logging.warning('Manifest arrived before waveform; defer %s until next sync',row['id'])
                            continue
                    try:
                        samples=audio(plan/'teacher',row)
                        if len(samples)>20*16000:raise ValueError('Unexpected long rendered input needs whole-group CPU handling')
                        text=job.get('canonical_text',job['text'])
                        detail,arrays=engine.analyze(samples,text)
                        values={'phone_log_probs':arrays['phone_log_probs']}
                        if args.secondary:
                            with engine.torch.inference_mode():
                                other=secondary(**{k:v.to('mps') for k,v in extractor(samples,sampling_rate=16000,return_tensors='pt').items()})
                                other_lp=other.logits[0].log_softmax(-1).cpu().numpy()
                            if not np.isfinite(other_lp).all():raise ValueError('Nonfinite secondary phone output')
                            values['secondary_phone_log_probs']=other_lp.astype(np.float32)
                            detail['secondary_phone_frame_seconds']=len(samples)/16000/len(other_lp)
                        detail.update(generation_text=job['text'],canonical_analysis_text=text,source_audio_sha256=row['sha256'],
                            requested_vowel_change_class=job.get('requested_vowel_change_class'),actual_phonetic_gold=None)
                        dp=args.out/'details'/(row['clip_id']+'.json');fp=args.out/'features'/(row['clip_id']+'.npz')
                        if dp.exists() or fp.exists():raise FileExistsError('Unreceipted feature identity')
                        dp.write_text(json.dumps(detail,ensure_ascii=False,allow_nan=False)+'\n');np.savez_compressed(fp,**values)
                        record.update(status='ok_rendered_phone_frames',detail=str(dp.relative_to(args.out)),detail_sha256=sha(dp),
                                      features=str(fp.relative_to(args.out)),features_sha256=sha(fp))
                    except ValueError:
                        logging.exception('Explicit rendered input exclusion %s',row['id'])
                        record.update(status='unrepresentable_rendered_input',error=traceback.format_exc())
                with (args.out/'items.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                done.add(row['id']);counts[record['status']]+=1
                status('extracting_rendered_phonetic_frames',completed=len(done),planned=len(jobs),counts=dict(counts))
            controller=json.loads((args.controller/'status.json').read_text())
            if controller['phase']=='failed_explicitly':raise RuntimeError('Owned synthesis stopped explicitly; retain partial features')
            if controller['phase']=='complete_best_teacher_reference_audio':
                if len(rows)!=len(jobs):
                    sync();rows=read_rows(plan/'teacher/items.jsonl')
                if len(rows)!=len(jobs):raise ValueError('Completed source receipt count differs')
                if len(done)==len(jobs):break
            else:os.kill(controller['pid'],0)
            time.sleep(5)
        (args.out/'report.json').write_text(json.dumps({'status':'complete_rendered_phone_frames',
            'planned':len(jobs),'counts':dict(counts),'actual_pronunciation_gold':None},indent=2)+'\n')
        status('complete_rendered_phonetic_frames',completed=len(done),counts=dict(counts))
    except Exception:
        status('failed_explicitly',error=traceback.format_exc());raise


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Rendered phonetic frame extraction failed explicitly');raise
