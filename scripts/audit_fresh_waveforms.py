"""Fresh known waveform interventions after final candidate freeze."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import traceback

import numpy as np
import soundfile as sf

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.dual_runtime import PhonePair
from hindi_detector.edit_geometry import map_word
from hindi_detector.evaluation import binary_metrics
from hindi_detector.intrinsic_vowels import make_paths,measure
from hindi_detector.paired_local_vowels import vector
from hindi_detector.phone_lattice import path_score
from hindi_detector.pronunciation_sites import evidence_for_site,paired_site_features,select_reference_structure,windows
from hindi_detector.task_vowel_detector import decisions,load,score
from hindi_detector.vowel_detector import load as old_load,score as old_score,decisions as old_decisions
from scripts.build_vowel_controls import splice
from scripts.calibrate_asr_gate import audio
from scripts.extract_corrected_training import read_rows
from scripts.prepare_intrinsic_minimal import template
from scripts.score_swar_sites import load_item

ROOT=Path(__file__).resolve().parents[1]


def native_observed(f):return f[2]<0 and f[8]>-1.5 and f[30]==1


def full_gap(item,ordinal,compiled,variant,contrast,evidence,alternative=False):
    crops=windows(item,ordinal);native,other,at=make_paths(compiled,variant,contrast,evidence['retained_native_indices'])
    units=other if alternative else native;trace=path_score(crops[1][2],units,trace=True)
    if at and units[at-1]<=compiled['any_vowel'] or at+1<len(units) and units[at+1]<=compiled['any_vowel']:
        raise ValueError('Adjacent-vowel sequence needs a separate full-region segmentation')
    left=crops[1][1]+trace['spans'][at-1]['end_frame'] if at else crops[0][1]
    right=crops[1][1]+trace['spans'][at+1]['start_frame'] if at+1<len(units) else crops[0][1]+len(crops[0][2])
    start=max(crops[0][1],left);end=min(crops[0][1]+len(crops[0][2]),right)
    dt=item['detail']['phone_frame_seconds'];return int(start*dt*16000),int(end*dt*16000)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--full-gap',action='store_true')
    parser.add_argument('--plan',type=Path,default=ROOT/'data/minimal_audit_v29')
    parser.add_argument('--frames',type=Path,default=ROOT/'reports/minimal_audit_features_v29')
    parser.add_argument('--intrinsic',type=Path,default=ROOT/'reports/intrinsic_audit_v29')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    model=ROOT/'reports/task_vowel_release_v31';out=args.out or ROOT/('reports/fresh_gap_audit_v32' if args.full_gap else 'reports/fresh_waveform_audit_v31');base=args.intrinsic;frames=args.frames;plan=args.plan
    if out.exists():raise FileExistsError(out)
    networks,state=load(model/'model.pt');expected=json.loads((model/'report.json').read_text())['model_sha256']
    if sha(model/'model.pt')!=expected:raise ValueError('Frozen release changed before fresh waveform audit')
    old,old_state=old_load(ROOT/'reports/vowel_detector_v19/model.pt')
    write_group(out,{'protocol.json':{'model_sha256':expected,'source_plan_sha256':sha(plan/'selection.json'),
        'source_intrinsic_items_sha256':sha(base/'items.jsonl'),
        'labels':'Known actual observed-nucleus removal/replacement versus matched sham reinsertion; not Hindi-error truth',
        'eligibility':'Both probes favor/observe canonical native vowel; measured region fully inside word and'+('40..300ms' if args.full_gap else '20..200ms')+'; quality donor alternative observed/favored by both, duration ratio0.5..2',
        'geometry':'Exact piecewise coordinate mapping through the actual cut/replacement;2ms matched join fades',
        'region':('Full interval between neighboring consonant peak spans, capped at owned word bounds; includes blank-labeled vowel frames. Follows the short-nucleus audit; models remain frozen.' if args.full_gap else 'CTC focal-unit peak span only; this is a nucleus edit and may leave the phoneme audible.'),
        'selection':'No classifier probability used for input selection; all new words/conditioning groups excluded from fit; no threshold/model changes after audit'},
        'source_snapshot.json':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'source_utf8':p.read_text()}
            for p in (Path(__file__),ROOT/'hindi_detector/edit_geometry.py',ROOT/'hindi_detector/dual_runtime.py')]},
        'Fresh task-compatible actual waveform edit audit after release freeze. controls/*.wav actual '
        '16kPCM16 same-voice sham/change pairs, every filename/bytes/SHA/source and exact segment geometry '
        'in items.jsonl. That file also retains features/decisions for release and legacy baseline; '
        'exclusions.json every preflight exclusion, report.json actual counts/metrics/limits. No new '
        'weights, NOT on HF. Source copies remain documented in the exact --plan teacher/ receipts. Current '
        'consumer release validation; review2026-11-09. Keep compact conclusions/source pins; raw controls '
        'disposable24h after consumer closes unless explicitly selected release evidence. Inherited rights.',
        ['.venv-indicf5/bin/python -m scripts.audit_fresh_waveforms'])
    (out/'controls').mkdir()
    def status(phase,**kw):
        p=out/'status.json';t=p.with_suffix('.partial');t.write_text(json.dumps({'phase':phase,'pid':os.getpid(),
            'updated_utc':datetime.now(timezone.utc).isoformat(),**kw},indent=2)+'\n');t.replace(p)
    try:
        status('loading_verified_dual_phone_runtime');runtime=PhonePair(ROOT,state)
        selection=json.loads((plan/'selection.json').read_text());wordbank={w['word_id']:w for w in selection['words']};jobs={j['id']:j for j in selection['jobs']}
        receipts={r['id']:r for r in read_rows(frames/'items.jsonl')};intrinsic=read_rows(base/'items.jsonl');by_job={r['job_id']:r for r in intrinsic}
        by_tuple={(j['word_id'],j['voice_ordinal'],j['requested_vowel_change_class']):j['id'] for j in selection['jobs']}
        ipa=json.loads((ROOT/'data/swar_phone_plan_v18/ipa_map.json').read_text());records=[];excluded=[]
        for q in [r for r in intrinsic if r['label']==0]:
            job=jobs[q['job_id']];word=wordbank[q['word_id']]
            if not native_observed(q['features']) or not native_observed(q['secondary_features']):
                excluded.append({'job_id':q['job_id'],'reason':'Canonical Native nucleus not established by both probes'});continue
            receipt=receipts[q['job_id']];reference=load_item(frames,receipt);detail=reference['detail'];ordinal=next(i for i,w in enumerate(detail['words']) if w['word']==word['word']);original_word=detail['words'][ordinal]
            span=q['evidence']['native_site_span'];origin=q['evidence']['windows'][1]['start_frame'];dt=detail['phone_frame_seconds']
            start=int((origin+span['start_frame'])*dt*16000);end=int((origin+span['end_frame'])*dt*16000)
            if args.full_gap:
                try:start,end=full_gap(reference,ordinal,*template(word,ipa,runtime.vocabularies[0]),q['evidence'])
                except ValueError as error:
                    excluded.append({'job_id':q['job_id'],'reason':str(error)});continue
            low,high=(.040,.300) if args.full_gap else (.020,.200)
            if not original_word['start']*16000<=start<end<=original_word['end']*16000 or not low<=(end-start)/16000<=high:
                excluded.append({'job_id':q['job_id'],'reason':f'Observed region does not have a fully owned{low*1000:g}..{high*1000:g}ms word interval'});continue
            samples=audio(plan/'teacher',receipt);donor=None;donor_record=None
            if word['kind']=='vowel_quality':
                other_id=by_tuple[q['word_id'],job['voice_ordinal'],1];other=by_job[other_id]
                if any(f[1]<=0 or f[9]<=-1.5 for f in (other['features'],other['secondary_features'])):
                    excluded.append({'job_id':q['job_id'],'reason':'Edited-reading donor alternative not established by both probes'});continue
                other_receipt=receipts[other_id];other_detail=json.loads((frames/other_receipt['detail']).read_text())
                if sha(frames/other_receipt['detail'])!=other_receipt['detail_sha256']:raise ValueError('Donor detail changed')
                ds=other['evidence']['alternative_site_span'];dorigin=other['evidence']['windows'][1]['start_frame'];ddt=other_detail['phone_frame_seconds']
                a=int((dorigin+ds['start_frame'])*ddt*16000);b=int((dorigin+ds['end_frame'])*ddt*16000)
                if args.full_gap:
                    other_item=load_item(frames,other_receipt);other_ordinal=next(i for i,w in enumerate(other_detail['words']) if w['word']==word['word'])
                    try:a,b=full_gap(other_item,other_ordinal,*template(word,ipa,runtime.vocabularies[0]),other['evidence'],alternative=True)
                    except ValueError as error:
                        excluded.append({'job_id':q['job_id'],'reason':str(error)});continue
                donor=audio(plan/'teacher',other_receipt)[a:b]
                if not .5<=len(donor)/(end-start)<=2:
                    excluded.append({'job_id':q['job_id'],'reason':'Actual donor nucleus duration is outside0.5..2 ratio'});continue
                donor_record={'job_id':other_id,'audio_sha256':other_receipt['sha256'],'start_sample':a,'end_sample':b}
            with np.load(frames/receipt['features'],allow_pickle=False) as values:secondary_reference_lp=values['secondary_phone_log_probs']
            references=[{'detail':{'words':[original_word],'phone_frame_seconds':detail['phone_frame_seconds']},'phone_log_probs':reference['phone_log_probs']},
                {'detail':{'words':[original_word],'phone_frame_seconds':detail['secondary_phone_frame_seconds']},'phone_log_probs':secondary_reference_lp}]
            templates=[template(word,ipa,vocab) for vocab in runtime.vocabularies];pair=[]
            for label in (0,1):
                chunk=samples[start:end] if label==0 else np.empty(0,np.float32) if donor is None else donor
                tag='full_gap_v32' if args.full_gap else 'fresh_physical_v31'
                edited=splice(samples,start,end,chunk);mapped=map_word(original_word,start,end,len(chunk));identity=hashlib.sha256((q['job_id']+':'+tag+':'+str(label)).encode()).hexdigest()[:24]
                path=out/'controls'/(identity+'.wav');sf.write(path,edited,16000,subtype='PCM_16');actual,rate=sf.read(path,dtype='float32')
                if rate!=16000 or actual.ndim!=1:raise ValueError('Saved control format changed')
                lp=runtime.encode(actual);features=[];measurements=[]
                for probe in (0,1):
                    compiled,variant,contrast=templates[probe];query={'detail':{'words':[mapped],'phone_frame_seconds':len(actual)/16000/len(lp[probe])},'phone_log_probs':lp[probe]}
                    qf,qe=measure(compiled,variant,contrast,windows(query,0));rf,re=measure(compiled,variant,contrast,windows(references[probe],0))
                    features.append(vector(qf,rf,qe,re,word['kind']));measurements.append({'query':qe,'reference':re})
                x=np.r_[features[0],features[1]][None].astype(np.float32);p=score(networks,state,x);decision=decisions(state,x,p)[0]
                baseline={'decision':'uncertain','score':None}
                try:
                    compiled,variant,contrast=templates[0];rc=windows(references[0],0);_,reduced=select_reference_structure({**compiled,'variants':[variant]},rc)
                    re=evidence_for_site(compiled,variant,reduced,contrast,rc);qe=evidence_for_site(compiled,variant,reduced,contrast,windows({'detail':{'words':[mapped],'phone_frame_seconds':len(actual)/16000/len(lp[0])},'phone_log_probs':lp[0]},0))
                    f=paired_site_features(qe,re,contrast)[None];bp=old_score(old,old_state,f);baseline={'decision':old_decisions(old_state,f,bp)[0],'score':float(bp[0])}
                except ValueError as error:
                    logging.warning('Explicit legacy baseline abstention %s: %s',identity,error);baseline['unavailable_reason']=str(error)
                pair.append({'word':word['word'],'word_id':word['word_id'],'voice':job['speaker_id'],'kind':word['kind'],'label':label,
                    'job_id':q['job_id'],'source_audio_sha256':receipt['sha256'],'audio':str(path.relative_to(out)),'audio_bytes':path.stat().st_size,'audio_sha256':sha(path),
                    'geometry':{'start_sample':start,'end_sample':end,'replacement_samples':len(chunk),'original_word':{k:original_word[k] for k in ('start','end')},'mapped_word':{k:mapped[k] for k in ('start','end')}},
                    'donor':donor_record if label else None,'features':x[0].tolist(),'evidence':measurements,
                    'release':{'score':float(p[0]),'decision':decision},'legacy_v19':baseline,'actual_hindi_error_label':None})
            for record in pair:
                records.append(record)
                with (out/'items.jsonl').open('a') as stream:stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
            status('evaluating_fresh_actual_waveform_edits',pairs=len(records)//2,excluded=len(excluded))
        (out/'exclusions.json').write_text(json.dumps(excluded,ensure_ascii=False,indent=2)+'\n');result={}
        for name in ('release','legacy_v19'):
            result[name]={}
            for kind in ('vowel_deletion','vowel_quality'):
                subset=[r for r in records if r['kind']==kind]
                if not subset:continue
                y=np.array([r['label'] for r in subset]);pred=np.array([r[name]['decision']=='strong_vowel_change' for r in subset],float)
                result[name][kind]=binary_metrics(y,pred,.5)
        report={'status':'complete_fresh_waveform_audit','created_utc':datetime.now(timezone.utc).isoformat(),'model_sha256':expected,
            'pairs':len(records)//2,'word_families':len({r['word'] for r in records}),'conditioning_groups':sorted({r['voice'] for r in records}),
            'excluded_canonical_readings':len(excluded),'metrics':result,'actual_hindi_error_accuracy':None,
            'limits':['Known waveform nucleus edits establish controlled-change truth, not real Hindi-error truth or fluent pronunciation.',
                'Source/donor selection uses both phone probes and favors clearly observed nuclei; results do not cover weak/unrepresentable cases.',
                'New words/conditioning groups/carrier are held out, but same-voice sham/control comparison is easier than generated-reference deployment.']}
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if sha(model/'model.pt')!=expected:raise ValueError('Release changed during fresh waveform evaluation')
        status(report['status'],pairs=report['pairs']);print(json.dumps(report))
    except Exception:
        status('failed_explicitly',error=traceback.format_exc());raise


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Fresh waveform audit failed explicitly');raise
