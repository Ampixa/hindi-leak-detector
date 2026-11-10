"""Stream corrected quartets and content-checked physical word controls."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import csv
from datetime import datetime,timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import select
import subprocess
import time
import traceback

import numpy as np
import soundfile as sf

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import SYSTEMS,require_pair,sha
from hindi_detector.local_features import PROSODY_NAMES
from hindi_detector.splice_quality import boundary_issues
from hindi_detector.word_evidence import WordEvidenceEngine
from scripts.calibrate_asr_gate import audio

ROOT=Path(__file__).resolve().parents[1];LOG=logging.getLogger(__name__)


def read_rows(path):
    return [json.loads(line) for line in path.read_bytes().split(b'\n')[:-1]] if path.is_file() else []


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--resume-failed',action='store_true');args=parser.parse_args()
    if args.out!=ROOT/'reports/structured_evidence_v15':raise ValueError('Wrong corrected evidence namespace')
    if args.resume_failed:
        prior=json.loads((args.out/'status.json').read_text())
        if prior['phase']!='failed_explicitly' or 'Insufficient source word boundary support' not in prior.get('error',''):
            raise ValueError('Only inspected boundary failure may resume with preserved existing receipts')
        write_group(args.out/'resume_v17',{'previous_status.json':prior,
            'previous_asr_runtime.json':json.loads((args.out/'separate_asr_runtime.json').read_text()),
            'source_snapshot.json':{'sha256':sha(Path(__file__)),'source_utf8':Path(__file__).read_text()}},
            'Resume inspected boundary failure without rewriting existing audio/features. previous_status.json preserves exact failure; previous_asr_runtime.json preserves former child identity. Existing four-source text with no physical receipts is explicitly quarantined in text_exclusions.jsonl; only unseen text IDs continue. Review2026-11-09; remove redundant logs after consumer review.',
            ['Existing corrected model, input, source snapshots and per-clip checksums remain authoritative'])
    elif args.out.exists():raise FileExistsError(args.out)
    plan_path=ROOT/'data/teacher_indicf5_v14';pair=json.loads((plan_path/'model_pair.json').read_text());require_pair(pair)
    plan=json.loads((plan_path/'selection.json').read_text());expected=defaultdict(set)
    for job in plan['jobs']:
        for system in SYSTEMS:expected[job['text_id']].add((system,job['speaker_id']))
    priors_path=ROOT/'data/pocket_bulk_plan_v11/g2p_priors.json';priors=json.loads(priors_path.read_text())
    candidates_path=ROOT/'data/schwa_candidates_v1/candidates.json';candidates=json.loads(candidates_path.read_text())
    ipa_path=ROOT/'.sources/frontend/data/frontend/ipa_map.tsv'
    with ipa_path.open() as f:
        reader=csv.DictReader(f,delimiter='\t')
        if reader.fieldnames!=['phone','ipa_candidate','status','notes']:raise ValueError('Actual lexical IPA schema differs')
        ipa={r['phone']:r['ipa_candidate'] for r in reader}
    bank=[{'word':r['word'],'nepali_g2p':{'ipa_candidate_units':r['nepali_ipa_variants'][0]},
           'hindi_g2p':{'units_with_stress_preserved':r['hindi_espeak_raw'].strip().split('|')}} for r in priors]
    source_paths=[Path(__file__),ROOT/'hindi_detector/word_evidence.py',ROOT/'hindi_detector/schwa_probe.py',ROOT/'hindi_detector/ctc.py',
                  ROOT/'hindi_detector/structured_detector.py',ROOT/'scripts/content_gate_rpc.py',ROOT/'hindi_detector/asr_gate.py']
    if not args.resume_failed:
        write_group(args.out,{'inputs.json':{'model_pair_sha256':sha(plan_path/'model_pair.json'),'plan_sha256':sha(plan_path/'selection.json'),
            'prior_sha256':sha(priors_path),'candidate_sha256':sha(candidates_path),'ipa_map_sha256':sha(ipa_path),
            'physical_query_voice':'text-hash chooses one of the two reference groups; other teacher voice is comparison reference',
            'training_supervision':'known physical inserted-source word versus matched teacher sham; not pronunciation gold'},
            'source_snapshot.json':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'source_utf8':p.read_text()} for p in source_paths]},
            'Corrected training inputs. items.jsonl holds per-clip detail/NPZ/audio SHA and status; physical_items.jsonl holds2 actual source-word/sham controls per complete text with ASR decisions and hashes. details/*.json contain measured word times/site comparisons at0/30/60ms pads; features/*.npz contain phone/prosody/20x8 temporal phone channels, no hidden embeddings. controls/*.wav generated16kPCM16 physical inputs. status.json live counts/PID; runtime.json exact encoder, separate_asr_runtime.json exact isolated child gate. asr_worker.log explicit child errors. Failures/quarantines never become negative labels. Shared source-model/corpus rights apply; no HF upload. Current consumer corrected detector fitting/evaluation; review2026-11-09, remove redundant controls/features within24h after consumer closes, retain compact metrics/selected model.',
            ['.venv-indicf5/bin/python -m scripts.extract_corrected_training --out /Volumes/T9/hindi-detector/reports/structured_evidence_v15',
             'Pinned corrected model pair; no old generated WAVs or old classifier weights consumed'])
        for name in ('details','features','controls'):(args.out/name).mkdir()
    engine=WordEvidenceEngine(ROOT,g2p_word_bank=bank,device='mps',pitch_method='dio',temporal_bins=20,
        schwa_candidates=candidates,lexical_ipa=ipa)
    measured_runtime=json.dumps({'alignment':engine.aligner_inventory,'phone_encoder':engine.phone_inventory,
        'vocabulary':engine.vocab,'prosody_names':PROSODY_NAMES,'device':'mps','pitch_method':'dio','temporal_bins':20,
        'temporal_channels':['raw_ʌ','raw_ə','raw_ɐ','nonblank_conditional_ʌ','nonblank_conditional_ə','nonblank_conditional_ɐ','nonblank_mass','other_vowel_mass'],
        'other_vowel_ids':engine.other_vowels,'hidden_embeddings':False},indent=2)+'\n'
    if args.resume_failed:
        if json.loads(measured_runtime)!=json.loads((args.out/'runtime.json').read_text()):raise ValueError('Resumed encoders/runtime differ')
    else:(args.out/'runtime.json').write_text(measured_runtime)
    log=(args.out/('resume_v17/asr_worker.log' if args.resume_failed else 'asr_worker.log')).open('xb')
    child=subprocess.Popen([str(ROOT/'.venv-asr-gate/bin/python'),'-u','-m','scripts.content_gate_rpc'],cwd=ROOT,
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
    def response():
        ready,_,_=select.select([child.stdout],[],[],120)
        if not ready:raise TimeoutError('Content RPC did not respond; inspect owned ASR log')
        line=child.stdout.readline()
        if not line:raise RuntimeError('Content RPC exited unexpectedly; inspect owned ASR log')
        return json.loads(line)
    hello=response()
    if hello['status']!='ready':raise ValueError('Unexpected content RPC initialization')
    (args.out/'separate_asr_runtime.json').write_text(json.dumps({'pid':child.pid,**hello},indent=2)+'\n')
    prior_rows=read_rows(args.out/'items.jsonl') if args.resume_failed else []
    done={r['text_id'] for r in prior_rows};counts=Counter();physical_counts=Counter();failures=sum(r['status']=='failed_structured_features' for r in prior_rows)
    prior_groups=defaultdict(list)
    for r in prior_rows:prior_groups[r['text_id']].append(r)
    for tid,records in prior_groups.items():
        original=[r for r in records if r['kind']=='original'];physical=[r for r in records if r['kind']=='physical']
        if len(original)==4 and all(r['status']=='ok_structured_features' for r in original):counts[original[0]['row']['split']]+=1
        if len(physical)==2 and all(r['status']=='ok_structured_features' for r in physical):physical_counts[physical[0]['row']['split']]+=1
    if args.resume_failed:
        physical_ids={r['text_id'] for r in read_rows(args.out/'physical_items.jsonl')}
        interrupted=[tid for tid,rs in prior_groups.items() if tid not in physical_ids and len(rs)==4 and all(r['status']=='ok_structured_features' for r in rs)]
        if len(interrupted)!=1:raise ValueError('Inspected interrupted text inventory differs')
        with (args.out/'text_exclusions.jsonl').open('a') as f:f.write(json.dumps({'text_id':interrupted[0],'reason':'inspected zero-context word boundary interrupted original run','status':'quarantined_not_error_label'})+'\n')
    def status(phase,**extra):
        p=args.out/'status.json';temp=p.with_suffix('.partial')
        temp.write_text(json.dumps({'phase':phase,'pid':os.getpid(),'asr_pid':child.pid,'updated_utc':datetime.now(timezone.utc).isoformat(),
            'original_texts':dict(counts),'physical_texts':dict(physical_counts),'failed_clips':failures,'processed_texts':len(done),**extra},indent=2)+'\n');temp.replace(p)
    def append(path,row):
        with path.open('a') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
    def encode(row,samples,kind):
        nonlocal failures
        record={'row':row,'kind':kind,'text_id':row['text_id'],'clip_id':row['clip_id']}
        try:
            detail,arrays=engine.analyze(samples,row['text'])
            if any(not np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite retained structured features')
            focal=[i for i,w in enumerate(detail['words']) if w['word'] in row['target_words']]
            if not focal:raise ValueError('Planned focal word missing from aligned words')
            dp=args.out/'details'/(row['clip_id']+'.json');fp=args.out/'features'/(row['clip_id']+'.npz')
            detail.update(clip_id=row['clip_id'],focal_word_indices=focal,source_audio_sha256=row['sha256'])
            if dp.exists() or fp.exists():raise FileExistsError('Structured artifact identity already exists')
            dp.write_text(json.dumps(detail,ensure_ascii=False,allow_nan=False)+'\n');np.savez_compressed(fp,**arrays)
            record.update(status='ok_structured_features',detail=str(dp.relative_to(args.out)),detail_sha256=sha(dp),detail_bytes=dp.stat().st_size,
                features=str(fp.relative_to(args.out)),features_sha256=sha(fp),feature_bytes=fp.stat().st_size,focal_word_indices=focal)
            result={'detail':detail,**arrays}
        except Exception:
            LOG.exception('Explicit structured feature exclusion %s',row['clip_id']);record.update(status='failed_structured_features',error=traceback.format_exc());failures+=1;result=None
        append(args.out/'items.jsonl',record);return record,result
    try:
        status('extracting_corrected_word_controls')
        while True:
            rows=read_rows(ROOT/'reports/corrected_pair_v14/content_items.jsonl');groups=defaultdict(dict)
            for row in rows:
                if row['system'] not in SYSTEMS:raise ValueError('Wrong source in corrected ASR journal')
                if row['content_quality']['decision']=='pass':groups[row['text_id']][row['system'],row['speaker_id']]=row
            ready=[(tid,g) for tid,g in groups.items() if tid not in done and set(g)==expected[tid]]
            if not ready:
                state=json.loads((ROOT/'reports/corrected_pair_v14/status.json').read_text())
                if state['phase']=='failed_explicitly':raise RuntimeError('Corrected synthesis controller failed; preserve partial training inputs')
                if state['phase']=='complete_corrected_generation_and_content_checks':break
                status('waiting_for_more_corrected_quartets');os.kill(state['pid'],0);time.sleep(10);continue
            tid,group=ready[0];encoded={};complete=True
            for key,row in group.items():
                record,result=encode(row,audio(Path(row['source_pack']),row),'original')
                encoded[key]=(row,record,result)
                if result is None:complete=False
            done.add(tid)
            if not complete:continue
            split=next(iter(group.values()))['split'];counts[split]+=1
            voices=sorted({key[1] for key in group});ordinal=int(hashlib.sha256(tid.encode()).hexdigest()[:8],16)%2
            voice=voices[ordinal];other=voices[1-ordinal]
            nr,nrec,native=encoded[SYSTEMS[0],voice];hr,hrec,hindi=encoded[SYSTEMS[1],voice]
            reference=encoded[SYSTEMS[0],other][0];focal=nrec['focal_word_indices']
            if [w['word'] for w in native['detail']['words']]!=[w['word'] for w in hindi['detail']['words']]:raise ValueError('Same-text aligned word sequences differ')
            waves=[audio(Path(r['source_pack']),r) for r in (nr,hr)];physical=[]
            issues=boundary_issues([len(w) for w in waves],native['detail']['words'],hindi['detail']['words'],focal)
            if issues:
                append(args.out/'text_exclusions.jsonl',{'text_id':tid,'reason':'insufficient measured context for exact control fades','status':'quarantined_not_error_label','issues':issues})
                LOG.warning('Explicit splice-boundary quarantine %s: %s',tid,issues)
                status('extracting_corrected_word_controls');continue
            for label,donor in enumerate((native,hindi)):
                samples=waves[0].copy()
                for i in sorted(focal,reverse=True):
                    n,h=native['detail']['words'][i],donor['detail']['words'][i];start,end=int(n['start']*16000),int(n['end']*16000)
                    chunk=waves[label][int(h['start']*16000):int(h['end']*16000)].copy();width=min(80,len(chunk)//4,start,len(samples)-end)
                    if width<2:raise ValueError('Insufficient source word boundary support')
                    samples[start-width:start]*=np.linspace(1,0,width);samples[end:end+width]*=np.linspace(0,1,width)
                    chunk[:width]*=np.linspace(0,1,width);chunk[-width:]*=np.linspace(1,0,width)
                    samples=np.concatenate([samples[:start],chunk,samples[end:]])
                path=args.out/'controls'/(tid+'_'+str(label)+'.wav');sf.write(path,samples,16000,subtype='PCM_16')
                row={**nr,'system':'indicf5_stock' if label else SYSTEMS[0],'kind':'physical','condition':'f5_word_insert' if label else 'teacher_sham',
                    'audio':str(path.relative_to(args.out)),'sha256':sha(path),'bytes':path.stat().st_size,'seconds':len(samples)/16000,
                    'clip_id':hashlib.sha256(('physical_v15:'+tid+':'+str(label)).encode()).hexdigest()[:24],
                    'source_pack':str(args.out),'reference_clip_id':reference['clip_id'],'native_clip_id':nr['clip_id'],'donor_clip_id':hr['clip_id'],
                    'source_label':label,'actual_hindi_error_label':None}
                child.stdin.write(json.dumps({'pack':str(args.out),'row':row},ensure_ascii=False)+'\n');child.stdin.flush();reply=response()
                if reply['id']!=row['clip_id']:raise ValueError('Content RPC identity mismatch')
                row['content_quality']=reply['content_quality'];physical.append(row)
            good=all(r['content_quality']['decision']=='pass' for r in physical);features_ok=True
            for row in physical:
                if good:
                    receipt,result=encode(row,audio(args.out,row),'physical');features_ok &= result is not None
                    row['feature_status']=receipt['status']
                else:row['feature_status']='pair_content_quarantined'
                append(args.out/'physical_items.jsonl',row)
            if good and features_ok:physical_counts[split]+=1
            status('extracting_corrected_word_controls');LOG.info('Corrected physical training texts %s',dict(physical_counts))
        status('complete_corrected_structured_evidence',content_rows=len(rows))
    except Exception:
        status('failed_explicitly',error=traceback.format_exc());raise
    finally:
        child.stdin.close()
        try:code=child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            LOG.error('Owned ASR child did not exit on closed stdin; terminating PID%s',child.pid);child.terminate();code=child.wait(timeout=15)
        log.close()
        if code!=0:LOG.error('Owned ASR child exit code%s; inspect asr_worker.log',code)


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        LOG.exception('Corrected structured extraction failed; no silent retry');raise
