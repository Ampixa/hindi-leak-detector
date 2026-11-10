"""Frozen release gain/F0 sensitivity, reusing verified owned null waveforms."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime,timezone
import json
import logging
import os
from pathlib import Path
import traceback

import numpy as np
import pyworld
import soundfile as sf

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.dual_paired_inference import analyze_pair
from hindi_detector.dual_runtime import PhonePair
from hindi_detector.intrinsic_inference import assign_decisions
from hindi_detector.task_vowel_detector import decisions,load,score
from scripts.calibrate_asr_gate import audio
from scripts.extract_corrected_training import read_rows
from scripts.score_swar_sites import load_item

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'reports/release_voice_audit_v34';model=ROOT/'reports/task_vowel_release_v31';old=ROOT/'reports/vowel_voice_null_v20'
    if out.exists():raise FileExistsError(out)
    networks,state=load(model/'model.pt');digest=sha(model/'model.pt')
    write_group(out,{'protocol.json':{'model_sha256':digest,'source_null_items_sha256':sha(old/'items.jsonl'),
        'conditions':['gain_half','world_original_f0','world_plus3_semitones','world_minus3_semitones'],
        'native_controls':'Reuse21 preselected held-out source cases/3voice groups, exact old waveform hashes',
        'actual_error_checks':['real_yt_yt01_005/हैन','real_yt_yt01_008/किनकि'],
        'interpretation':'Pitch shifts preserve estimated envelope/aperiodicity and relative F0 shape; compare added flags against WORLD0. Reconstruction is not a perfect voice-only intervention. No refit.'},
        'source_snapshot.json':{'sha256':sha(Path(__file__)),'source_utf8':Path(__file__).read_text()}},
        'Frozen release nuisance audit. items.jsonl exact original/control waveform hashes and measured '
        'site scores/decisions; controls/*.wav only8 new actual-error transformations, Native84 waveforms '
        'reused without copying from vowel_voice_null_v20. report.json gain/reconstruction/pitch-added '
        'effects and known-error robustness. No raw frame/model copy, NOT on HF. Current consumer release '
        'voice-confound validation, review2026-11-09; compact conclusions retained, diagnostic waves/logs '
        'disposable24h after consumer closes. Source/MMS research rights unchanged.',
        ['.venv-indicf5/bin/python -m scripts.audit_release_voice'])
    (out/'controls').mkdir()
    def status(phase,**kw):
        p=out/'status.json';t=p.with_suffix('.partial');t.write_text(json.dumps({'phase':phase,'pid':os.getpid(),
            'updated_utc':datetime.now(timezone.utc).isoformat(),**kw},indent=2)+'\n');t.replace(p)
    try:
        status('loading_two_frozen_phone_probes');runtime=PhonePair(ROOT,state)
        groups=[ROOT/'reports/swar_corpus_features_v18',ROOT/'reports/swar_phone_crosscheck_v20']
        receipts=[{(r['id'],r['kind']):r for r in read_rows(g/'items.jsonl')} for g in groups];cache=[{},{}]
        def item(key,kind,probe):
            if (key,kind) not in cache[probe]:cache[probe][key,kind]=load_item(groups[probe],receipts[probe][key,kind])
            return cache[probe][key,kind]
        ipa=json.loads((ROOT/'data/swar_phone_plan_v18/ipa_map.json').read_text());lexical={r['word']:r for r in json.loads((ROOT/'data/swar_phone_plan_v18/words.json').read_text())}
        controls={}
        for r in read_rows(old/'items.jsonl'):
            key=r['id'],'real',r['condition']
            if key in controls:
                if controls[key]['audio_sha256']!=r['audio_sha256']:raise ValueError('Old null-control duplicate identity differs')
                continue
            controls[key]={'path':old/r['audio'],'audio_sha256':r['audio_sha256'],'source':'real','origin':'reused_existing_null_waveform'}
        targets={'real_yt_yt01_005':'हैन','real_yt_yt01_008':'किनकि'}
        for key in targets:
            receipt=receipts[0][key,'swar'];samples=audio(ROOT/'data/swar_speech_v18',{'audio':receipt['path'],'sha256':receipt['sha256']});x=samples.astype(np.float64)
            f0,times=pyworld.dio(x,16000,f0_floor=50,f0_ceil=650,frame_period=5);f0=pyworld.stonemask(x,f0,times,16000)
            spectral=pyworld.cheaptrick(x,f0,times,16000,f0_floor=50);ap=pyworld.d4c(x,f0,times,16000,threshold=0,fft_size=(spectral.shape[1]-1)*2)
            waves={'gain_half':samples*.5}
            for condition,shift in [('world_original_f0',0),('world_plus3_semitones',3),('world_minus3_semitones',-3)]:
                wave=pyworld.synthesize(np.ascontiguousarray(f0*2**(shift/12)),spectral,ap,16000,frame_period=5)
                wave=np.pad(wave,(0,max(0,len(samples)-len(wave))))[:len(samples)]
                peak=np.max(np.abs(wave))
                if peak>.98:wave*=.98/peak
                waves[condition]=wave.astype(np.float32)
            for condition,wave in waves.items():
                path=out/'controls'/(key+'_'+condition+'.wav');sf.write(path,wave,16000,subtype='PCM_16')
                controls[key,'swar',condition]={'path':path,'audio_sha256':sha(path),'source':'swar','origin':'new_actual_error_null_waveform'}
        records=[]
        for number,((key,source,condition),control) in enumerate(sorted(controls.items())):
            if source!=control['source']:raise ValueError('Null source identity collision')
            if sha(control['path'])!=control['audio_sha256']:raise ValueError('Null waveform bytes changed')
            samples,rate=sf.read(control['path'],dtype='float32')
            if rate!=16000 or samples.ndim!=1:raise ValueError('Unexpected null format')
            lp=runtime.encode(samples);queries=[]
            for probe in (0,1):
                original=item(key,control['source'],probe)
                queries.append({'detail':{**original['detail'],'phone_frame_seconds':len(samples)/16000/len(lp[probe])},'phone_log_probs':lp[probe]})
            words,sites,x=analyze_pair(queries[0],item(key,'real',0),queries[1],item(key,'real',1),lexical.__getitem__,ipa,*runtime.vocabularies)
            if len(x):
                p=score(networks,state,x);assign_decisions(words,sites,p,decisions(state,x,p))
            else:assign_decisions(words,[],[],[])
            result={'id':key,'condition':condition,'source':control['source'],'audio':str(control['path']),
                'audio_sha256':control['audio_sha256'],'audio_bytes':control['path'].stat().st_size,
                'sites':[{'word':words[s['word_ordinal']]['word'],'word_ordinal':s['word_ordinal'],'kind':s['site']['kind'],'index':s['site']['index'],
                    'score':s['controlled_vowel_change_score'],'decision':s['decision']} for s in sites],
                'target_word':targets.get(key) if control['source']=='swar' else None}
            if result['target_word']:result['target_flagged']=any(w['word']==result['target_word'] and w['decision']=='strong_vowel_change' for w in words)
            records.append(result)
            with (out/'items.jsonl').open('a') as stream:stream.write(json.dumps(result,ensure_ascii=False)+'\n')
            status('checking_gain_pitch_and_known_errors',completed=number+1,total=len(controls))
        native=[r for r in records if r['source']=='real'];by_case=defaultdict(dict)
        for r in native:by_case[r['id']][r['condition']]=r
        counts={};added={}
        for condition in ('gain_half','world_original_f0','world_plus3_semitones','world_minus3_semitones'):
            subset=[r for r in native if r['condition']==condition];flags=[s for r in subset for s in r['sites'] if s['decision']=='strong_vowel_change']
            counts[condition]={'cases':len(subset),'sites':sum(len(r['sites']) for r in subset),'site_flags':len(flags),
                'cases_with_flags':sum(any(s['decision']=='strong_vowel_change' for s in r['sites']) for r in subset)}
        for condition in ('world_plus3_semitones','world_minus3_semitones'):
            diffs=[];introduced=0
            for variants in by_case.values():
                a={(s['word_ordinal'],s['kind'],s['index']):s for s in variants['world_original_f0']['sites']}
                b={(s['word_ordinal'],s['kind'],s['index']):s for s in variants[condition]['sites']}
                if set(a)!=set(b):raise ValueError('Fixed reference null site coverage differs')
                introduced+=sum(b[k]['decision']=='strong_vowel_change' and a[k]['decision']!='strong_vowel_change' for k in a);diffs.extend(b[k]['score']-a[k]['score'] for k in a)
            added[condition]={'added_flags_over_WORLD0':introduced,'median_score_shift':float(np.median(diffs))}
        report={'status':'complete_release_voice_audit','model_sha256':digest,'native_controls':counts,'pitch_added_effect':added,
            'known_error_checks':[{'id':r['id'],'condition':r['condition'],'word':r['target_word'],'flagged':r['target_flagged']} for r in records if r['source']=='swar'],
            'actual_hindi_error_accuracy':None,'limits':'Nuisance sensitivity relative to the same recorded reference, not Native-error FPR or validated Hindi intonation. WORLD reconstruction can alter phonetic detail.'}
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        if sha(model/'model.pt')!=digest:raise ValueError('Release model changed during voice audit')
        status(report['status']);print(json.dumps(report,ensure_ascii=False))
    except Exception:
        status('failed_explicitly',error=traceback.format_exc());raise


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Release voice audit failed explicitly');raise
