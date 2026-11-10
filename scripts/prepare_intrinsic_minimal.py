"""Data-only natural minimal-pair matrices, including independent probe audit."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import time
import traceback

import numpy as np

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.intrinsic_vowels import FEATURE_NAMES, measure
from hindi_detector.phone_lattice import VOWELS, units_for_project
from hindi_detector.pronunciation_sites import windows
from scripts.extract_corrected_training import read_rows

ROOT=Path(__file__).resolve().parents[1]


def template(word,ipa,vocabulary):
    units,projection=units_for_project(word['native_project_phones'],ipa,vocabulary)
    edited,_=units_for_project(word['edited_project_phones'],ipa,vocabulary)
    at=word['site_index']
    contrast={'kind':word['kind'],'index':at}
    if word['kind']=='vowel_deletion':
        if edited!=units[:at]+units[at+1:] or word['native_project_phones'][at]!='ax':
            raise ValueError('Declared minimal deletion does not match actual G2P')
    else:
        if len(edited)!=len(units) or [i for i,(a,b) in enumerate(zip(units,edited)) if a!=b]!=[at]:
            raise ValueError('Declared quality edit does not match actual G2P')
        contrast['alternative_unit']=edited[at]
    return {'any_vowel':frozenset(vocabulary[t] for t in VOWELS if t in vocabulary)}, {
        'units':units,'phones':word['native_project_phones'],'projections':projection},contrast


def qualified(pair):
    """Separate frozen acoustic model, never primary-feature-derived labels."""
    a,b=pair
    return bool(a[2]<0 and a[8]>-1.5 and b[1]>0 and b[6]>-1.5 and
                (b[9]>-1.5 if b[29] else True) and a[30]==1 and b[31]==1)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'reports/intrinsic_minimal_v24')
    parser.add_argument('--plan',type=Path,default=ROOT/'data/minimal_vowels_v23')
    parser.add_argument('--features',type=Path,default=ROOT/'reports/minimal_vowel_features_v23')
    args=parser.parse_args();out=args.out
    if out.exists():raise FileExistsError(out)
    base=args.features;plan=args.plan
    selection=json.loads((plan/'selection.json').read_text())
    job_lookup={(j['word_id'],j['voice_ordinal'],j['requested_vowel_change_class']):j['id'] for j in selection['jobs']}
    if len(job_lookup)!=len(selection['jobs']):raise ValueError('Duplicate minimal job tuple')
    write_group(out,{'protocol.json':{'plan_sha256':sha(plan/'selection.json'),'feature_names':FEATURE_NAMES,
        'labels':'Requested single phonemic edit, not independently observed Hindi-error gold',
        'qualification':'Secondary LV60 model: all3 native margins negative for class0; all3 alternative margins positive for class1; expected/alternative focal emissions and fit >-1.5; feasible paths',
        'test_policy':'Prepare isolated test arrays without fitting; trainer must freeze before reading test features'},
        'source_snapshot.json':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'source_utf8':p.read_text()}
            for p in (Path(__file__),ROOT/'hindi_detector/intrinsic_vowels.py')]},
        'Natural same-teacher same-voice vowel requests transformed into reference-free32-feature matrices. '
        'items.jsonl records both primary/secondary evidence, requested label and whole-pair realization '
        'qualification. arrays_train/development/test.npz separate primary features/labels/qualification; '
        'index.json preserves row identities, report.json counts and SHA inventory. Exclusions are explicit '
        'whole pairs; no source identity in numeric model inputs. No new audio/model copy. Current consumer '
        'reference-free candidate; review2026-11-09, retain compact data/model pins, remove redundant raw '
        'frame caches24h after experiment closes. NOT on HF; source rights/MMS research restrictions apply.',
        ['.venv-indicf5/bin/python -m scripts.prepare_intrinsic_minimal'])
    def status(phase,**kw):
        p=out/'status.json';t=p.with_suffix('.partial');t.write_text(json.dumps({'phase':phase,'pid':os.getpid(),
            'updated_utc':datetime.now(timezone.utc).isoformat(),**kw},indent=2)+'\n');t.replace(p)
    try:
        status('waiting_for_natural_acoustic_features')
        while True:
            state=json.loads((base/'status.json').read_text())
            if state['phase']=='failed_explicitly':raise RuntimeError('Natural acoustic producer failed')
            if state['phase']=='complete_rendered_phonetic_frames':break
            os.kill(state['pid'],0);time.sleep(5)
        runtime=json.loads((base/'runtime.json').read_text());ipa=json.loads((ROOT/'data/swar_phone_plan_v18/ipa_map.json').read_text())
        receipts={r['id']:r for r in read_rows(base/'items.jsonl')};index=[];vectors=[];ys=[];qs=[];excluded=[]
        for number,word in enumerate(selection['words']):
            for voice in (0,1):
                pair=[]
                try:
                    for label in (0,1):
                        job_id=job_lookup[word['word_id'],voice,label]
                        receipt=receipts[job_id]
                        if receipt['status']!='ok_rendered_phone_frames':raise ValueError('Explicit source feature rejection')
                        for name in ('detail','features'):
                            if sha(base/receipt[name])!=receipt[name+'_sha256']:raise ValueError('Rendered evidence bytes changed')
                        detail=json.loads((base/receipt['detail']).read_text())
                        ordinals=[i for i,w in enumerate(detail['words']) if w['word']==word['word']]
                        if len(ordinals)!=1:raise ValueError('Canonical target is not uniquely aligned')
                        primary={};secondary={}
                        with np.load(base/receipt['features'],allow_pickle=False) as data:
                            primary['phone_log_probs']=data['phone_log_probs'];secondary['phone_log_probs']=data['secondary_phone_log_probs']
                        primary['detail']=detail;secondary['detail']={**detail,'phone_frame_seconds':detail['secondary_phone_frame_seconds']}
                        f,e=measure(*template(word,ipa,runtime['vocabulary']),windows(primary,ordinals[0]))
                        sf,se=measure(*template(word,ipa,runtime['secondary_vocabulary']),windows(secondary,ordinals[0]))
                        pair.append((f,sf,{'job_id':job_id,'word':word['word'],'word_id':word['word_id'],'voice':receipt['speaker_id'],
                            'split':word['partition'],'label':label,'kind':word['kind'],'features':f.tolist(),
                            'secondary_features':sf.tolist(),'evidence':e,'secondary_evidence':se,
                            'source_sha256':receipt['sha256'],'source_features_sha256':receipt['features_sha256']}))
                except ValueError as error:
                    logging.warning('Explicit minimal pair exclusion %s/%s: %s',word['word'],voice,error)
                    excluded.append({'word_id':word['word_id'],'voice_ordinal':voice,'reason':str(error)});continue
                qualification=qualified([r[1] for r in pair])
                for f,sf,record in pair:
                    record.update(row_index=len(index),secondary_realization_qualified=qualification)
                    with (out/'items.jsonl').open('a') as stream:stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                    index.append({k:record[k] for k in ('row_index','job_id','word','word_id','voice','split','label','kind','secondary_realization_qualified')})
                    vectors.append(f);ys.append(record['label']);qs.append(qualification)
            status('measuring_natural_minimal_sites',words=number+1,total=len(selection['words']),rows=len(index))
        matrix=np.stack(vectors);labels=np.array(ys,np.int64);qualification=np.array(qs,bool);inventory=[]
        for part in ('train','development','test'):
            mask=np.array([r['split']==part for r in index]);p=out/('arrays_'+part+'.npz')
            np.savez_compressed(p,features=matrix[mask],labels=labels[mask],qualified=qualification[mask])
            inventory.append({'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size,'rows':int(mask.sum()),
                'qualified_pairs':int(qualification[mask].sum())//2})
        (out/'index.json').write_text(json.dumps(index,ensure_ascii=False)+'\n')
        (out/'exclusions.json').write_text(json.dumps(excluded,ensure_ascii=False,indent=2)+'\n')
        report={'status':'complete_natural_intrinsic_matrices','inventory':inventory,'index_sha256':sha(out/'index.json'),
            'feature_names':FEATURE_NAMES,'runtime_sha256':sha(base/'runtime.json'),'source_runtime':runtime,
            'pairs':len(index)//2,'qualified_pairs':sum(qs)//2,'excluded_pairs':len(excluded),
            'counts':dict(Counter(r['split']+'/'+r['kind'] for r in index)),'actual_hindi_error_accuracy':None}
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');status(report['status'],pairs=report['pairs'],qualified_pairs=report['qualified_pairs'])
    except Exception:
        status('failed_explicitly',error=traceback.format_exc());raise


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Natural intrinsic data preparation failed explicitly');raise
