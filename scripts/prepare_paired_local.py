"""Build reference-conditioned57-feature datasets from verified existing evidence."""
from __future__ import annotations

from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
import logging
import os
from pathlib import Path
import time
import traceback

import numpy as np

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.intrinsic_vowels import measure
from hindi_detector.paired_local_vowels import FEATURE_NAMES,vector
from hindi_detector.phone_lattice import VOWELS,units_for_project
from hindi_detector.pronunciation_sites import select_reference_structure,windows
from scripts.extract_corrected_training import read_rows
from scripts.score_swar_sites import load_item

ROOT=Path(__file__).resolve().parents[1]


def native_supported(features):
    f=np.asarray(features)
    return bool(f[2]<0 and f[8]>-1.5 and f[5]>-1.5 and f[30]==1)


def main():
    out=ROOT/'reports/paired_local_features_v27'
    if out.exists():raise FileExistsError(out)
    write_group(out,{'protocol.json':{'features':FEATURE_NAMES,
        'inputs':['intrinsic_physical_v24','intrinsic_minimal_v24','indic_phonetic_pairs_v25','intrinsic_internal_v26'],
        'references':'Physical original same-speaker reference; natural/source canonical other conditioning voice, never query-versus-itself',
        'qualification':'Physical intervention truth; natural/source entire query pair qualified by LV60 plus independently expected Native vowel in other-voice reference',
        'partitions':'Existing disjoint word/voice partitions preserved; new internal_v26 test first candidate consumption after freeze'},
        'source_snapshot.json':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'source_utf8':p.read_text()}
            for p in (Path(__file__),ROOT/'hindi_detector/paired_local_vowels.py',ROOT/'hindi_detector/intrinsic_vowels.py')]},
        'Reference-conditioned57-feature dataset. items.jsonl pins source query/reference measured evidence, '
        'identities, labels and qualification; index.json row metadata; arrays_DOMAIN_PART.npz separate '
        'train/dev/test matrices/labels/qualification. report.json actual counts/byte/SHA inventory. No '
        'waveform/model copies. Current consumer stronger word-vowel detector; review2026-11-09, retain '
        'compact data/model provenance, retire redundant diagnostic copies24h after experiment closes. '
        'NOT on HF; inherited source and MMS research restrictions.',
        ['.venv-indicf5/bin/python -m scripts.prepare_paired_local'])
    def status(phase,**kw):
        p=out/'status.json';t=p.with_suffix('.partial');t.write_text(json.dumps({'phase':phase,'pid':os.getpid(),
            'updated_utc':datetime.now(timezone.utc).isoformat(),**kw},indent=2)+'\n');t.replace(p)
    vectors=[];index=[];exclusions=[]
    def append(q,r,domain,reference_id,qualified):
        f=vector(q['features'],r['features'],q['evidence'],r['evidence'],q['kind'])
        record={**{k:q[k] for k in ('word','voice','split','kind','label')},'domain':domain,
            'key':q.get('job_id',q.get('key')),'reference_id':reference_id,'qualified':bool(qualified),
            'features':f.tolist(),'query_evidence':q['evidence'],'reference_evidence':r['evidence'],'row_index':len(index)}
        with (out/'items.jsonl').open('a') as stream:stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
        index.append({k:record[k] for k in ('row_index','word','voice','split','kind','label','domain','key','reference_id','qualified')});vectors.append(f)
    try:
        status('building_physical_local_reference_evidence')
        source=ROOT/'reports/swar_corpus_features_v18';runtime=json.loads((source/'runtime.json').read_text());vocab=runtime['vocabulary']
        ipa=json.loads((ROOT/'data/swar_phone_plan_v18/ipa_map.json').read_text());lexical={r['word']:r for r in json.loads((ROOT/'data/swar_phone_plan_v18/words.json').read_text())}
        receipts={r['id']:r for r in read_rows(source/'items.jsonl') if r['kind']=='real'};cache={};structures={};refcache={}
        def structure(key,ordinal):
            if (key,ordinal) in structures:return structures[key,ordinal]
            if key not in cache:cache[key]=load_item(source,receipts[key])
            item=cache[key];raw=lexical[item['detail']['words'][ordinal]['word']];variants=[]
            for number,phones in enumerate(raw['nepali_project_variants']):
                try:units,projections=units_for_project(phones,ipa,vocab)
                except ValueError as error:
                    logging.info('Documented unsupported native variant skipped %s: %s',raw['word'],error);continue
                variants.append({'variant_id':number,'phones':[p for p in phones if p!='.'],'units':units,'projections':projections})
            compiled={'variants':variants,'any_vowel':frozenset(vocab[t] for t in VOWELS if t in vocab)}
            crops=windows(item,ordinal);variant,_=select_reference_structure(compiled,crops)
            structures[key,ordinal]=compiled,variant,crops;return structures[key,ordinal]
        old_rows=read_rows(ROOT/'reports/vowel_controls_v19/items.jsonl')
        original={(r['key'],r['ordinal'],r['unit_index'],r['kind'],r['label']):r for r in old_rows}
        for q in read_rows(ROOT/'reports/intrinsic_physical_v24/items.jsonl'):
            row=original[q['key'],q['ordinal'],q['unit_index'],q['kind'],q['label']]
            key=(q['key'],q['ordinal'],q['unit_index'],q['kind'])
            if key not in refcache:
                c,v,crops=structure(q['key'],q['ordinal']);contrast={'kind':q['kind'],'index':q['unit_index']}
                if q['kind']=='vowel_quality':
                    donor=row['other_donor'];_,dv,_=structure(donor['key'],donor['ordinal']);contrast['alternative_unit']=dv['units'][donor['unit_index']]
                f,e=measure(c,v,contrast,crops)
                if e['native_target_phones']!=q['evidence']['native_target_phones']:raise ValueError('Physical native template changed')
                refcache[key]={'features':f.tolist(),'evidence':e}
            append(q,refcache[key],'physical',q['key']+':original',True)
        natural_specs=[('intrinsic_minimal_v24','minimal_vowels_v23'),('intrinsic_internal_v26','minimal_internal_v26')]
        for group,plan_name in natural_specs:
            base=ROOT/'reports'/group;status('waiting_for_natural_local_data',group=group,rows=len(index))
            while True:
                previous=json.loads((base/'status.json').read_text())
                if previous['phase']=='failed_explicitly':raise RuntimeError('Natural producer failed: '+group)
                if previous['phase']=='complete_natural_intrinsic_matrices':break
                os.kill(previous['pid'],0);time.sleep(5)
            rows=read_rows(base/'items.jsonl');jobs={j['id']:j for j in json.loads((ROOT/'data'/plan_name/'selection.json').read_text())['jobs']}
            lookup={(r['word_id'],jobs[r['job_id']]['voice_ordinal'],r['label']):r for r in rows}
            for q in rows:
                voice=jobs[q['job_id']]['voice_ordinal'];r=lookup.get((q['word_id'],1-voice,0))
                if r is None:
                    exclusions.append({'domain':group,'query':q['job_id'],'reason':'Missing other-voice canonical reference'});continue
                eligible=q['secondary_realization_qualified'] and native_supported(r['secondary_features'])
                append(q,r,'natural',r['job_id'],eligible)
        status('building_indic_local_reference_evidence',rows=len(index))
        rows=read_rows(ROOT/'reports/indic_phonetic_pairs_v25/items.jsonl')
        selections={p['id']:p for p in json.loads((ROOT/'reports/indic_phonetic_pairs_v25/selection.json').read_text())}
        lookup=defaultdict(list)
        for row in rows:
            if row['label']==0:
                tid=selections[row['key']]['receipts'][0]['row']['text_id']
                lookup[tid,row['word_ordinal'],row['kind'],row['unit_index'],tuple(row['evidence']['native_target_phones'])].append(row)
        for q in rows:
            tid=selections[q['key']]['receipts'][0]['row']['text_id']
            candidates=[r for r in lookup[tid,q['word_ordinal'],q['kind'],q['unit_index'],tuple(q['evidence']['native_target_phones'])] if r['voice']!=q['voice']]
            if len(candidates)!=1:
                exclusions.append({'domain':'indic','query':q['key'],'word':q['word'],'reason':'No unique same-text other-voice native variant reference'});continue
            r=candidates[0];eligible=q['secondary_realization_qualified'] and native_supported(r['secondary_features'])
            append(q,r,'indic',r['key'],eligible)
        if len(index)%2 or any((a['label'],b['label'])!=(0,1) or a['word']!=b['word'] or a['qualified']!=b['qualified']
            for a,b in zip(index[::2],index[1::2],strict=True)):raise ValueError('Local matrices require intact adjacent pairs')
        matrix=np.stack(vectors);labels=np.array([r['label'] for r in index],np.int64);q=np.array([r['qualified'] for r in index],bool);inventory=[]
        for domain in ('physical','natural','indic'):
            for part in ('train','development','test'):
                mask=np.array([r['domain']==domain and r['split']==part for r in index]);p=out/('arrays_'+domain+'_'+part+'.npz')
                np.savez_compressed(p,features=matrix[mask],labels=labels[mask],qualified=q[mask])
                inventory.append({'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size,'rows':int(mask.sum()),'qualified_pairs':int(q[mask].sum())//2})
        (out/'index.json').write_text(json.dumps(index,ensure_ascii=False)+'\n');(out/'exclusions.json').write_text(json.dumps(exclusions,ensure_ascii=False)+'\n')
        report={'status':'complete_paired_local_matrices','inventory':inventory,'index_sha256':sha(out/'index.json'),
            'feature_names':FEATURE_NAMES,'pairs':len(index)//2,'qualified_pairs':sum(r['qualified'] for r in index)//2,
            'counts':dict(Counter(r['domain']+'/'+r['split']+'/'+r['kind'] for r in index)),'actual_hindi_error_accuracy':None}
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');status(report['status'],pairs=report['pairs'],qualified_pairs=report['qualified_pairs'])
    except Exception:
        status('failed_explicitly',error=traceback.format_exc());raise


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Reference-conditioned local data preparation failed explicitly');raise
