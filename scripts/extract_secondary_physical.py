"""Second acoustic measurements of actual known physical interventions."""
from __future__ import annotations

from collections import Counter
from datetime import datetime,timezone
import json
import logging
import os
from pathlib import Path
import traceback

import numpy as np
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor,AutoModelForCTC

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.intrinsic_vowels import measure
from hindi_detector.paired_local_vowels import vector
from hindi_detector.phone_lattice import VOWELS,units_for_project
from hindi_detector.pronunciation_sites import select_reference_structure,windows
from scripts.extract_corrected_training import read_rows
from scripts.score_swar_sites import load_item

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'reports/secondary_physical_v28';controls=ROOT/'reports/vowel_controls_v19';primary=ROOT/'reports/swar_corpus_features_v18';secondary=ROOT/'reports/swar_phone_crosscheck_v20'
    if out.exists():raise FileExistsError(out)
    write_group(out,{'inputs.json':{'physical_items_sha256':sha(controls/'items.jsonl'),
        'primary_intrinsic_items_sha256':sha(ROOT/'reports/intrinsic_physical_v24/items.jsonl'),
        'secondary_runtime_sha256':sha(secondary/'runtime.json'),'labels':'Actual intervention/sham truth, not supplied by the second model'},
        'source_snapshot.json':{'sha256':sha(Path(__file__)),'source_utf8':Path(__file__).read_text()}},
        'Second-backbone57-feature evidence for known physical interventions. items.jsonl exact source row '
        'index/query/ref local features/evidence, arrays.npz numeric features plus original row indices; '
        'exclusions.json explicit whole unsupported pairs, report.json counts/SHA. No waveform or model '
        'copy and no pseudo-labeling of physical truth. Current consumer dual-probe candidate, review '
        '2026-11-09; keep compact evidence through release, NOT on HF, inherited source/MMS research rights.',
        ['.venv-indicf5/bin/python -m scripts.extract_secondary_physical'])
    def status(phase,**kw):
        p=out/'status.json';t=p.with_suffix('.partial');t.write_text(json.dumps({'phase':phase,'pid':os.getpid(),
            'updated_utc':datetime.now(timezone.utc).isoformat(),**kw},indent=2)+'\n');t.replace(p)
    try:
        status('loading_verified_secondary_encoder')
        sr=json.loads((secondary/'runtime.json').read_text());pr=json.loads((primary/'runtime.json').read_text());vocab=sr['vocabulary'];pvocab=pr['vocabulary']
        directory=ROOT/'models/phoneme_probe_lv60_v20'
        for member in sr['phone']['files']:
            if sha(directory/member['path'])!=member['sha256']:raise ValueError('Secondary checkpoint changed')
        extractor=AutoFeatureExtractor.from_pretrained(directory,local_files_only=True);model=AutoModelForCTC.from_pretrained(directory,local_files_only=True).eval().to('mps')
        if extractor.sampling_rate!=16000 or model.config.pad_token_id!=0:raise ValueError('Secondary input/blank contract differs')
        torch.set_num_threads(2);ipa=json.loads((ROOT/'data/swar_phone_plan_v18/ipa_map.json').read_text());lexical={r['word']:r for r in json.loads((ROOT/'data/swar_phone_plan_v18/words.json').read_text())}
        primary_receipts={r['id']:r for r in read_rows(primary/'items.jsonl') if r['kind']=='real'}
        secondary_receipts={r['id']:r for r in read_rows(secondary/'items.jsonl') if r['kind']=='real'}
        pcache={};scache={};donor_cache={};reference_cache={}
        def original_item(key,second):
            cache=scache if second else pcache;group=secondary if second else primary;receipts=secondary_receipts if second else primary_receipts
            if key not in cache:cache[key]=load_item(group,receipts[key])
            return cache[key]
        def donor_phone(donor):
            key=donor['key'];ordinal=donor['ordinal'];identity=(key,ordinal)
            if identity not in donor_cache:
                item=original_item(key,False);raw=lexical[item['detail']['words'][ordinal]['word']];variants=[]
                for number,phones in enumerate(raw['nepali_project_variants']):
                    try:units,projections=units_for_project(phones,ipa,pvocab)
                    except ValueError as error:
                        logging.info('Previously unsupported donor variant skipped: %s',error);continue
                    variants.append({'phones':[p for p in phones if p!='.'],'units':units,'projections':projections,'variant_id':number})
                chosen,_=select_reference_structure({'variants':variants,'any_vowel':frozenset(pvocab[t] for t in VOWELS if t in pvocab)},windows(item,ordinal))
                donor_cache[identity]=chosen['phones']
            return donor_cache[identity][donor['unit_index']]
        old=read_rows(controls/'items.jsonl');intrinsic=read_rows(ROOT/'reports/intrinsic_physical_v24/items.jsonl')
        if len(old)!=len(intrinsic):raise ValueError('Physical row histories differ')
        records=[];vectors=[];excluded=[]
        for pair_number in range(len(old)//2):
            pending=[]
            try:
                for offset in (0,1):
                    position=2*pair_number+offset;row=old[position];qold=intrinsic[position]
                    if any(row[k]!=qold[k] for k in ('key','ordinal','unit_index','kind','label','audio_sha256')):
                        raise ValueError('Physical row order/identity differs')
                    phones=qold['evidence']['native_target_phones'];units,projections=units_for_project(phones,ipa,vocab)
                    variant={'phones':phones,'units':units,'projections':projections};compiled={'any_vowel':frozenset(vocab[t] for t in VOWELS if t in vocab)}
                    contrast={'kind':row['kind'],'index':row['unit_index']}
                    if row['kind']=='vowel_quality':
                        other_unit,_=units_for_project([donor_phone(row['other_donor'])],ipa,vocab);contrast['alternative_unit']=other_unit[0]
                        if other_unit[0]&units[row['unit_index']]:raise ValueError('Secondary model merges the physical vowel categories')
                    identity=(row['key'],row['ordinal'],row['unit_index'],row['kind'])
                    reference=original_item(row['key'],True)
                    if identity not in reference_cache:reference_cache[identity]=measure(compiled,variant,contrast,windows(reference,row['ordinal']))
                    r,re=reference_cache[identity];path=controls/row['audio']
                    if sha(path)!=row['audio_sha256']:raise ValueError('Retained physical bytes changed')
                    samples,rate=sf.read(path,dtype='float32')
                    if rate!=16000 or samples.ndim!=1:raise ValueError('Unexpected physical format')
                    donor=row['native_donor'] if not row['label'] else row['other_donor']
                    delta=(donor['end']-donor['start'] if donor else 0)-(row['end']-row['start'])
                    word={**reference['detail']['words'][row['ordinal']],'end':reference['detail']['words'][row['ordinal']]['end']+delta/16000}
                    with torch.inference_mode():
                        output=model(**{k:v.to('mps') for k,v in extractor(samples,sampling_rate=16000,return_tensors='pt').items()});lp=output.logits[0].log_softmax(-1).cpu().numpy()
                    if not np.isfinite(lp).all():raise ValueError('Nonfinite secondary physical frames')
                    query={'detail':{'words':[word],'phone_frame_seconds':len(samples)/16000/len(lp)},'phone_log_probs':lp}
                    q,qe=measure(compiled,variant,contrast,windows(query,0));f=vector(q,r,qe,re,row['kind'])
                    pending.append((f,{'source_row_index':position,**{k:row[k] for k in ('key','word','ordinal','unit_index','voice','split','kind','label','audio_sha256')},
                        'features':f.tolist(),'query_intrinsic_features':q.tolist(),'reference_intrinsic_features':r.tolist(),
                        'query_evidence':qe,'reference_evidence':re}))
            except ValueError as error:
                logging.warning('Explicit whole secondary physical pair exclusion %s: %s',pair_number,error);excluded.append({'pair_number':pair_number,'reason':str(error)});continue
            for f,record in pending:
                record['row_index']=len(records)
                with (out/'items.jsonl').open('a') as stream:stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                records.append(record);vectors.append(f)
            status('measuring_secondary_physical_controls',pairs=pair_number+1,total_pairs=len(old)//2,available_pairs=len(records)//2)
        np.savez_compressed(out/'arrays.npz',features=np.stack(vectors),source_row_indices=np.array([r['source_row_index'] for r in records],np.int64))
        (out/'exclusions.json').write_text(json.dumps(excluded,indent=2)+'\n')
        report={'status':'complete_secondary_physical','pairs':len(records)//2,'excluded_pairs':len(excluded),
            'arrays_sha256':sha(out/'arrays.npz'),'items_sha256':sha(out/'items.jsonl'),'phone_inventory':sr['phone'],
            'counts':dict(Counter(r['split']+'/'+r['kind'] for r in records))}
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');status(report['status'],pairs=report['pairs'])
    except Exception:
        status('failed_explicitly',error=traceback.format_exc());raise


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Second physical probe failed explicitly');raise
