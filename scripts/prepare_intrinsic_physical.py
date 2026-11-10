"""Re-encode actual known physical interventions into query-only evidence."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import traceback

import numpy as np
import soundfile as sf

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.intrinsic_vowels import FEATURE_NAMES, measure
from hindi_detector.phone_lattice import VOWELS, units_for_project
from hindi_detector.pronunciation_sites import select_reference_structure, windows
from hindi_detector.word_evidence import WordEvidenceEngine
from scripts.extract_corrected_training import read_rows
from scripts.score_swar_sites import load_item

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'reports/intrinsic_physical_v24';base=ROOT/'reports/swar_corpus_features_v18';controls=ROOT/'reports/vowel_controls_v19'
    if out.exists():raise FileExistsError(out)
    write_group(out,{'protocol.json':{'features':FEATURE_NAMES,'source_items_sha256':sha(controls/'items.jsonl'),
        'labels':'Known same-speaker physical vowel removal/replacement and sham; existing previously evaluated split',
        'test_policy':'Isolated test arrays are read by trainer only after freezing; old test is retrospective'},
        'source_snapshot.json':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'source_utf8':p.read_text()}
            for p in (Path(__file__),ROOT/'hindi_detector/intrinsic_vowels.py')]},
        'Query-only32-feature data from actual physically edited recordings. items.jsonl pins actual WAVs '
        'and evidence; arrays_train/development/test.npz isolate partitions; index.json preserves complete '
        'pairs; report.json inventories bytes/SHA/exclusions. No raw frame/audio/weight duplicate. Current '
        'consumer query-only candidate training; review2026-11-09, retain compact inputs/results through '
        'model review. NOT on HF; inherited source rights/MMS research licence. Physical artifacts can '
        'bias detection; natural pairs are evaluated separately.',
        ['.venv-indicf5/bin/python -m scripts.prepare_intrinsic_physical'])
    def status(phase,**kw):
        p=out/'status.json';t=p.with_suffix('.partial');t.write_text(json.dumps({'phase':phase,'pid':os.getpid(),
            'updated_utc':datetime.now(timezone.utc).isoformat(),**kw},indent=2)+'\n');t.replace(p)
    try:
        status('loading_owned_acoustic_encoder')
        engine=WordEvidenceEngine(ROOT,device='mps',pitch_method='dio',temporal_bins=20)
        runtime=json.loads((base/'runtime.json').read_text());vocab=runtime['vocabulary']
        if engine.phone_inventory!=runtime['phone']:raise ValueError('Physical source encoder changed')
        ipa=json.loads((ROOT/'data/swar_phone_plan_v18/ipa_map.json').read_text())
        lexical={r['word']:r for r in json.loads((ROOT/'data/swar_phone_plan_v18/words.json').read_text())}
        receipts={r['id']:r for r in read_rows(base/'items.jsonl') if r['kind']=='real'}
        cache={};structures={}
        def structure(key,ordinal):
            if (key,ordinal) in structures:return structures[key,ordinal]
            if key not in cache:cache[key]=load_item(base,receipts[key])
            item=cache[key];raw=lexical[item['detail']['words'][ordinal]['word']];variants=[]
            for number,phones in enumerate(raw['nepali_project_variants']):
                try:units,projection=units_for_project(phones,ipa,vocab)
                except ValueError as error:
                    logging.info('Documented unsupported native variant %s/%s: %s',raw['word'],number,error);continue
                variants.append({'variant_id':number,'phones':[p for p in phones if p!='.'],'units':units,'projections':projection})
            compiled={'variants':variants,'any_vowel':frozenset(vocab[t] for t in VOWELS if t in vocab)}
            chosen,_=select_reference_structure(compiled,windows(item,ordinal))
            value=compiled,chosen,item['detail']['words'][ordinal];structures[key,ordinal]=value;return value
        rows=read_rows(controls/'items.jsonl');vectors=[];index=[];excluded=[]
        for pair_number in range(len(rows)//2):
            pair=rows[2*pair_number:2*pair_number+2];output=[]
            if [r['label'] for r in pair]!=[0,1]:raise ValueError('Physical pair order differs')
            try:
                for row in pair:
                    c,v,original=structure(row['key'],row['ordinal']);contrast={'kind':row['kind'],'index':row['unit_index']}
                    donor=row['native_donor'] if not row['label'] else row['other_donor']
                    if row['kind']=='vowel_quality':
                        other=row['other_donor'];_,ov,_=structure(other['key'],other['ordinal'])
                        contrast['alternative_unit']=ov['units'][other['unit_index']]
                    path=controls/row['audio']
                    if sha(path)!=row['audio_sha256']:raise ValueError('Physical WAV bytes changed')
                    samples,rate=sf.read(path,dtype='float32')
                    if rate!=16000 or samples.ndim!=1:raise ValueError('Physical WAV format differs')
                    delta=(donor['end']-donor['start'] if donor else 0)-(row['end']-row['start'])
                    word={**original,'end':original['end']+delta/16000}
                    with engine.torch.inference_mode():
                        result=engine.phone(**{k:x.to('mps') for k,x in engine.extractor(samples,sampling_rate=16000,return_tensors='pt').items()})
                        lp=result.logits[0].log_softmax(-1).cpu().numpy()
                    if not np.isfinite(lp).all():raise ValueError('Nonfinite physical probe frames')
                    item={'detail':{'words':[word],'phone_frame_seconds':len(samples)/16000/len(lp)},'phone_log_probs':lp}
                    values,evidence=measure(c,v,contrast,windows(item,0))
                    output.append((values,{**{k:row[k] for k in ('key','word','ordinal','unit_index','voice','split','kind','label','audio','audio_sha256')},
                        'features':values.tolist(),'evidence':evidence}))
            except ValueError as error:
                logging.warning('Explicit physical whole-pair exclusion %s: %s',pair_number,error)
                excluded.append({'pair_number':pair_number,'reason':str(error)});continue
            for values,record in output:
                record['row_index']=len(index)
                with (out/'items.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                vectors.append(values);index.append({k:record[k] for k in ('row_index','key','word','ordinal','unit_index','voice','split','kind','label')})
            status('encoding_physical_query_only_evidence',pairs=pair_number+1,total_pairs=len(rows)//2,rows=len(index))
        matrix=np.stack(vectors);labels=np.array([r['label'] for r in index],np.int64);inventory=[]
        for part in ('train','development','test'):
            mask=np.array([r['split']==part for r in index]);p=out/('arrays_'+part+'.npz')
            np.savez_compressed(p,features=matrix[mask],labels=labels[mask],qualified=np.ones(int(mask.sum()),bool))
            inventory.append({'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size,'rows':int(mask.sum())})
        (out/'index.json').write_text(json.dumps(index,ensure_ascii=False)+'\n');(out/'exclusions.json').write_text(json.dumps(excluded,indent=2)+'\n')
        report={'status':'complete_physical_intrinsic_matrices','inventory':inventory,'feature_names':FEATURE_NAMES,
            'index_sha256':sha(out/'index.json'),'source_runtime_sha256':sha(base/'runtime.json'),
            'phone_inventory':engine.phone_inventory,'pairs':len(index)//2,'excluded_pairs':len(excluded),
            'counts':dict(Counter(r['split']+'/'+r['kind'] for r in index)),'actual_hindi_error_accuracy':None}
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');status(report['status'],pairs=report['pairs'])
    except Exception:
        status('failed_explicitly',error=traceback.format_exc());raise


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Physical query-only preparation failed explicitly');raise
