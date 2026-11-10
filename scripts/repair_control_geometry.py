"""Correct only measured coordinate errors, preserving original evidence/reports."""
from __future__ import annotations

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
from hindi_detector.edit_geometry import map_word
from hindi_detector.intrinsic_vowels import measure
from hindi_detector.paired_local_vowels import vector
from hindi_detector.phone_lattice import VOWELS,units_for_project
from hindi_detector.pronunciation_sites import select_reference_structure,windows
from scripts.extract_corrected_training import read_rows
from scripts.score_swar_sites import load_item

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'reports/dual_geometry_features_v30';base=ROOT/'reports/dual_paired_features_v28';controls=ROOT/'reports/vowel_controls_v19'
    if out.exists():raise FileExistsError(out)
    groups=[ROOT/'reports/swar_corpus_features_v18',ROOT/'reports/swar_phone_crosscheck_v20']
    runtimes=[json.loads((g/'runtime.json').read_text()) for g in groups]
    receipts=[{r['id']:r for r in read_rows(g/'items.jsonl') if r['kind']=='real'} for g in groups];caches=[{},{}]
    def original(key,model):
        if key not in caches[model]:caches[model][key]=load_item(groups[model],receipts[model][key])
        return caches[model][key]
    rows=read_rows(controls/'items.jsonl');intrinsic=read_rows(ROOT/'reports/intrinsic_physical_v24/items.jsonl');audit=[]
    for number,row in enumerate(rows):
        word=original(row['key'],0)['detail']['words'][row['ordinal']];donor=row['native_donor'] if row['label']==0 else row['other_donor'];replacement=donor['end']-donor['start'] if donor else 0
        corrected=map_word(word,row['start'],row['end'],replacement);naive={**word,'end':word['end']+(replacement-(row['end']-row['start']))/16000}
        if max(abs(corrected[k]-naive[k]) for k in ('start','end'))>1e-7:
            audit.append({'source_row_index':number,'word':row['word'],'split':row['split'],'kind':row['kind'],'label':row['label'],
                'naive':{k:naive[k] for k in ('start','end')},'corrected':{k:corrected[k] for k in ('start','end')},
                'edit_start_sample':row['start'],'edit_end_sample':row['end'],'replacement_samples':replacement})
    write_group(out,{'geometry_audit.json':audit,'protocol.json':{'source_data_report_sha256':sha(base/'report.json'),
        'rule':'Map both word boundaries piecewise through actual edit: before unchanged, after shifted, inside affinely mapped to replacement',
        'affected_rows':len(audit),'total_control_rows':len(rows),'new_labels_or_new_audio':False},
        'source_snapshot.json':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'source_utf8':p.read_text()}
            for p in (Path(__file__),ROOT/'hindi_detector/edit_geometry.py')]},
        'Corrected coordinate data group, preserving all old source WAVs/reports. geometry_audit.json '
        'lists every affected source row and exact old/new boundaries; repairs.jsonl remeasured dual '
        'features/evidence from same immutable WAVs; arrays_DOMAIN_PART.npz retain unchanged rows and '
        'replace only affected physical feature rows. index.json same identities/labels; report.json '
        'new array hashes/counts. No raw waveform/weight copies. Current consumer corrected per-cause '
        'training; review2026-11-09, keep compact audit/model pins, NOT on HF/inherited research rights.',
        ['.venv-indicf5/bin/python -m scripts.repair_control_geometry'])
    def status(phase,**kw):
        p=out/'status.json';t=p.with_suffix('.partial');t.write_text(json.dumps({'phase':phase,'pid':os.getpid(),
            'updated_utc':datetime.now(timezone.utc).isoformat(),**kw},indent=2)+'\n');t.replace(p)
    try:
        status('loading_verified_two_phone_probes',affected_rows=len(audit))
        models=[]
        for model,directory in enumerate([ROOT/'models/phoneme_probe',ROOT/'models/phoneme_probe_lv60_v20']):
            for member in runtimes[model]['phone']['files']:
                if sha(directory/member['path'])!=member['sha256']:raise ValueError('Phone checkpoint changed')
            extractor=AutoFeatureExtractor.from_pretrained(directory,local_files_only=True);network=AutoModelForCTC.from_pretrained(directory,local_files_only=True).eval().to('mps')
            if extractor.sampling_rate!=16000 or network.config.pad_token_id!=0:raise ValueError('Probe input/blank contract changed')
            models.append((extractor,network))
        torch.set_num_threads(2);ipa=json.loads((ROOT/'data/swar_phone_plan_v18/ipa_map.json').read_text());lexical={r['word']:r for r in json.loads((ROOT/'data/swar_phone_plan_v18/words.json').read_text())};donors={}
        def donor_phone(donor):
            identity=donor['key'],donor['ordinal']
            if identity not in donors:
                item=original(donor['key'],0);raw=lexical[item['detail']['words'][donor['ordinal']]['word']];variants=[]
                for number,phones in enumerate(raw['nepali_project_variants']):
                    try:units,projection=units_for_project(phones,ipa,runtimes[0]['vocabulary'])
                    except ValueError as error:
                        logging.info('Previously unsupported donor variant skipped: %s',error);continue
                    variants.append({'phones':[p for p in phones if p!='.'],'units':units,'projections':projection,'variant_id':number})
                v,_=select_reference_structure({'variants':variants,'any_vowel':frozenset(runtimes[0]['vocabulary'][t] for t in VOWELS if t in runtimes[0]['vocabulary'])},windows(item,donor['ordinal']));donors[identity]=v['phones']
            return donors[identity][donor['unit_index']]
        repairs={}
        for count,entry in enumerate(audit):
            number=entry['source_row_index'];row=rows[number];old=intrinsic[number]
            if any(row[k]!=old[k] for k in ('key','ordinal','unit_index','kind','label','audio_sha256')):raise ValueError('Source row order changed')
            path=controls/row['audio']
            if sha(path)!=row['audio_sha256']:raise ValueError('Source control waveform changed')
            samples,rate=sf.read(path,dtype='float32')
            if rate!=16000 or samples.ndim!=1:raise ValueError('Source format differs')
            features=[];evidence=[]
            for model,(extractor,network) in enumerate(models):
                vocab=runtimes[model]['vocabulary'];phones=old['evidence']['native_target_phones'];units,projections=units_for_project(phones,ipa,vocab)
                variant={'phones':phones,'units':units,'projections':projections};compiled={'any_vowel':frozenset(vocab[t] for t in VOWELS if t in vocab)};contrast={'kind':row['kind'],'index':row['unit_index']}
                if row['kind']=='vowel_quality':contrast['alternative_unit']=units_for_project([donor_phone(row['other_donor'])],ipa,vocab)[0][0]
                reference=original(row['key'],model);rf,re=measure(compiled,variant,contrast,windows(reference,row['ordinal']))
                with torch.inference_mode():
                    encoded=network(**{k:v.to('mps') for k,v in extractor(samples,sampling_rate=16000,return_tensors='pt').items()});lp=encoded.logits[0].log_softmax(-1).cpu().numpy()
                if not np.isfinite(lp).all():raise ValueError('Nonfinite corrected acoustic frames')
                query={'phone_log_probs':lp,'detail':{'words':[entry['corrected']],'phone_frame_seconds':len(samples)/16000/len(lp)}}
                q,qe=measure(compiled,variant,contrast,windows(query,0));features.append(vector(q,rf,qe,re,row['kind']));evidence.append({'query':qe,'reference':re})
            f=np.r_[features[0],features[1]].astype(np.float32);repairs[number]=f
            record={**entry,'audio_sha256':row['audio_sha256'],'features':f.tolist(),'evidence':evidence}
            with (out/'repairs.jsonl').open('a') as stream:stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
            status('remeasuring_corrected_control_coordinates',completed=count+1,total=len(audit))
        index=json.loads((base/'index.json').read_text());report=json.loads((base/'report.json').read_text());inventory=[];applied=set()
        for domain in ('physical','natural','indic'):
            for part in ('train','development','test'):
                name='arrays_'+domain+'_'+part+'.npz';path=base/name;pin=next(r for r in report['inventory'] if r['path']==name)
                if sha(path)!=pin['sha256']:raise ValueError('Original dual matrix changed')
                with np.load(path,allow_pickle=False) as data:values={k:data[k].copy() for k in data.files}
                subset=[r for r in index if r['domain']==domain and r['split']==part]
                for ordinal,row in enumerate(subset):
                    if domain=='physical' and row['primary_row_index'] in repairs:
                        values['features'][ordinal]=repairs[row['primary_row_index']];applied.add(row['primary_row_index'])
                target=out/name;np.savez_compressed(target,**values);inventory.append({'path':name,'sha256':sha(target),'bytes':target.stat().st_size,'rows':len(subset),'qualified_pairs':int(values['qualified'].sum())//2})
        if applied!=set(repairs):raise ValueError('Not every geometry repair reached its partition')
        (out/'index.json').write_text(json.dumps(index,ensure_ascii=False)+'\n')
        result={'status':'complete_corrected_dual_geometry','inventory':inventory,'index_sha256':sha(out/'index.json'),
            'pairs':len(index)//2,'corrected_control_rows':len(applied),'feature_dimension':114,
            'phone_inventory':runtimes[0]['phone'],'secondary_phone_inventory':runtimes[1]['phone'],'actual_hindi_error_accuracy':None}
        (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');status(result['status'],corrected_rows=len(applied))
    except Exception:
        status('failed_explicitly',error=traceback.format_exc());raise


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Coordinate repair failed explicitly');raise
