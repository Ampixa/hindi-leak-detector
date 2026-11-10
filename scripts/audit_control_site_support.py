"""Evaluate the fixed local-acoustic validity policy on existing held-out controls."""
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
from hindi_detector.evaluation import binary_metrics
from hindi_detector.phone_lattice import VOWELS, units_for_project
from hindi_detector.pronunciation_sites import evidence_for_site, select_reference_structure, windows
from hindi_detector.site_validation import POLICY, validate_site
from hindi_detector.vowel_detector import decisions, load, score
from hindi_detector.word_evidence import WordEvidenceEngine
from scripts.extract_corrected_training import read_rows
from scripts.score_swar_sites import load_item

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'reports/control_site_support_v21'
    if out.exists():raise FileExistsError(out)
    base=ROOT/'reports/swar_corpus_features_v18';controls=ROOT/'reports/vowel_controls_v19';model=ROOT/'reports/vowel_detector_v19'
    model_report=json.loads((model/'report.json').read_text())
    if sha(model/'model.pt')!=model_report['model_sha256']:raise ValueError('Frozen model changed')
    write_group(out,{'protocol.json':{'model_sha256':model_report['model_sha256'],'policy':POLICY,
        'partitions':['development','test'],'refitting':False,'previously_evaluated_control_set':True},
        'source_snapshot.json':{'sha256':sha(Path(__file__)),'source_utf8':Path(__file__).read_text()}},
        'Fixed local-support guard evaluation on the existing controlled test/development sets. No refit '
        'or threshold adjustment; original neural report remains unchanged. items.jsonl pins control/source '
        'audio and exact before/after decisions plus measured anchors. report.json gives actual intervention '
        'metrics, not real Hindi-error accuracy or a new untouched benchmark. Same selected model and prior '
        'nucleus-selection floor. Current consumer guarded detector decision; review2026-11-09, keep compact '
        'audit/config through final review. No new audio/model copy or HF upload; inherited rights.',
        ['.venv-indicf5/bin/python -m scripts.audit_control_site_support'])

    def status(phase,**extra):
        p=out/'status.json';t=p.with_suffix('.partial');t.write_text(json.dumps({'phase':phase,'pid':os.getpid(),
            'updated_utc':datetime.now(timezone.utc).isoformat(),**extra},indent=2)+'\n');t.replace(p)

    try:
        status('loading_frozen_acoustic_models')
        network,state=load(model/'model.pt');engine=WordEvidenceEngine(ROOT,device='mps',pitch_method='dio',temporal_bins=20)
        runtime=json.loads((base/'runtime.json').read_text());vocab=runtime['vocabulary']
        if engine.phone_inventory!=runtime['phone']:raise ValueError('Control validation acoustic model differs')
        all_vowels=frozenset(vocab[t] for t in VOWELS if t in vocab)
        plan=ROOT/'data/swar_phone_plan_v18';ipa=json.loads((plan/'ipa_map.json').read_text())
        lexical={r['word']:r for r in json.loads((plan/'words.json').read_text())}
        receipts={r['id']:r for r in read_rows(base/'items.jsonl') if r['kind']=='real'}
        cache={};structures={}

        def structure(key,ordinal):
            identity=(key,ordinal)
            if identity in structures:return structures[identity]
            if key not in cache:cache[key]=load_item(base,receipts[key])
            item=cache[key];word=item['detail']['words'][ordinal]['word'];raw=lexical[word]
            variants=[]
            for number,phones in enumerate(raw['nepali_project_variants']):
                try:units,projections=units_for_project(phones,ipa,vocab)
                except ValueError:
                    logging.info('Same documented unsupported native variant skipped: %s/%s',word,number);continue
                variants.append({'variant_id':number,'phones':[p for p in phones if p!='.'],'units':units,'projections':projections})
            compiled={'word':word,'variants':variants,'any_vowel':all_vowels}
            crops=windows(item,ordinal);variant,reduced=select_reference_structure(compiled,crops)
            value=(compiled,variant,reduced,crops,item['detail']['words'][ordinal]);structures[identity]=value;return value

        input_rows=[r for r in read_rows(controls/'items.jsonl') if r['split'] in {'development','test'}]
        outputs=[]
        for number,row in enumerate(input_rows):
            c,v,reduced,reference_crops,original_word=structure(row['key'],row['ordinal'])
            contrast={'kind':row['kind'],'index':row['unit_index']}
            donor=row['native_donor'] if not row['label'] else row['other_donor']
            if row['kind']=='vowel_quality':
                alternative=row['other_donor'];_,variant,_,_,_=structure(alternative['key'],alternative['ordinal'])
                contrast['alternative_unit']=variant['units'][alternative['unit_index']]
            reference=evidence_for_site(c,v,reduced,contrast,reference_crops)
            old=row['reference_evidence']['windows']
            discrepancy=max(abs(a['alternative_minus_target_per_frame']-b['alternative_minus_target_per_frame'])
                            for a,b in zip(reference['windows'],old,strict=True))
            if discrepancy>1e-7:raise ValueError('Reconstructed frozen control hypothesis differs')
            path=controls/row['audio']
            if sha(path)!=row['audio_sha256']:raise ValueError('Retained physical control bytes differ')
            samples,rate=sf.read(path,dtype='float32')
            if rate!=16000 or samples.ndim!=1:raise ValueError('Control input format differs')
            delta=(donor['end']-donor['start'] if donor else 0)-(row['end']-row['start'])
            word={**original_word,'end':original_word['end']+delta/16000}
            with engine.torch.inference_mode():
                encoded=engine.phone(**{k:x.to('mps') for k,x in engine.extractor(samples,sampling_rate=16000,return_tensors='pt').items()})
                lp=encoded.logits[0].log_softmax(-1).cpu().numpy()
            item={'detail':{'words':[word],'phone_frame_seconds':len(samples)/16000/len(lp)},'phone_log_probs':lp}
            support=validate_site(c,v,contrast,reference,windows(item,0),reference_crops)
            vector=np.array(row['features'],np.float32)[None]
            probability=score(network,state,vector)
            before=decisions(state,vector,probability)[0]
            after=before if support['status']=='supported_local_vowel_diagnosis' or before!='strong_vowel_change' else 'uncertain'
            result={**{k:row[k] for k in ('key','word','ordinal','kind','label','split','voice','audio','audio_sha256')},
                'before':before,'after':after,'local_support':support,'score':float(probability[0])}
            outputs.append(result)
            with (out/'items.jsonl').open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
            status('measuring_fixed_control_validity',completed=number+1,total=len(input_rows))
        metrics={}
        for part in ('development','test'):
            metrics[part]={}
            for kind in ('vowel_deletion','vowel_quality'):
                subset=[r for r in outputs if r['split']==part and r['kind']==kind]
                y=np.array([r['label'] for r in subset]);p=np.array([r['after']=='strong_vowel_change' for r in subset],float)
                metrics[part][kind]=binary_metrics(y,p,.5)
        report={'status':'complete_fixed_local_support_audit','model_sha256':model_report['model_sha256'],
            'metrics':metrics,'policy':POLICY,'existing_test_set_reused_for_guard_audit':True,
            'no_training_or_threshold_adjustment':True,'actual_hindi_error_accuracy':None}
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        status('complete_local_support_control_audit',observations=len(outputs))
    except Exception:
        status('failed_explicitly',error=traceback.format_exc());raise


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Fixed local-support control audit failed explicitly');raise
