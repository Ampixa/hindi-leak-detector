"""Observed human-to-human pronunciation variation; no native-error gold inferred."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.pronunciation_sites import (compile_word, evidence_for_site, paired_site_features,
    select_reference_structure, windows)
from hindi_detector.site_validation import POLICY, validate_site
from hindi_detector.vowel_detector import decisions, load, score
from scripts.extract_corrected_training import read_rows
from scripts.score_swar_sites import load_item

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'reports/native_word_variation_v22'
    if out.exists():raise FileExistsError(out)
    base=ROOT/'reports/swar_corpus_features_v18';plan=ROOT/'data/swar_phone_plan_v18';model=ROOT/'reports/vowel_detector_v19'
    neural_report=json.loads((model/'report.json').read_text())
    if sha(model/'model.pt')!=neural_report['model_sha256']:raise ValueError('Selected head changed')
    network,state=load(model/'model.pt');runtime=json.loads((base/'runtime.json').read_text())
    ipa=json.loads((plan/'ipa_map.json').read_text());raw={r['word']:r for r in json.loads((plan/'words.json').read_text())}
    selection={r['id']:r for r in json.loads((ROOT/'data/swar_speech_v18/selection.json').read_text())}
    receipts={r['id']:r for r in read_rows(base/'items.jsonl') if r['kind']=='real'}
    cache={key:load_item(base,r) for key,r in receipts.items()}
    occurrences=defaultdict(list)
    for key,item in cache.items():
        for ordinal,word in enumerate(item['detail']['words']):
            occurrences[word['word']].append((key,ordinal))
    compiled={};unsupported={}
    for word in occurrences:
        if word not in raw:continue
        try:compiled[word]=compile_word(raw[word],ipa,runtime['vocabulary'])
        except ValueError as error:unsupported[word]=str(error)
    records=[];excluded=[]
    for word,places in occurrences.items():
        if word not in compiled or len(places)<2:continue
        c=compiled[word]
        for query_key,query_ordinal in places:
            voice=selection[query_key]['voice']
            if voice not in {'yt01','yt10','yt12'}:continue
            candidates=[p for p in places if selection[p[0]]['voice']!=voice]
            candidates.sort(key=lambda p:hashlib.sha256((query_key+':'+str(query_ordinal)+':'+p[0]+':'+str(p[1])).encode()).hexdigest())
            chosen=[];voices=set()
            for p in candidates:
                if selection[p[0]]['voice'] not in voices:
                    chosen.append(p);voices.add(selection[p[0]]['voice'])
                if len(chosen)==3:break
            for reference_key,reference_ordinal in chosen:
                try:
                    rcrops=windows(cache[reference_key],reference_ordinal)
                    variant,reduced=select_reference_structure(c,rcrops)
                    for contrast in variant['contrasts']:
                        qcrop=windows(cache[query_key],query_ordinal)
                        re=evidence_for_site(c,variant,reduced,contrast,rcrops)
                        qe=evidence_for_site(c,variant,reduced,contrast,qcrop)
                        vector=paired_site_features(qe,re,contrast)[None]
                        probability=score(network,state,vector);before=decisions(state,vector,probability)[0]
                        support=validate_site(c,variant,contrast,re,qcrop,rcrops)
                        after=before if before!='strong_vowel_change' or support['status']=='supported_local_vowel_diagnosis' else 'uncertain'
                        records.append({'word':word,'kind':contrast['kind'],'index':contrast['index'],
                            'query_case':query_key,'query_ordinal':query_ordinal,'query_voice':voice,
                            'reference_case':reference_key,'reference_ordinal':reference_ordinal,
                            'reference_voice':selection[reference_key]['voice'],'score':float(probability[0]),
                            'before':before,'after':after,'features':vector[0].tolist(),'local_support':support,
                            'actual_hindi_error_label':None})
                except ValueError as error:
                    logging.warning('Explicit unrepresentable native comparison %s/%s/%s: %s',word,query_key,reference_key,error)
                    excluded.append({'word':word,'query':query_key,'reference':reference_key,'reason':str(error)})
    before=[r for r in records if r['before']=='strong_vowel_change'];after=[r for r in records if r['after']=='strong_vowel_change']
    pairs={(r['query_case'],r['query_ordinal'],r['reference_case'],r['reference_ordinal']) for r in records}
    by_word={word:{'site_comparisons':sum(r['word']==word for r in records),
        'raw_site_flags':sum(r['word']==word for r in before),'locally_supported_site_flags':sum(r['word']==word for r in after),
        'actual_reference_recordings':len({key for key,_ in occurrences[word]}),
        'actual_reference_voice_groups':len({selection[key]['voice'] for key,_ in occurrences[word]})} for word in sorted({r['word'] for r in records})}
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'model_sha256':neural_report['model_sha256'],
        'status':'recorded_native_word_variation_measured_not_error_FPR','word_comparison_pairs':len(pairs),
        'site_comparisons':len(records),'word_families':len(by_word),'query_voice_groups':['yt01','yt10','yt12'],
        'raw_site_flags':len(before),'locally_supported_site_flags':len(after),
        'locally_supported_site_flag_fraction':len(after)/len(records) if records else None,
        'by_word':by_word,'guard_policy':POLICY,'actual_hindi_error_accuracy':None,
        'limits':['Human recordings are observed regional/colloquial variation, not independently labeled acceptable/error realizations.',
                  'Flag rates are paired style/phonetic differences, not a human pronunciation-error false-positive rate.',
                  'Same word can differ legitimately across contexts; source speakers/residence are not independently audited here.',
                  'No refitting or threshold tuning on these observations.']}
    write_group(out,{'report.json':result,'comparisons.json':records,'exclusions.json':excluded,
        'unsupported_words.json':unsupported,'source_snapshot.json':{'sha256':sha(Path(__file__)),'source_utf8':Path(__file__).read_text()}},
        'Actual recorded human-to-human same-word variation across different source voice groups. Query '
        'voices were held out from neural head fitting; up to three deterministic other-voice references per '
        'occurrence. No source audio copy or new labels. comparisons.json contains every raw/guarded '
        'decision and local acoustic evidence; report.json reports variation counts, explicitly not error '
        'false-positive accuracy. Current consumer native-variation-aware interpretation; review2026-11-09. '
        'No HF upload; inherited source/research licences. Keep compact conclusions through current model review.',
        ['.venv-indicf5/bin/python -m scripts.audit_native_word_variation'])
    print(json.dumps({k:result[k] for k in ('site_comparisons','word_families','raw_site_flags','locally_supported_site_flags')}))


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Recorded native word-variation audit failed explicitly');raise
