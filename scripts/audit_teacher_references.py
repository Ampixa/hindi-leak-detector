"""Frozen paired detector with acoustically qualified24-layer references."""
from __future__ import annotations

from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
import logging
from pathlib import Path

import numpy as np

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.pronunciation_sites import compile_word,evidence_for_site,paired_site_features,select_reference_structure,windows
from hindi_detector.vowel_detector import decisions,load,score
from scripts.extract_corrected_training import read_rows
from scripts.score_swar_sites import load_item

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'reports/swar_teacher_consensus_v24'
    if out.exists():raise FileExistsError(out)
    refs=ROOT/'reports/swar_teacher_features_v21';queries=ROOT/'reports/swar_corpus_features_v18';model=ROOT/'reports/vowel_detector_v19'
    for directory,terminal in [(refs,'complete_rendered_phonetic_frames'),(queries,'complete_actual_picks_features')]:
        if json.loads((directory/'status.json').read_text())['phase']!=terminal:raise ValueError('Required input incomplete')
    network,state=load(model/'model.pt');expected=json.loads((model/'report.json').read_text())['model_sha256']
    if sha(model/'model.pt')!=expected:raise ValueError('Frozen paired model changed')
    runtime=json.loads((queries/'runtime.json').read_text());rr=json.loads((refs/'runtime.json').read_text())
    if runtime['phone']!=rr['primary_phone']:raise ValueError('Query/reference acoustic encoders differ')
    ipa=json.loads((ROOT/'data/swar_phone_plan_v18/ipa_map.json').read_text());compiled={};unsupported={}
    for row in json.loads((ROOT/'data/swar_phone_plan_v18/words.json').read_text()):
        try:compiled[row['word']]=compile_word(row,ipa,runtime['vocabulary'])
        except ValueError as error:unsupported[row['word']]=str(error)
    content={r['id']:r for r in read_rows(ROOT/'reports/swar_teacher_content_v21/items.jsonl')}
    reference_rows=read_rows(refs/'items.jsonl');query_rows=read_rows(queries/'items.jsonl')
    query_lookup={(r['id'],r['kind']):load_item(queries,r) for r in query_rows};all_records=[];exclusions=[]
    for number,receipt in enumerate(reference_rows):
        reference=load_item(refs,receipt);key=receipt['text_id'];cq=content[receipt['id']]['content_quality']['decision']
        for ordinal,word in enumerate(reference['detail']['words']):
            name=word['word']
            if name not in compiled:continue
            c=compiled[name]
            try:
                rc=windows(reference,ordinal);variant,reduced=select_reference_structure(c,rc)
                for contrast in variant['contrasts']:
                    re=evidence_for_site(c,variant,reduced,contrast,rc)
                    adequate=(cq!='reject' and re['target_phone_span']['mean_log_emission']>-1.5 and
                        max(w['alternative_minus_target_per_frame'] for w in re['windows'])<0 and
                        min(w['target_score_per_frame'] for w in re['windows'])>-1.5)
                    for source in ('real','swar','a450'):
                        query=query_lookup[key,source]
                        if query['detail']['words'][ordinal]['word']!=name:raise ValueError('Canonical word alignment identity differs')
                        qe=evidence_for_site(c,variant,reduced,contrast,windows(query,ordinal));f=paired_site_features(qe,re,contrast)
                        p=score(network,state,f[None]);decision=decisions(state,f[None],p)[0]
                        all_records.append({'id':key,'source':source,'word':name,'word_ordinal':ordinal,'kind':contrast['kind'],
                            'site_index':contrast['index'],'expected':contrast['expected'],'alternative':contrast.get('alternative'),
                            'reference_id':receipt['id'],'reference_speaker':receipt['speaker_id'],'content_decision':cq,
                            'reference_qualified':bool(adequate),'score':float(p[0]),'decision':decision,'features':f.tolist(),
                            'query_evidence':qe,'reference_evidence':re,'actual_hindi_error_label':None})
            except ValueError as error:
                logging.warning('Explicit teacher site exclusion %s/%s: %s',receipt['id'],name,error)
                exclusions.append({'reference_id':receipt['id'],'word_ordinal':ordinal,'word':name,'reason':str(error)})
        if (number+1)%20==0:logging.info('Teacher references measured %s/%s',number+1,len(reference_rows))
    grouped=defaultdict(list)
    for row in all_records:grouped[tuple(row[k] for k in ('id','source','word_ordinal','kind','site_index','expected','alternative'))].append(row)
    consensus=[]
    for key,records in grouped.items():
        valid=[r for r in records if r['reference_qualified']];ds={r['decision'] for r in valid}
        decision=next(iter(ds)) if len(ds)==1 else 'uncertain'
        if not valid:decision='unsupported_reference'
        consensus.append({**{k:records[0][k] for k in ('id','source','word','word_ordinal','kind','site_index','expected','alternative')},
            'decision':decision,'qualified_references':len(valid),'available_references':len(records),
            'reference_decisions':[{'id':r['reference_id'],'qualified':r['reference_qualified'],'score':r['score'],'decision':r['decision']} for r in records]})
    regressions=[]
    for row in json.loads((ROOT/'data/swar_phone_plan_v18/regressions.json').read_text()):
        matches=[r for r in consensus if r['id']==row['id'] and r['source']==row['source'] and r['word']==row['word']]
        regressions.append({**row,'flagged':any(r['decision']=='strong_vowel_change' for r in matches),'sites':matches})
    counts=Counter(r['source']+'/'+r['decision'] for r in consensus)
    report={'status':'complete_qualified_teacher_reference_audit','created_utc':datetime.now(timezone.utc).isoformat(),
        'paired_model_sha256':expected,'references':len(reference_rows),'site_comparisons':len(all_records),'consensus_sites':len(consensus),
        'reference_policy':'Expected vowel emission >-1.5, Native path fit >-1.5 and favored in all3 crops; no global content reject. Ignore unqualified references; conflicting qualified references =>uncertain.',
        'decisions':dict(counts),'regressions':regressions,'regressions_passed':sum(r['flagged'] for r in regressions),
        'actual_hindi_error_accuracy':None,'limits':['Teacher output is not pronunciation gold; qualification uses the same acoustic probe and shares biases.',
            'Original model calibrated on human-reference physical controls, not this generated-reference deployment.',
            'One qualified reference is explicitly reported; consensus with one reference is weaker than replicated support.',
            'Actual unconfirmed flags remain hypotheses, not annotations.']}
    write_group(out,{'report.json':report,'comparisons.json':all_records,'consensus.json':consensus,
        'exclusions.json':exclusions,'unsupported_words.json':unsupported,'source_snapshot.json':{'sha256':sha(Path(__file__)),'source_utf8':Path(__file__).read_text()}},
        'Automated generated-reference fallback audit on actual86 Swar cases. Comparisons preserve all '
        'reference suitability/evidence/decisions and consensus explicitly abstains on conflicts or no '
        'qualified reference. No model refitting, new model/audio copy or real-error labels. Current consumer '
        'reference-generalization decision, review2026-11-09; retain compact conclusions through model '
        'selection, delete redundant diagnostic comparisons after consumer closes. NOT on HF, inherited rights.',
        ['.venv-indicf5/bin/python -m scripts.audit_teacher_references'])
    print(json.dumps({k:report[k] for k in ('references','site_comparisons','decisions','regressions_passed')}))


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Teacher-reference audit failed explicitly');raise
