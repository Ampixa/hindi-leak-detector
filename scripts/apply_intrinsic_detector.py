"""Apply a frozen development-selected query-only head to actual Swar audio."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime,timezone
import json
import logging
from pathlib import Path

import numpy as np

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.intrinsic_detector import decisions,load,score
from hindi_detector.intrinsic_inference import assign_decisions

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',type=Path);args=parser.parse_args()
    chosen=json.loads((ROOT/'reports/intrinsic_selection_v25/selected.json').read_text())
    model=args.model or ROOT/chosen['model'];out=model.parent/'actual_swar';base=ROOT/'reports/intrinsic_swar_features_v24'
    if out.exists():raise FileExistsError(out)
    network,state=load(model);report=json.loads((base/'report.json').read_text())
    for file,key in [('arrays.npz','array_sha256'),('cases.json','cases_sha256'),('index.json','index_sha256')]:
        if sha(base/file)!=report[key]:raise ValueError('Actual query-only data changed')
    with np.load(base/'arrays.npz',allow_pickle=False) as values:matrix=values['features']
    p=score(network,state,matrix);ds=decisions(state,matrix,p)
    readings=json.loads((base/'cases.json').read_text());lookup={}
    for reading in readings:
        sites=[s for w in reading['words'] for s in w['sites']];indices=np.array([s['row_index'] for s in sites],int)
        assign_decisions(reading['words'],sites,p[indices],[ds[i] for i in indices]);lookup[reading['id'],reading['source']]=reading
    cases=[]
    for row in json.loads((ROOT/'data/swar_speech_v18/selection.json').read_text()):
        analysis={source:lookup[row['id'],source] for source in ('real','swar','a450') if (row['id'],source) in lookup}
        cases.append({**row,'analysis':analysis,'coverage':'scored' if analysis else 'unscored_numeric_transcript_needs_verification'})
    regressions=[]
    for label in json.loads((ROOT/'data/swar_phone_plan_v18/regressions.json').read_text()):
        case=next(c for c in cases if c['id']==label['id']);words=[w for w in case['analysis'][label['source']]['words'] if w['word']==label['word']]
        regressions.append({**label,'flagged':any(w['decision']=='strong_vowel_change' for w in words),
            'sites':[{'kind':s['site']['kind'],'score':s['controlled_vowel_change_score'],'decision':s['decision']} for w in words for s in w['sites']]})
    correct=next(c for c in cases if c['id']=='real_yt_yt04_030')
    correct_words=[w for w in correct['analysis']['swar']['words'] if w['word']=='किनकि']
    checks={'known_user_regressions_passed':sum(r['flagged'] for r in regressions),'known_user_regressions_total':len(regressions),
        'correct_same_word_kinaki030_unflagged':bool(correct_words) and not any(w['decision']=='strong_vowel_change' for w in correct_words)}
    counts=Counter(source+'/'+w['decision'] for c in cases for source,reading in c['analysis'].items() for w in reading['words'])
    site_flags=Counter()
    for c in cases:
        for source,reading in c['analysis'].items():
            for w in reading['words']:
                for s in w['sites']:
                    if s['decision']=='strong_vowel_change':
                        site_flags[source+'/'+s['site']['kind']+'/'+('final' if s['features'][27] else 'internal')]+=1
    summary={'created_utc':datetime.now(timezone.utc).isoformat(),'model_sha256':sha(model),'model':str(model.relative_to(ROOT)),
        'requires_reference_audio':False,'model_kind':'intrinsic_vowel_site_v24','cases_planned':len(cases),
        'cases_scored':sum(bool(c['analysis']) for c in cases),'user_picks_scored':sum(c['user_pick'] and bool(c['analysis']) for c in cases),
        'regressions':regressions,'regressions_passed':sum(r['flagged'] for r in regressions),'promotion_checks':checks,
        'word_decisions':dict(counts),'strong_site_kinds':dict(site_flags),'actual_hindi_error_accuracy':None,
        'limits':['Only two user-confirmed real errors; other flags remain candidates.',
            'Query-only Native G2P expectations can conflict with legitimate regional or colloquial reductions.',
            'Secondary acoustic corroboration shares corpus-family/alignment/G2P biases.',
            'No Hindi-specific intonation accuracy established.']}
    write_group(out,{'report.json':summary,'cases.json':cases,'source_snapshot.json':{'sha256':sha(Path(__file__)),'source_utf8':Path(__file__).read_text()}},
        'Frozen query-only actual Swar application. cases.json preserves every original source/word interval '
        'and primary/secondary phonetic evidence, scores and explicit unsupported words. report.json records '
        'the two development regressions, correct same-word smoke and source flag counts; observed real '
        'speech flag counts are not native-error FPR. No reference audio used in features; no new raw '
        'audio/model copies. Current consumer promotion decision, review2026-11-09, retain compact results '
        'through current release; NOT on HF/inherited source/MMS research licence.',
        ['.venv-indicf5/bin/python -m scripts.apply_intrinsic_detector [--model OWNED_CHECKPOINT]'])
    print(json.dumps({'checks':checks,'word_decisions':dict(counts),'strong_site_kinds':dict(site_flags),'regressions':regressions},ensure_ascii=False))


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Query-only actual application failed explicitly');raise
