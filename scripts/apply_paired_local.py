"""Measure the frozen stronger paired model on actual human-referenced Swar."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime,timezone
import json
import logging
from pathlib import Path

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.dual_paired_detector import load as dual_load,score as dual_score,decisions as dual_decisions
from hindi_detector.dual_paired_inference import analyze_pair as dual_pair
from hindi_detector.intrinsic_inference import assign_decisions
from hindi_detector.paired_local_detector import decisions,load,score
from hindi_detector.paired_local_inference import analyze_pair
from hindi_detector.task_vowel_detector import load as task_load,score as task_score,decisions as task_decisions
from scripts.extract_corrected_training import read_rows
from scripts.score_swar_sites import load_item

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group();mode.add_argument('--dual',action='store_true');mode.add_argument('--task',action='store_true')
    parser.add_argument('--model-directory',type=Path);args=parser.parse_args()
    dual=args.dual or args.task
    model=ROOT/('reports/task_vowel_detector_v30' if args.task else 'reports/dual_paired_detector_v28' if args.dual else 'reports/paired_local_detector_v27')
    if args.model_directory is not None:
        if not args.task:raise ValueError('Explicit release directory requires the verified task schema')
        model=args.model_directory.resolve(strict=True)
        if not model.is_relative_to(ROOT/'reports'):raise ValueError('Release model must be in the owned reports tree')
    base=ROOT/'reports/swar_corpus_features_v18';out=model/'actual_swar'
    load_fn=dual_load if args.dual else load;score_fn=dual_score if args.dual else score;decide=dual_decisions if args.dual else decisions
    if args.task:load_fn,score_fn,decide=task_load,task_score,task_decisions
    if out.exists():raise FileExistsError(out)
    expected=json.loads((model/'report.json').read_text())['model_sha256']
    if sha(model/'model.pt')!=expected:raise ValueError('Paired local model identity changed')
    network,state=load_fn(model/'model.pt')
    for path,digest in state['code_sha256'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Frozen paired local dependency changed')
    runtime=json.loads((base/'runtime.json').read_text());ipa=json.loads((ROOT/'data/swar_phone_plan_v18/ipa_map.json').read_text())
    lexical={r['word']:r for r in json.loads((ROOT/'data/swar_phone_plan_v18/words.json').read_text())}
    if runtime['phone']!=state['phone_inventory']:raise ValueError('Actual paired phone encoder differs')
    receipts={(r['id'],r['kind']):r for r in read_rows(base/'items.jsonl')};cache={k:load_item(base,v) for k,v in receipts.items()};cases=[]
    secondary_cache={};secondary_runtime=None
    if dual:
        second=ROOT/'reports/swar_phone_crosscheck_v20';secondary_runtime=json.loads((second/'runtime.json').read_text())
        if secondary_runtime['phone']!=state['secondary_phone_inventory']:raise ValueError('Secondary actual encoder differs')
        secondary_cache={(r['id'],r['kind']):load_item(second,r) for r in read_rows(second/'items.jsonl')}
    for number,row in enumerate(json.loads((ROOT/'data/swar_speech_v18/selection.json').read_text())):
        analysis={}
        if (row['id'],'real') in cache:
            reference=cache[row['id'],'real']
            for source in ('real','swar','a450'):
                query=cache[row['id'],source]
                if dual:
                    words,sites,x=dual_pair(query,reference,secondary_cache[row['id'],source],secondary_cache[row['id'],'real'],
                        lexical.__getitem__,ipa,runtime['vocabulary'],secondary_runtime['vocabulary'])
                else:words,sites,x=analyze_pair(query,reference,lexical.__getitem__,ipa,runtime['vocabulary'])
                if len(x):
                    p=score_fn(network,state,x);assign_decisions(words,sites,p,decide(state,x,p))
                else:assign_decisions(words,[],[],[])
                analysis[source]={'words':words,'audio':query['receipt']['path'],'audio_sha256':query['receipt']['sha256'],
                    'seconds':query['detail']['seconds'],'tracks':query['detail']['tracks']}
        cases.append({**row,'analysis':analysis,'coverage':'scored' if analysis else 'unscored_numeric_transcript_needs_verification'})
        if (number+1)%20==0:logging.info('Actual local paired cases %s/96',number+1)
    regressions=[]
    for label in json.loads((ROOT/'data/swar_phone_plan_v18/regressions.json').read_text()):
        case=next(c for c in cases if c['id']==label['id']);words=[w for w in case['analysis'][label['source']]['words'] if w['word']==label['word']]
        regressions.append({**label,'flagged':any(w['decision']=='strong_vowel_change' for w in words),
            'site_decisions':[{'kind':s['site']['kind'],'score':s['controlled_vowel_change_score'],'decision':s['decision'],'relative_margin':s['features'][8]} for w in words for s in w['sites']]})
    correct=next(c for c in cases if c['id']=='real_yt_yt04_030');cw=[w for w in correct['analysis']['swar']['words'] if w['word']=='किनकि']
    checks={'known_user_regressions_passed':sum(r['flagged'] for r in regressions),'known_user_regressions_total':len(regressions),
        'correct_same_word_kinaki030_unflagged':bool(cw) and not any(w['decision']=='strong_vowel_change' for w in cw)}
    counts=Counter(source+'/'+w['decision'] for c in cases for source,r in c['analysis'].items() for w in r['words'])
    summary={'created_utc':datetime.now(timezone.utc).isoformat(),'model_sha256':expected,'model_kind':state['schema'],
        'cases_planned':len(cases),'cases_scored':sum(bool(c['analysis']) for c in cases),'user_picks_scored':sum(c['user_pick'] and bool(c['analysis']) for c in cases),
        'regressions':regressions,'regressions_passed':sum(r['flagged'] for r in regressions),'promotion_checks':checks,
        'word_decisions':dict(counts),'actual_hindi_error_accuracy':None,'limits':[
            'Two user examples are development regressions, not independent real-error accuracy.',
            'Other flags remain candidates; same-text human references do not define all native varieties.',
            'No flag does not prove correctness; unsupported sites and numeric transcripts abstain.',
            'Pitch contours are diagnostic; Hindi-specific intonation accuracy is not established.']}
    write_group(out,{'report.json':summary,'cases.json':cases,'source_snapshot.json':{'sha256':sha(Path(__file__)),'source_utf8':Path(__file__).read_text()}},
        'Frozen stronger paired model applied to actual Swar corpus after held-out evaluation. cases.json '
        'has all exact source/word intervals/local phonetic evidence and decisions; report.json two user '
        'regressions/correct same-word check/coverage, no invented labels. No new waveform/encoder copy. '
        'Current consumer morning release decision, review2026-11-09; keep compact conclusions while '
        'selected model is active. NOT on HF; inherited source and MMS research licence.',
        ['.venv-indicf5/bin/python -m scripts.apply_paired_local'])
    print(json.dumps({'checks':checks,'counts':dict(counts),'regressions':regressions},ensure_ascii=False))


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Actual paired local application failed explicitly');raise
