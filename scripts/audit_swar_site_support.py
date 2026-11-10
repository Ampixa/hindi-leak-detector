"""Audit actual flags for local acoustic support and second-model corroboration."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.pronunciation_sites import compile_word, windows
from hindi_detector.site_validation import POLICY, validate_site
from scripts.extract_corrected_training import read_rows
from scripts.score_swar_sites import load_item

ROOT = Path(__file__).resolve().parents[1]


def key(row):
    return (row['id'],row['source'],row['word_ordinal'],tuple(row['native_phones']),row['site']['kind'],row['site']['index'])


def main():
    out=ROOT/'reports/swar_site_support_v21'
    if out.exists():
        raise FileExistsError(out)
    base=ROOT/'reports/swar_corpus_features_v18';plan=ROOT/'data/swar_phone_plan_v18'
    model=ROOT/'reports/vowel_detector_v19'
    model_report=json.loads((model/'report.json').read_text())
    if sha(model/'model.pt')!=model_report['model_sha256']:
        raise ValueError('Frozen site head changed')
    runtime=json.loads((base/'runtime.json').read_text())
    ipa=json.loads((plan/'ipa_map.json').read_text())
    raw={r['word']:r for r in json.loads((plan/'words.json').read_text())}
    actual=json.loads((model/'actual_swar/cases.json').read_text())
    observed={key(s):s for c in actual for reading in c['analysis'].values() for w in reading['words'] for s in w['sites']}
    primary=read_rows(ROOT/'reports/swar_sites_v18/items.jsonl')
    secondary={key(r):r for r in read_rows(ROOT/'reports/swar_sites_lv60_v20/items.jsonl')}
    receipts={(r['id'],r['kind']):r for r in read_rows(base/'items.jsonl')}
    cache={};compiled={};records=[]
    for row in primary:
        k=key(row);before=observed[k]
        query_key=(row['id'],row['source']);reference_key=(row['id'],'real')
        for item in (query_key,reference_key):
            if item not in cache:cache[item]=load_item(base,receipts[item])
        if row['word'] not in compiled:compiled[row['word']]=compile_word(raw[row['word']],ipa,runtime['vocabulary'])
        c=compiled[row['word']];variant=next(v for v in c['variants'] if v['variant_id']==row['variant_id'])
        contrast=next(s for s in variant['contrasts'] if s['kind']==row['site']['kind'] and s['index']==row['site']['index'])
        support=validate_site(c,variant,contrast,row['reference'],
            windows(cache[query_key],row['word_ordinal']),windows(cache[reference_key],row['word_ordinal']))
        after=before['decision']
        if after=='strong_vowel_change' and support['status']!='supported_local_vowel_diagnosis':
            after='uncertain'
        second=secondary.get(k)
        if second is None:
            corroboration='different_or_unsupported_native_variant'
        else:
            q=[w['alternative_minus_target_per_frame'] for w in second['query']['windows']]
            r=[w['alternative_minus_target_per_frame'] for w in second['reference']['windows']]
            corroboration=('same_direction_at_all_boundaries' if min(q)>0 and max(r)<0 and second['features'][8]>0
                           else 'opposite_query_direction' if max(q)<0 else 'uncertain_or_reference_sensitive')
        record={**{k:row[k] for k in ('id','source','word','word_ordinal','native_phones','site')},
            'before_decision':before['decision'],'after_local_support_decision':after,
            'controlled_vowel_change_score':before['controlled_vowel_change_score'],
            'local_support':support,'second_acoustic_model':corroboration,
            'secondary_features':second['features'] if second else None,'actual_hindi_error_label':None}
        records.append(record)
    flags=[r for r in records if r['before_decision']=='strong_vowel_change']
    retained=[r for r in flags if r['after_local_support_decision']=='strong_vowel_change']
    regressions=[]
    for label in json.loads((plan/'regressions.json').read_text()):
        subset=[r for r in records if r['id']==label['id'] and r['source']==label['source'] and r['word']==label['word']]
        regressions.append({**label,'flag_after_local_support':any(r['after_local_support_decision']=='strong_vowel_change' for r in subset),
                            'second_acoustic_evidence':[r['second_acoustic_model'] for r in subset if r['before_decision']=='strong_vowel_change']})
    summary={'created_utc':datetime.now(timezone.utc).isoformat(),'model_sha256':model_report['model_sha256'],
        'policy':POLICY,'status':'actual_site_support_and_crosscheck_measured_no_refit',
        'observations':len(records),'before_site_flags':dict(Counter(r['source'] for r in flags)),
        'after_local_support_site_flags':dict(Counter(r['source'] for r in retained)),
        'retained_second_model_directions':dict(Counter(r['second_acoustic_model'] for r in retained)),
        'regressions':regressions,'actual_hindi_error_accuracy':None,
        'limits':['Local support floor is inherited from physical nucleus selection, not calibrated native-error correctness.',
                  'Both acoustic checkpoints share the CommonVoice research family and the same MMS word bounds/G2P priors; agreement is corroboration, not independent human gold.',
                  'This audit does not refit the model or alter its original test report. Proposed local-validity guard needs separate evaluation.']}
    write_group(out,{'report.json':summary,'items.json':records,'source_snapshot.json':[
        {'path':str(p.relative_to(ROOT)),'sha256':sha(p),'source_utf8':p.read_text()} for p in (Path(__file__),ROOT/'hindi_detector/site_validation.py')]},
        'Actual flag validity audit. items.json contains every local expected/alternative vowel or adjacent '
        'consonant emission check, proposed guarded decision and second-backbone direction. report.json counts '
        'flag changes and two development regressions. No model refitting or automatic pronunciation gold. '
        'Current consumer morning detector validation; review2026-11-09, retain compact audit until superseded. '
        'No waveform/model copy/HF upload; existing source/MMS research licences.',
        ['.venv-indicf5/bin/python -m scripts.audit_swar_site_support'])
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Local actual-site support audit failed explicitly');raise
