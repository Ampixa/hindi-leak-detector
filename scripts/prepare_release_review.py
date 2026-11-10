"""Compact readable-review payload and auditable per-site playback additions."""
from __future__ import annotations

from datetime import datetime,timezone
import json
import logging
import math
from pathlib import Path
from statistics import median

from hindi_detector.corrected_synthesis import sha
from scripts.extract_corrected_training import read_rows

ROOT=Path(__file__).resolve().parents[1]


def interval(evidence,dt,kind,reference=False):
    if reference or kind=='vowel_quality':
        span=evidence['native_site_span' if reference else 'alternative_site_span']
        if span is None:return None
        origin=evidence['windows'][1]['start_frame'];start=origin+span['start_frame'];end=origin+span['end_frame']
    else:
        if not evidence['gap']['available']:return None
        start=evidence['gap']['start_frame'];end=evidence['gap']['end_frame']
    return {'start':start*dt,'end':end*dt,'meaning':'Estimated phone evidence interval, not a human-verified phoneme boundary'}


def wilson(successes,total):
    if not total:return None
    z=1.959963984540054;p=successes/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den;half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [max(0,center-half),min(1,center+half)]


def main():
    model=ROOT/'reports/task_vowel_release_v31';report=model/'review_report.json';addons=model/'review_additions.json'
    if report.exists() or addons.exists():raise FileExistsError('Existing review payload')
    final=json.loads((model/'report.json').read_text());actual=json.loads((model/'actual_swar/report.json').read_text())
    fresh=json.loads((ROOT/'reports/fresh_gap_audit_v32/report.json').read_text());short=json.loads((ROOT/'reports/fresh_waveform_audit_v31/report.json').read_text())
    cold=json.loads((ROOT/'reports/release_cold_verification_v31/report.json').read_text());voice_path=ROOT/'reports/release_voice_audit_v34/report.json'
    voice=json.loads(voice_path.read_text()) if voice_path.exists() else None
    if sha(model/'model.pt')!=final['model_sha256'] or actual['model_sha256']!=final['model_sha256'] or cold['checks_passed']!=cold['checks_total']:
        raise ValueError('Release identity/cold verification not ready')
    if actual['promotion_checks']['known_user_regressions_passed']!=2 or not actual['promotion_checks']['correct_same_word_kinaki030_unflagged']:
        raise ValueError('Mandatory release regressions failed')
    test={kind:{'final_gated_decision':metrics,'recall_interval_95':wilson(metrics['tp'],metrics['positive_labels']),
        'sham_flag_interval_95':wilson(metrics['fp'],metrics['negative_labels'])} for kind,metrics in fresh['metrics']['release'].items()}
    view={'model_sha256':final['model_sha256'],'architecture':'Two task heads (MLP16), with two frozen phone probes',
        'model_kind':'task_vowel_v30','control_pairs':{'shared_training_corpus':418,'development':147},
        'training_description':'Deletion head:238 training pairs. Frozen quality head:418 combined-kind training pairs. Development corpus:147 pairs; kind-specific calibration. Words हैन and किनकि excluded from fitting; their development regressions used for release rollback.',
        'test':test,'evaluation_description':f"Fresh full-region waveform edits: {fresh['pairs']} same-voice pairs, {fresh['word_families']} new word families and two held-out conditioning groups. Source/donor nuclei selected for clear phone-probe evidence; this is not real Hindi-error accuracy.",
        'short_edit_diagnostic':short['metrics']['release'],'cold_checks':{'passed':cold['checks_passed'],'total':cold['checks_total']},
        'voice_audit':voice,'retrospective_model_report':final,'actual_hindi_error_accuracy':None,
        'limitations':['Only two user-confirmed real errors; other flags are candidate Hindi-compatible changes.',
            'Quality edit evaluation has only7 positive/sham pairs and cannot establish reliable general quality-error performance.',
            'Observed zero sham flags in46 selected controls does not establish zero Native-error false positives.',
            'Short nucleus edits often preserve much of the inter-consonant region and were detected less often; both audits are retained.',
            'Reference style, G2P and shared acoustic-probe biases constrain inference; unsupported sites abstain.',
            'Prosody is shown as relative pitch/duration evidence. Hindi-specific intonation accuracy is unvalidated.']}
    region_maps=[]
    for name in ('fresh_waveform_audit_v31','fresh_gap_audit_v32'):
        region_maps.append({(r['job_id'],r['kind'],r['label']):r for r in read_rows(ROOT/'reports'/name/'items.jsonl')})
    matched=set(region_maps[0])&set(region_maps[1]);comparison={}
    for kind in ('vowel_deletion','vowel_quality'):
        keys=sorted(k for k in matched if k[1]==kind);positives=[k for k in keys if k[2]==1]
        if not positives:continue
        durations=[];conditions=[]
        for data in region_maps:
            durations.append(median((data[k]['geometry']['end_sample']-data[k]['geometry']['start_sample'])/16 for k in positives))
            conditions.append({'known_edits':len(positives),'edits_flagged':sum(data[k]['release']['decision']=='strong_vowel_change' for k in positives),
                'shams':len(keys)-len(positives),'shams_flagged':sum(data[k]['release']['decision']=='strong_vowel_change' for k in keys if k[2]==0)})
        comparison[kind]={'matched_pairs':len(positives),'short_nucleus':conditions[0],'full_region':conditions[1],
            'median_cut_milliseconds':{'short_nucleus':durations[0],'full_region':durations[1]},
            'median_fraction_of_full_region_cut_by_nucleus_edit':median((region_maps[0][k]['geometry']['end_sample']-region_maps[0][k]['geometry']['start_sample'])/(region_maps[1][k]['geometry']['end_sample']-region_maps[1][k]['geometry']['start_sample']) for k in positives)}
    view['matched_region_audit']={'model_unchanged_between_conditions':True,'metrics':comparison,
        'meaning':'Matched source readings, different edit extent; CTC unit peaks omit blank-labeled frames between consonant peaks. This is an edit-detection diagnostic, not human phoneme-absence gold.'}
    cases=json.loads((model/'actual_swar/cases.json').read_text());ipa=json.loads((ROOT/'data/swar_phone_plan_v18/ipa_map.json').read_text())
    primary=ROOT/'reports/swar_corpus_features_v18';secondary=ROOT/'reports/swar_phone_crosscheck_v20'
    primary_rows={(r['id'],r['kind']):r for r in read_rows(primary/'items.jsonl')};secondary_rows={(r['id'],r['kind']):r for r in read_rows(secondary/'items.jsonl')};additions={}
    for case in cases:
        if not case['analysis']:continue
        reference_receipt=primary_rows[case['id'],'real'];rp=primary/reference_receipt['detail']
        if sha(rp)!=reference_receipt['detail_sha256']:raise ValueError('Review reference detail changed')
        reference_detail=json.loads(rp.read_text());addition={}
        for source,reading in case['analysis'].items():
            receipt=primary_rows[case['id'],source];path=primary/receipt['detail'];second_receipt=secondary_rows[case['id'],source];sp=secondary/second_receipt['detail']
            if sha(path)!=receipt['detail_sha256'] or sha(sp)!=second_receipt['detail_sha256']:raise ValueError('Review measured detail changed')
            detail=json.loads(path.read_text());other=json.loads(sp.read_text());words=[]
            for ordinal,word in enumerate(reading['words']):
                if other['words'][ordinal]['word']!=word['word']:raise ValueError('Review probe word identity differs')
                sites=[]
                for site in word['sites']:
                    sites.append({'kind':site['site']['kind'],'index':site['site']['index'],'native_phones':site['native_phones'],
                        'native_ipa_sequence':[ipa[p] for p in site['native_phones']],
                        'query_focus':interval(site['query'],detail['phone_frame_seconds'],site['site']['kind']),
                        'reference_focus':interval(site['reference'],reference_detail['phone_frame_seconds'],site['site']['kind'],True)})
                words.append({'secondary_phone_probe_sequence':other['words'][ordinal]['phone_probe_sequence'],'sites':sites})
            addition[source]=words
        additions[case['id']]=addition
    report.write_text(json.dumps(view,ensure_ascii=False,indent=2)+'\n');addons.write_text(json.dumps(additions,ensure_ascii=False)+'\n')
    pins=[model/'actual_swar/report.json',ROOT/'reports/fresh_gap_audit_v32/report.json',ROOT/'reports/fresh_waveform_audit_v31/report.json',ROOT/'reports/release_cold_verification_v31/report.json']
    pins.extend(ROOT/'reports'/name/'items.jsonl' for name in ('fresh_waveform_audit_v31','fresh_gap_audit_v32'))
    if voice_path.exists():pins.append(voice_path)
    provenance={'created_utc':datetime.now(timezone.utc).isoformat(),'creator':'Codex GPT-6 /root','owner':'Ampixa/user','status':'active',
        'review_utc':'2026-11-09','next_review':'2026-11-10','consumer':'Private actual Swar word review',
        'inputs':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in pins],
        'files':[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p),'role':'Compact UI metrics' if p==report else 'Derived IPA/probe/phone-focus additions; no raw waveform copy'} for p in (report,addons)],
        'source_sha256':sha(Path(__file__)),'source_utf8':Path(__file__).read_text(),'not_on_HF':True,
        'retention':'Keep compact review while current model is active; remove superseded additions after consumer closes. Inherited source/MMS rights.'}
    (model/'review_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    with (model/'README.md').open('a') as stream:stream.write('\nreview_report.json / review_additions.json / review_provenance.json: current private review metrics and compact measured IPA/phone-focus additions, exact sizes/input/member hashes in review_provenance.json. No audio duplicate. Review2026-11-09; retain while this model is current.\n')
    print(json.dumps({'model_sha256':final['model_sha256'],'fresh_pairs':fresh['pairs'],'cold_checks':cold['checks_passed'],'review_addition_bytes':addons.stat().st_size}))


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Release review preparation failed explicitly');raise
