"""Apply verified expanded-case results and failed-prosody admission transparently."""
from __future__ import annotations

from collections import Counter
import copy
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.dual_g2p import DualG2P

ROOT = Path(__file__).resolve().parents[1]


def key(site):
    return site['site']['kind'], site['site']['index'], tuple(site['native_phones'])


def main():
    release = ROOT / 'reports/task_vowel_release_v31'
    expanded = ROOT / 'reports/numeric_swar_expansions_v54'
    out = ROOT / 'reports/review_amendment_v56'
    if out.exists():
        raise FileExistsError(out)
    completed = json.loads((expanded / 'completed.json').read_text())
    if completed['status'] != 'complete_expanded_numeric_inference' or completed['outputs'] != 11:
        raise ValueError('Expanded-case cold verification incomplete')
    original_cases = json.loads((release / 'actual_swar/cases.json').read_text())
    original_report = json.loads((release / 'actual_swar/report.json').read_text())
    original_review = json.loads((release / 'review_report.json').read_text())
    original_priority = json.loads((release / 'priority_flags.json').read_text())
    plan = {row['id']: row for row in json.loads((expanded / 'plan.json').read_text())}
    if len(original_cases) != 96 or original_report['cases_scored'] != 86:
        raise ValueError('Review baseline differs from inspected v31 release')
    model_sha = sha(release / 'model.pt')
    if model_sha != original_report['model_sha256']:
        raise ValueError('Selected model identity changed')
    cases = copy.deepcopy(original_cases)
    indexed = {case['id']: case for case in cases}
    assets = {(row['id'], row['kind']): row for row in json.loads((ROOT / 'data/swar_speech_v18/inventory.json').read_text())['assets']}
    ipa = DualG2P(ROOT).ipa
    teacher = json.loads((ROOT / 'reports/numeric_teacher_check_v54/result.json').read_text())
    if teacher['model_sha256'] != model_sha:
        raise ValueError('Supplemental teacher comparison model differs')
    teacher_words = {i: word for i, word in enumerate(teacher['words'])}
    for case_id, job in plan.items():
        case = indexed[case_id]
        if case['analysis'] or case['text'] != job['original_text']:
            raise ValueError('Numeric case already scored or original text changed')
        case.update(original_text=case['text'], text=job['analysis_text'], analysis_text=job['analysis_text'],
            coverage='scored_asr_corroborated_numeric_expansion',
            transcript_expansion={'kind': 'ASR_corroborated_spoken_numeric_value', 'original_text': job['original_text'],
                'analysis_text': job['analysis_text'], 'qualified_sources': job['qualified_sources'],
                'source': str(expanded / 'plan.json'), 'source_sha256': sha(expanded / 'plan.json'),
                'human_phone_or_error_gold': False})
    for row in completed['results']:
        path = expanded / row['result']
        if sha(path) != row['sha256']:
            raise ValueError('Verified cold result changed')
        result = json.loads(path.read_text())
        if result['model_sha256'] != model_sha or result['status'] != 'scored_vowel_evidence':
            raise ValueError('Cold result model/status differs')
        if len(result['per_reference']) != 1 or result['query_content_check']['decision'] != 'pass':
            raise ValueError('Expected one real reference and content-pass expanded result')
        raw_words = copy.deepcopy(result['per_reference'][0]['words'])
        if len(raw_words) != len(result['words']):
            raise ValueError('Consensus/per-reference word identity differs')
        for i, (word, aggregate) in enumerate(zip(raw_words, result['words'], strict=True)):
            if word['word'] != aggregate['word']:
                raise ValueError('Consensus/per-reference lexical identity differs')
            word['decision'] = aggregate['decision']
            consensus = {key(site): site for site in aggregate['sites']}
            for site in word['sites']:
                chosen = consensus[key(site)]
                site.update(decision=chosen['decision'], qualified_references=chosen['qualified_references'],
                    available_references=chosen['available_references'], observed_human_variation=chosen['observed_human_variation'],
                    native_ipa_sequence=[ipa[p] for p in site['native_phones']])
                if row['id'] == 'real_yt_yt09_069' and row['kind'] == 'swar':
                    tw = teacher_words[i]
                    if tw['word'] != word['word']:
                        raise ValueError('Supplemental teacher lexical identity differs')
                    matches = [s for s in tw['sites'] if key(s) == key(site)]
                    if len(matches) == 1:
                        match = matches[0]
                        site['teacher_reference_support'] = {k: match[k] for k in ('decision', 'qualified_references', 'available_references')}
                    elif matches:
                        raise ValueError('Duplicate supplemental teacher site')
        identity = assets[row['id'], row['kind']]
        if result['inputs'][0]['sha256'] != identity['sha256']:
            raise ValueError('Expanded case query differs from served original WAV')
        indexed[row['id']]['analysis'][row['kind']] = {'words': raw_words, 'audio': identity['path'],
            'audio_sha256': identity['sha256'], 'seconds': result['seconds'], 'tracks': result['tracks'],
            'content_check': result['query_content_check'], 'cold_result_sha256': row['sha256']}
    for case_id in plan:
        case = indexed[case_id]
        if not {'real', 'swar'} <= set(case['analysis']):
            raise ValueError('Expanded case lacks required comparison readings')
        if any(word['decision'] == 'strong_vowel_change' for word in case['analysis']['real']['words']):
            raise ValueError('Real-versus-itself cold result unexpectedly flags')
    report = copy.deepcopy(original_report)
    report.update(amended_utc=datetime.now(timezone.utc).isoformat(), cases_scored=sum(bool(c['analysis']) for c in cases),
        unscored_numeric_count=sum(not bool(c['analysis']) for c in cases),
        scored_readings=sum(len(c['analysis']) for c in cases),
        numeric_expansions={'cases': 4, 'cold_outputs': 11, 'all_content_pass': True,
            'source': str(expanded / 'completed.json'), 'error_gold': False})
    report['word_decisions'] = dict(Counter(source + '/' + word['decision'] for c in cases
        for source, reading in c['analysis'].items() for word in reading['words']))
    review = copy.deepcopy(original_review)
    review['prosody_admission'] = json.loads((ROOT / 'reports/prosody_admission_v55/report.json').read_text())
    native_words = [word for c in cases for word in c['analysis'].get('real', {}).get('words', [])]
    tested = sum(bool(word['sites']) for word in native_words)
    review['word_coverage'] = {'total_word_positions': len(native_words), 'tested_word_positions': tested,
        'untested_word_positions': len(native_words) - tested,
        'meaning': 'All90 scored same-text cases, including4 explicit ASR-corroborated numeric expansions; supported vowel contrast only.'}
    review['numeric_expansion_audit'] = {'content_audit_cases': 10, 'content_audit_clips': 30,
        'corroborated_cases_scored': 4, 'ambiguous_cases_unscored': 6, 'unqualified_source_readings_omitted': 1,
        'actual_hindi_error_accuracy': None}
    priority = copy.deepcopy(original_priority)
    for case_id in plan:
        case = indexed[case_id]
        for i, word in enumerate(case['analysis']['swar']['words']):
            if word['decision'] != 'strong_vowel_change':
                continue
            sites = [s for s in word['sites'] if s['decision'] == 'strong_vowel_change']
            replicated = any(s.get('teacher_reference_support', {}).get('decision') == 'strong_vowel_change'
                and s['teacher_reference_support']['qualified_references'] == 2 for s in sites)
            priority.append({'id': case_id, 'word': word['word'], 'word_ordinal': i, 'query_start': word['start'],
                'query_end': word['end'], 'user_pick': False, 'user_confirmed_development_regression': False,
                'replicated_teacher_reference_support': replicated, 'query_content_decision': 'pass',
                'sites': [{'kind': s['site']['kind'], 'index': s['site']['index'], 'native_phones': s['native_phones'],
                    'native_ipa_sequence': s['native_ipa_sequence'], 'teacher_reference_support': s.get('teacher_reference_support'),
                    'observed_human_variation': s['observed_human_variation']} for s in sites],
                'actual_hindi_error_label': None})
    review['additional_flag_evidence'].update(word_flags=len(priority),
        replicated_teacher_supported_words=sum(p['replicated_teacher_reference_support'] for p in priority), priority_flags=priority)
    if report['cases_scored'] != 90 or report['unscored_numeric_count'] != 6:
        raise ValueError('Expanded review count invariant failed')
    targets = {release / 'actual_swar/cases.json': cases, release / 'actual_swar/report.json': report,
               release / 'review_report.json': review, release / 'priority_flags.json': priority}
    write_group(out, {'previous_review_report.json': original_review, 'previous_actual_report.json': original_report,
        'previous_priority_flags.json': original_priority,
        'previous_numeric_cases.json': [c for c in original_cases if c['id'] in plan],
        'inputs.json': {'model_sha256_unchanged': model_sha, 'cold_expansions_sha256': sha(expanded / 'completed.json'),
            'teacher_check_sha256': sha(ROOT / 'reports/numeric_teacher_check_v54/result.json'),
            'old_targets': {str(p.relative_to(ROOT)): sha(p) for p in targets}},
        'source_snapshot.json': {'sha256': sha(Path(__file__)), 'source_utf8': Path(__file__).read_text()}},
        'Concrete read-only review amendment:90/96 cases and21 candidate word flags, two '
        'known development errors and9 two-teacher-supported candidates unchanged. Four '
        'number expansions preserve original text; one Audio8 reading omitted without '
        'numeric corroboration. Failed prosody heads displayed and remain unserved. '
        'previous_* files are compact rollback state for exact prior review/priority reports '
        'and only the four changed cases (no55MB full-case backup). applied.json target SHA. '
        'Restore those four cases by id and previous reports to reverse; model/service '
        'unchanged. Active consumer review provenance/rollback, review2026-11-10; retain '
        'compact rollback while amendment consumed, no audio/model copies/HF upload.',
        ['.venv-indicf5/bin/python -m scripts.amend_swar_review'])
    for path, value in targets.items():
        temporary = path.with_suffix('.amendment_v56_partial')
        temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        temporary.replace(path)
    applied = {'created_utc': datetime.now(timezone.utc).isoformat(), 'targets': {str(p.relative_to(ROOT)): sha(p) for p in targets},
        'model_sha256': sha(release / 'model.pt'), 'cases_scored': report['cases_scored'], 'word_flags': len(priority),
        'replicated_teacher_supported_words': review['additional_flag_evidence']['replicated_teacher_supported_words']}
    (out / 'applied.json').write_text(json.dumps(applied, indent=2) + '\n')
    for directory in (release, release / 'actual_swar'):
        with (directory / 'README.md').open('a') as f:
            f.write('\n2026-11-09 v56 review amendment:4 explicitly corroborated numeric cases added,90/96 scored,21 Swar candidates/9 two-teacher-supported, failed prosody heads not promoted. Compact rollback, exact source/result receipts and target SHA in reports/review_amendment_v56. No model/threshold/source-WAV changes. Review2026-11-10.\n')
    print(json.dumps(applied))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Review amendment failed explicitly; retained rollback state permits exact restoration')
        raise
