"""Publish private review metadata for the unchanged frozen release."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path

from hindi_detector.corrected_synthesis import sha
from scripts.extract_corrected_training import read_rows

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / 'reports/task_vowel_release_v31'
    receipt = directory / 'review_finalization_v45.json'
    if receipt.exists():
        raise FileExistsError(receipt)
    view_path, addon_path = directory / 'review_report.json', directory / 'review_additions.json'
    before = {path.name: sha(path) for path in (view_path, addon_path)}
    view = json.loads(view_path.read_text())
    qpath = ROOT / 'reports/quality_followup_summary_v38/report.json'
    quality = json.loads(qpath.read_text())
    cold_path = ROOT / 'reports/release_cold_verification_v43/report.json'
    guard_path = ROOT / 'reports/query_guard_verification_v44/report.json'
    cold, guard = (json.loads(path.read_text()) for path in (cold_path, guard_path))
    content_path = ROOT / 'reports/swar_content_audit_v41/report.json'
    content = json.loads(content_path.read_text())
    digest = sha(directory / 'model.pt')
    if any(report['model_sha256'] != digest for report in (quality, cold, guard)):
        raise ValueError('Final review inputs use different checkpoint bytes')
    if cold['checks_passed'] != cold['checks_total'] or guard['passed'] != guard['total']:
        raise ValueError('Final inference/abstention checks failed')
    view['original_full_region_test_v32'] = view['test']
    view['test'] = {**view['test'], 'vowel_quality': {
        'final_gated_decision': quality['pooled_physical']['counts'],
        'recall_interval_95': quality['pooled_physical']['recall_interval_95_wilson'],
        'sham_flag_interval_95': quality['pooled_physical']['sham_flag_interval_95_wilson'],
        'word_cluster_recall_interval_95': quality['pooled_recall_word_cluster_bootstrap_95']}}
    view['quality_followup'] = quality
    old_words = {row['word'] for row in read_rows(ROOT / 'reports/fresh_gap_audit_v32/items.jsonl')}
    new_words = {row['word'] for row in read_rows(ROOT / 'reports/release_quality_gap_audit_v38/items.jsonl')}
    if old_words & new_words:
        raise ValueError('Follow-up word families overlap first physical cohort')
    view['evaluation_description'] = f'Full-region waveform edits across two cohorts:64 same-voice change/sham pairs and{len(old_words | new_words)} word families, with two held-out conditioning groups. The quality row pools7 earlier and18 follow-up pairs. Clear source/donor phone evidence was required; this is not real Hindi-error accuracy.'
    view['cold_checks'] = {'passed': cold['checks_passed'], 'total': cold['checks_total'], 'report': str(cold_path.relative_to(ROOT))}
    view['guard_checks'] = guard
    view['lexical_content_audit'] = content
    view['limitations'] = [line for line in view['limitations'] if not line.startswith('Quality edit evaluation has only7')]
    view['limitations'].extend(['Quality evaluation now has25 selected positive/sham pairs across21 word families; general quality-error performance and Hindi intonation remain unvalidated.',
        'The128 new requested quality words yielded18 eligible physical pairs; requested spellings are not observed pronunciation labels.',
        'Automatic cached/new references require content pass and per-site phonetic qualification. Two cached case texts have no content-pass teacher reference and explicitly abstain.',
        'Optional query content checking rejects only gross mismatch. ASR quarantine is not an accent label and does not suppress valid known phonetic errors.'])
    cases = json.loads((directory / 'actual_swar/cases.json').read_text())
    observations = json.loads((directory / 'native_observations.json').read_text())
    teacher_words = json.loads((ROOT / 'reports/release_teacher_consensus_v36/words.json').read_text())
    teacher = {(row['id'], row['source'], row['word_ordinal']): row for row in teacher_words}
    content_rows = {(row['id'], row['kind']): row for row in read_rows(ROOT / 'reports/swar_content_audit_v41/items.jsonl')}
    additions = json.loads(addon_path.read_text())
    priority = []
    regressions = json.loads((ROOT / 'data/swar_phone_plan_v18/regressions.json').read_text())
    for case in cases:
        for source, reading in case['analysis'].items():
            for ordinal, word in enumerate(reading['words']):
                t = teacher[case['id'], source, ordinal]
                lookup = {(site['site']['kind'], site['site']['index'], tuple(site['native_phones'])): site for site in t['sites']}
                for addition in additions[case['id']][source][ordinal]['sites']:
                    other = lookup.get((addition['kind'], addition['index'], tuple(addition['native_phones'])))
                    addition['teacher_reference_support'] = ({k: other[k] for k in ('decision', 'qualified_references', 'available_references')}
                        if other else {'decision': 'matching_site_unavailable', 'qualified_references': 0, 'available_references': 2})
                if source != 'swar' or word['decision'] != 'strong_vowel_change':
                    continue
                sites = [site for site in additions[case['id']][source][ordinal]['sites']
                         if any(actual['decision'] == 'strong_vowel_change' and actual['site']['kind'] == site['kind']
                                and actual['site']['index'] == site['index'] and actual['native_phones'] == site['native_phones'] for actual in word['sites'])]
                confirmed = any(all(label[k] == value for k, value in [('id', case['id']), ('source', source), ('word', word['word'])]) for label in regressions)
                replicated = any(site['teacher_reference_support']['decision'] == 'strong_vowel_change' and site['teacher_reference_support']['qualified_references'] == 2 for site in sites)
                priority.append({'id': case['id'], 'word': word['word'], 'word_ordinal': ordinal,
                    'query_start': word['start'], 'query_end': word['end'], 'user_pick': case['user_pick'],
                    'user_confirmed_development_regression': confirmed, 'replicated_teacher_reference_support': replicated,
                    'query_content_decision': content_rows[case['id'], source]['decision'],
                    'sites': sites, 'actual_hindi_error_label': True if confirmed else None})
    priority.sort(key=lambda row: (not row['user_confirmed_development_regression'], not row['replicated_teacher_reference_support'], not row['user_pick'], row['id'], row['word_ordinal']))
    view['additional_flag_evidence'] = {'word_flags': len(priority),
        'replicated_teacher_supported_words': sum(row['replicated_teacher_reference_support'] for row in priority),
        'all_other_real_error_labels_unknown': True, 'priority_flags': priority}
    view['observed_human_context'] = {'contrast_entries': len(observations['entries']),
        'recorded_occurrences': sum(len(entry['recordings']) for entry in observations['entries']), 'used_for_prediction': False}
    view_path.write_text(json.dumps(view, ensure_ascii=False, indent=2) + '\n')
    addon_path.write_text(json.dumps(additions, ensure_ascii=False) + '\n')
    priority_path = directory / 'priority_flags.json'
    priority_path.write_text(json.dumps(priority, ensure_ascii=False, indent=2) + '\n')
    pins = [qpath, cold_path, guard_path, content_path, ROOT / 'reports/release_teacher_consensus_v36/words.json']
    final = {'created_utc': datetime.now(timezone.utc).isoformat(), 'creator': 'Codex GPT-6 /root', 'owner': 'Ampixa/user',
        'model_sha256_unchanged': digest, 'previous_hashes': before,
        'inputs': [{'path': str(path.relative_to(ROOT)), 'sha256': sha(path)} for path in pins],
        'files': [{'path': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path)} for path in (view_path, addon_path, priority_path)],
        'source_sha256': sha(Path(__file__)), 'page_sha256': sha(ROOT / 'review/swar-words.html'),
        'purpose': 'Private review of frozen-head validation, content/abstention checks and corroborated candidate locations. No predictions or model thresholds changed.',
        'status': 'active', 'review': '2026-11-09', 'next_review': '2026-11-10', 'not_on_HF': True}
    receipt.write_text(json.dumps(final, indent=2) + '\n')
    provenance = json.loads((directory / 'review_provenance.json').read_text())
    for row in provenance['files']:
        path = directory / row['path']
        row.update(bytes=path.stat().st_size, sha256=sha(path))
    provenance['amendments'].append(receipt.name)
    (directory / 'review_provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
    with (directory / 'README.md').open('a') as handle:
        handle.write('\nreview_finalization_v45.json inventories final pooled-quality review/counts,8/8 cold and5/5 guard checks, independent lexical-content audit, per-site teacher-reference corroboration and priority_flags.json. priority flags remain hypotheses except the two user-reported development regressions. Descriptive observed human context never changes predictions. All file/input/source/page hashes retained. Model bytes unchanged; review2026-11-09,next2026-11-10. Current consumer private morning model review; retain compact evidence while model consumed, transient diagnostics expire24h after consumer closes.\n')
    print(json.dumps({'quality_detected': quality['pooled_physical']['counts']['tp'], 'quality_edits': 25,
        'word_flags': len(priority), 'replicated_teacher_supported_words': sum(row['replicated_teacher_reference_support'] for row in priority)}))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Final private review metadata assembly failed explicitly')
        raise
