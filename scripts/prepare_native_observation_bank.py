"""Expose observed human variation beside flags, without changing predictions."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

from hindi_detector.corrected_synthesis import sha
from hindi_detector.native_observations import matching_observations
from scripts.extract_corrected_training import read_rows
from scripts.score_swar_sites import load_item

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / 'reports/task_vowel_release_v31'
    target = directory / 'native_observations.json'
    if target.exists():
        raise FileExistsError(target)
    teacher = ROOT / 'reports/release_teacher_consensus_v36/report.json'
    paired = ROOT / 'reports/release_native_variation_v37/comparisons.json'
    a = json.loads(teacher.read_text())
    pairs = json.loads(paired.read_text())
    primary = ROOT / 'reports/swar_corpus_features_v18'
    secondary = ROOT / 'reports/swar_phone_crosscheck_v20'
    receipts = [{row['id']: row for row in read_rows(folder / 'items.jsonl') if row['kind'] == 'real'}
                for folder in (primary, secondary)]
    metadata = {row['id']: row for row in json.loads((ROOT / 'data/swar_speech_v18/selection.json').read_text())}
    digest = sha(directory / 'model.pt')
    if a['model_sha256'] != digest:
        raise ValueError('Teacher variation report model differs')
    bank = defaultdict(dict)
    cache = {}

    def add(word, site, phones, case_id, ordinal, evidence):
        key = (word, site['kind'], site['index'], tuple(phones), site.get('alternative'))
        identity = (case_id, ordinal)
        if identity not in bank[key]:
            for index, folder in enumerate((primary, secondary)):
                if (index, case_id) not in cache:
                    cache[index, case_id] = load_item(folder, receipts[index][case_id])
            p = cache[0, case_id]['detail']['words'][ordinal]
            s = cache[1, case_id]['detail']['words'][ordinal]
            if p['word'] != word or s['word'] != word:
                raise ValueError('Observed word/audio alignment identity differs')
            bank[key][identity] = {'id': case_id, 'word_ordinal': ordinal, 'voice': metadata[case_id]['voice'],
                'word_start': p['start'], 'word_end': p['end'], 'source_audio_sha256': receipts[0][case_id]['sha256'],
                'phone_probe_sequence': p['phone_probe_sequence'], 'secondary_phone_probe_sequence': s['phone_probe_sequence'],
                'evidence_sources': []}
        bank[key][identity]['evidence_sources'].append(evidence)

    for row in a['flagged_words']:
        if row['source'] != 'real':
            continue
        for site in row['sites']:
            if site['decision'] == 'strong_vowel_change':
                add(row['word'], site['site'], site['native_phones'], row['id'], row['word_ordinal'],
                    {'audit': str(teacher.relative_to(ROOT)), 'qualified_teacher_references': site['qualified_references']})
    for row in pairs:
        for site in row['sites']:
            if site['decision'] == 'strong_vowel_change':
                add(row['word'], site['site'], site['native_phones'], row['query_case'], row['query_ordinal'],
                    {'audit': str(paired.relative_to(ROOT)), 'reference_case': row['reference_case']})
    entries = [{'word': key[0], 'site': {'kind': key[1], 'index': key[2], 'alternative': key[4]},
                'native_phones': list(key[3]), 'recordings': list(rows.values())} for key, rows in sorted(bank.items(), key=lambda item: repr(item[0]))]
    result = {'schema': 'observed_real_variation_v1', 'model_sha256': digest,
        'created_utc': datetime.now(timezone.utc).isoformat(), 'creator': 'Codex GPT-6 /root', 'owner': 'Ampixa/user',
        'status': 'active', 'next_review': '2026-11-10', 'scope_real_recordings': 86, 'scope_source_groups': 12,
        'entries': entries, 'used_for_training_or_prediction': False, 'human_error_labels': None,
        'inputs': [{'path': str(path.relative_to(ROOT)), 'sha256': sha(path)} for path in (teacher, paired)],
        'limits': ['Observed changes may be legitimate variants or acoustic/reference errors; Native acceptability and Hindi influence are not inferred.',
            'Video/source groups do not prove independent people or geography. Missing observations are not evidence of unnativeness.',
            'This word-indexed bank is descriptive context only; all numeric scores/thresholds/decisions remain unchanged.']}
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    addons = directory / 'review_additions.json'
    before = sha(addons)
    values = json.loads(addons.read_text())
    cases = {row['id']: row for row in json.loads((directory / 'actual_swar/cases.json').read_text())}
    for case_id, readings in values.items():
        for source, words in readings.items():
            actual = cases[case_id]['analysis'][source]['words']
            for ordinal, word in enumerate(words):
                actual_sites = {(site['site']['kind'], site['site']['index'], tuple(site['native_phones'])): site for site in actual[ordinal]['sites']}
                for site in word['sites']:
                    original = actual_sites[site['kind'], site['index'], tuple(site['native_phones'])]
                    site['observed_human_variation'] = matching_observations(result, actual[ordinal]['word'], original['site'], site['native_phones'], exclude_case=case_id)
    addons.write_text(json.dumps(values, ensure_ascii=False) + '\n')
    receipt = {'created_utc': result['created_utc'], 'creator': result['creator'], 'owner': result['owner'],
        'model_sha256_unchanged': digest, 'previous_additions_sha256': before, 'files': [
            {'path': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path)} for path in (target, addons)],
        'new_descriptive_entries': len(entries), 'source_sha256': sha(Path(__file__)),
        'purpose': 'Expose observed human variation alongside flags and generic CLI results; no prediction changes.',
        'status': 'active', 'next_review': '2026-11-10', 'not_on_HF': True}
    (directory / 'native_observation_provenance.json').write_text(json.dumps(receipt, indent=2) + '\n')
    provenance = json.loads((directory / 'review_provenance.json').read_text())
    for row in provenance['files']:
        if row['path'] == addons.name:
            row.update(sha256=sha(addons), bytes=addons.stat().st_size)
    provenance['amendments'].append('native_observation_provenance.json')
    (directory / 'review_provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
    with (directory / 'README.md').open('a') as handle:
        handle.write('\nnative_observations.json is descriptive observed real-speech variation, never a prediction/training input or acceptability label. native_observation_provenance.json inventories exact new bank/additions hashes and source. Review2026-11-09,next2026-11-10; retain compact context while current user review/inference consumes it. No waveform copies, NOT on HF, inherited source/MMS rights.\n')
    print(json.dumps({'entries': len(entries), 'recorded_occurrences': sum(len(entry['recordings']) for entry in entries)}))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Observed human-variation context preparation failed explicitly')
        raise
