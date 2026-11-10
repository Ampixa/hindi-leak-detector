"""Freeze voice/lexeme-disjoint schwa intervention controls before rendering."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

from hindi_detector.artifacts import write_group
from hindi_detector.candidates import schwa_slots
from hindi_detector.schwa_probe import site_paths

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config = json.loads((ROOT / 'models/kala_controls/config.json').read_text())
    human = json.loads((ROOT / 'data/human_g2p_bank_v5/word_bank.json').read_text())
    matrix = json.loads((ROOT / 'data/wordmatrix_v3/selection.json').read_text())
    excluded = {r['word'] for r in human} | {w for words in matrix['word_groups'].values() for w in words} | {'कमला', 'कर्म'}
    vocab = json.loads((ROOT / 'reports/human_g2p_evidence_v5/manifest.json').read_text())['phone_vocabulary']
    with (ROOT / '.sources/frontend/data/frontend/ipa_map.tsv').open() as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        if reader.fieldnames != ['phone', 'ipa_candidate', 'status', 'notes']:
            raise ValueError('Pinned IPA schema changed')
        ipa = {r['phone']: r['ipa_candidate'] for r in reader}
    eligible = defaultdict(list); rejected = Counter(); seen = set()
    for row in json.loads((ROOT / 'data/schwa_candidates_v1/candidates.json').read_text()):
        if row['word'] in excluded:
            rejected['excluded_existing_human_tts_or_pilot_word'] += 1; continue
        if len(row['sites']) != 1 or row['sites'][0]['target_count'] != 1 or row['sites'][0]['hindi_count'] != 0:
            rejected['not_one_1_to_0_site'] += 1; continue
        if any(p not in config['phoneme_id_map'] for p in row['target_phones'] + row['hindi_schwa_hypothesis']):
            rejected['renderer_missing_phone'] += 1; continue
        try:
            site_paths(row, ipa, vocab)
        except ValueError as error:
            rejected[str(error)] += 1; continue
        if row['word'] in seen:
            rejected['duplicate_lexeme'] += 1; continue
        seen.add(row['word'])
        skeleton, _ = schwa_slots(row['target_phones'])
        kind = 'final' if row['sites'][0]['gap_index'] == len(skeleton) else 'internal'
        eligible[kind].append(row)
    chosen = []
    for kind in ('internal', 'final'):
        rows = sorted(eligible[kind], key=lambda r: hashlib.sha256(('schwa-interventions-v7:' + r['word']).encode()).hexdigest())
        if len(rows) < 24:
            raise ValueError('Insufficient24 eligible lexemes for site kind ' + kind)
        for index, row in enumerate(rows[:24]):
            split = 'train' if index < 12 else 'development' if index < 18 else 'reserved_evaluation'
            chosen.append({**row, 'site_kind': kind, 'word_split': split})
    voices = {'barsha': 'train', 'kala': 'train', 'slr143_F': 'train',
              'slr43_0546': 'development', 'slr43_2099': 'reserved_evaluation'}
    jobs = []
    for candidate in chosen:
        for voice, voice_split in voices.items():
            for length_scale in (.85, 1.15):
                for condition, phones, class_id in [('lexicon_path', candidate['target_phones'], 0),
                                                     ('site_count_alternative', candidate['hindi_schwa_hypothesis'], 1)]:
                    identity = f"{candidate['candidate_id']}:{voice}:{length_scale}:{condition}"
                    partition = candidate['word_split'] if candidate['word_split'] == voice_split else 'crossed_diagnostic'
                    jobs.append({'id': 'cf7_' + hashlib.sha256(identity.encode()).hexdigest()[:24],
                                 'word': candidate['word'], 'candidate_id': candidate['candidate_id'],
                                 'site_kind': candidate['site_kind'], 'site': candidate['sites'][0],
                                 'word_split': candidate['word_split'], 'voice_split': voice_split,
                                 'partition': partition, 'speaker': voice, 'speaker_id': config['speaker_id_map'][voice],
                                 'length_scale': length_scale, 'condition': condition, 'requested_phones': phones,
                                 'intervention_class_id': class_id,
                                 'label_source': 'requested_single_schwa_count_intervention_not_human_error_gold',
                                 'actual_hindi_leakage_label': None})
    report = {'status': 'intervention_plan_fixed_before_rendering', 'word_families': len(chosen), 'jobs': len(jobs),
              'site_kind_counts': dict(Counter(c['site_kind'] for c in chosen)),
              'partitions': dict(Counter(j['partition'] for j in jobs)), 'voices': voices,
              'eligible_counts': {k: len(v) for k, v in eligible.items()}, 'exclusions': dict(rejected),
              'existing_human_and_tts_words_excluded': sorted(excluded),
              'native_hindi_error_labels': None}
    inputs = [ROOT / 'data/schwa_candidates_v1/candidates.json', ROOT / 'data/human_g2p_bank_v5/word_bank.json',
              ROOT / 'data/wordmatrix_v3/selection.json', ROOT / 'models/kala_controls/config.json']
    write_group(ROOT / 'data/schwa_interventions_v7', {'plan.json': {'candidates': chosen, 'jobs': jobs},
                'report.json': report, 'source_snapshot.json': {'path': str(Path(__file__).relative_to(ROOT)), 'sha256': sha(Path(__file__)), 'source_utf8': Path(__file__).read_text()}},
                'Create one-site same-voice controls with lexemes and voice IDs jointly held out. Exclude all current human-bank and original TTS focus words. Input intervention classes are not acoustic/native/Hindi gold.',
                [str(p) + ' SHA ' + sha(p) for p in inputs] + ['.venv-indicf5/bin/python -m scripts.build_intervention_pack'])
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
