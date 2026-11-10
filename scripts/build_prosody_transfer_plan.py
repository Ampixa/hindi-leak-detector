"""Freeze voice-controlled Hindi/Native prosody-transfer inputs before rendering."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import logging
from pathlib import Path
import shutil

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import require_pair, sha
from scripts.extract_corrected_training import read_rows

ROOT = Path(__file__).resolve().parents[1]
QUOTAS = {'train': 100, 'development': 24, 'reserved_evaluation': 40}


def rank(value):
    return hashlib.sha256(('prosody-transfer-v49:' + value).encode()).hexdigest()


def main():
    out = ROOT / 'data/prosody_transfer_v49'
    if out.exists():
        raise FileExistsError(out)
    original = ROOT / 'data/teacher_indicf5_v14'
    features = ROOT / 'reports/structured_evidence_v15'
    plan = json.loads((original / 'selection.json').read_text())
    pair = json.loads((original / 'model_pair.json').read_text())
    require_pair(pair)
    rows = read_rows(features / 'items.jsonl')
    grouped = defaultdict(list)
    for row in rows:
        if row['kind'] == 'original' and row['status'] == 'ok_structured_features' and row['row']['content_quality']['decision'] == 'pass':
            grouped[row['text_id']].append(row)
    picks = {row['text'] for row in json.loads((ROOT / 'data/swar_speech_v18/selection.json').read_text()) if row.get('user_pick')}
    references = defaultdict(dict)
    for job in plan['jobs']:
        references[job['split']][job['speaker_id']] = job
    candidates = defaultdict(list)
    details = {}
    for key, quartet in grouped.items():
        if len(quartet) != 4 or max(row['row']['seconds'] for row in quartet) > 12:
            continue
        if len({(row['row']['system'], row['row']['speaker_id']) for row in quartet}) != 4:
            raise ValueError('Original quartet has duplicate system/conditioning identity')
        if quartet[0]['row']['text'] in picks:
            continue
        for row in quartet:
            path = features / row['detail']
            if sha(path) != row['detail_sha256']:
                raise ValueError('Frozen original word metadata changed')
            details[row['clip_id']] = json.loads(path.read_text())
        words = [word['word'] for word in details[quartet[0]['clip_id']]['words']]
        if any([word['word'] for word in details[row['clip_id']]['words']] != words for row in quartet):
            raise ValueError('Original same-text quartet lost aligned word identity')
        part = quartet[0]['row']['split']
        for word in quartet[0]['row']['target_words']:
            if word in {'हैन', 'किनकि'} or plan['word_partitions'].get(word) != part or words.count(word) != 1:
                continue
            ordinal = words.index(word)
            spans = [details[row['clip_id']]['words'][ordinal] for row in quartet]
            if not all(.15 <= span['end'] - span['start'] <= 2.5 and span['prosody']['pitch_available'] == 1
                       and span['prosody']['voiced_fraction'] >= .3 for span in spans):
                continue
            candidates[part].append((rank(word + ':' + key), word, key, ordinal, quartet))
    selected = []
    jobs = []
    prompt_sources = {}
    for part, quota in QUOTAS.items():
        seen = set()
        for _, word, key, ordinal, quartet in sorted(candidates[part], key=lambda row: row[0]):
            if word in seen:
                continue
            native = sorted([row for row in quartet if row['row']['system'] == 'pocket_teacher_250000_24l'],
                            key=lambda row: rank(key + ':' + row['row']['speaker_id']))
            if len(native) != 2:
                raise ValueError('Need two verified original Native voices')
            carrier, reference = native
            hindi = next(row for row in quartet if row['row']['system'] == 'indicf5_stock'
                         and row['row']['speaker_id'] == carrier['row']['speaker_id'])
            used = {row['row']['speaker_id'] for row in native}
            extra = [job for speaker, job in references[part].items() if speaker not in used]
            if not extra:
                raise ValueError('Need an independent third Native conditioning group')
            prompt = sorted(extra, key=lambda job: rank(key + ':' + job['speaker_id']))[0]
            identifier = 'proso49_' + rank(key + ':' + word)[:20]
            source = original / prompt['reference_audio']
            if sha(source) != prompt['reference_sha256']:
                raise ValueError('Third Native prompt differs from exact24-layer plan')
            prompt_sources[source.name] = source
            job = {'id': identifier, 'text_id': key, 'text': carrier['row']['text'],
                'canonical_text': carrier['row']['text'], 'text_kind': 'third_native_prosody_donor',
                'target_words': [word], 'speaker_id': prompt['speaker_id'], 'split': part,
                'reference_audio': 'references/' + source.name, 'reference_text': prompt['reference_text'],
                'reference_sha256': prompt['reference_sha256'], 'source_reference_sha256': prompt['source_reference_sha256'],
                'seed': 0, 'automatic_leakage_label': None}
            selected.append({'id': identifier, 'word': word, 'word_ordinal': ordinal, 'text_id': key,
                'text': job['text'], 'split': part,
                'carrier': carrier, 'comparison_reference': reference, 'hindi_prosody_donor': hindi,
                'third_native_donor_job': identifier,
                'conditioning_groups': {'carrier': carrier['row']['speaker_id'],
                    'reference': reference['row']['speaker_id'], 'native_donor': prompt['speaker_id']}})
            jobs.append(job)
            seen.add(word)
            if len(seen) == quota:
                break
        if len(seen) != quota:
            raise ValueError('Insufficient distinct eligible words for ' + part + ': ' + str(len(seen)))
    if len({row['word'] for row in selected}) != len(selected):
        raise ValueError('Word family crossed frozen partitions')
    by_part = {part: {speaker for row in selected if row['split'] == part for speaker in row['conditioning_groups'].values()} for part in QUOTAS}
    if any(a & b for i, a in enumerate(by_part.values()) for b in list(by_part.values())[i + 1:]):
        raise ValueError('Conditioning group crossed partitions')
    protocol = {'partitions': QUOTAS, 'selection': 'Metadata hash rank; distinct focal words, content-pass exact original quartets, voiced word150..2500ms, utterances<=12sec; no classifier scores.',
        'carrier': 'Correct24-layer Native teacher waveform; Native spectral envelope, aperiodicity and voicing mask fixed for every class.',
        'conditions': ['WORLD sham using own contour', 'third-Native-donor contour transfer', 'stock-IndicF5-donor contour transfer'],
        'donor_normalization': 'Donor utterance-median-relative log F0; same20 normalized word-time knots and30ms edge blend for both donor classes. Native carrier median sets absolute pitch.',
        'measurement': 'Remeasure emitted waveform F0/energy; do not use intended target arrays as detector features.',
        'labels': 'Native versus Hindi-checkpoint prosody transfer is known source-style supervision, not real Hindi-error gold.',
        'phonetic_controls': 'No spectrum/aperiodicity/voicing-mask/word-boundary substitution. Check actual outputs and phonetic stability; reject invalid outputs explicitly.',
        'nuisance_audit': 'Global pitch+/-3 semitones and volume changes must not create additional prosody flags; Native-to-Native transfers are explicit controls.',
        'test_policy': 'Freeze model/thresholds on train/dev before opening reserved intervention scores; actual Swar remains diagnostic, not an independently labeled prosody benchmark.',
        'known_user_cases': 'Both reported target words and all six picked sentences excluded from new fitting.',
        'source_model_pair_sha256': sha(original / 'model_pair.json'), 'source_items_sha256': sha(features / 'items.jsonl')}
    write_group(out, {'selection.json': {'experiment_id': 'teacher_indicf5_v14', 'role': 'prosody_transfer_third_native_donor', 'jobs': jobs},
        'model_pair.json': pair, 'transfer_plan.json': selected, 'protocol.json': protocol,
        'source_snapshot.json': {'sha256': sha(Path(__file__)), 'source_utf8': Path(__file__).read_text()}},
        'Frozen voice-controlled prosody transfer experiment. transfer_plan.json pins164 exact '
        'Native carriers/comparison references and stockIndicF5 donors; selection.json requests '
        'one extra Native donor per context from a third conditioning group. Word families and '
        'conditioning groups remain disjoint across100/24/40 train/dev/reserved contexts. '
        'references/*.wav exact verified prompt copies, reference_inventory.json source/bytes/SHA; '
        'teacher/ later owns generated Native donor outputs and receipts. No real accent/error '
        'labels fabricated and no change to served vowel model. Consumer replacement for failed '
        'legacy prosody-source head; review2026-11-10. Raw redundant outputs expire24h after '
        'consumer closes; keep compact configs/metrics/selected state. NOT on HF; inherited rights.',
        ['.venv-indicf5/bin/python -m scripts.build_prosody_transfer_plan'])
    (out / 'references').mkdir()
    inventory = []
    for name, source in prompt_sources.items():
        target = out / 'references' / name
        shutil.copyfile(source, target)
        inventory.append({'path': str(target.relative_to(out)), 'source': str(source),
            'bytes': target.stat().st_size, 'sha256': sha(target)})
    (out / 'reference_inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    print(json.dumps({'contexts': len(selected), 'new_Native_donors': len(jobs), 'parts': dict(Counter(row['split'] for row in selected))}))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Voice-controlled prosody plan freezing failed explicitly')
        raise
