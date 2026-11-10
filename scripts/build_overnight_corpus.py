"""Freeze large paired Pocket texts/voices and variant-aware G2P focus metadata."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import unicodedata

from hindi_detector.artifacts import write_group

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)
CARRIERS = [
    'कृपया {word} भन्ने शब्द एकपटक भन्नुहोस्।',
    'यो वाक्यमा {word} शब्द प्रयोग गरिएको छ।',
    'उनले अन्त्यमा {word} शब्द दोहोर्‍याए।',
    'तपाईंले {word} भन्ने शब्द सुन्नुभयो?',
    'आजको अभ्यासमा हामी {word} शब्द पढ्छौँ।',
    'यहाँ {word} शब्द स्पष्ट सुनिनुपर्छ।',
    'म फेरि {word} भन्ने शब्द भन्छु।',
    'अर्को शब्द होइन, मैले {word} भनेको हुँ।',
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def words(text: str) -> list[str]:
    return [''.join(c for c in token if unicodedata.category(c)[0] in {'L', 'M'}) for token in text.split()]


def partition(key: str) -> str:
    value = int(hashlib.sha256(('overnight-v11:' + key).encode()).hexdigest()[:8], 16) % 100
    return 'train' if value < 70 else 'development' if value < 85 else 'reserved_evaluation'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    candidates_path = ROOT / 'data/schwa_candidates_v1/candidates.json'
    candidates = json.loads(candidates_path.read_text()); by_word = defaultdict(list)
    for row in candidates:
        by_word[row['word']].append(row)
    word_parts = {w: partition('word:' + w) for w in by_word}
    matrix_path = ROOT / 'data/wordmatrix_v3/selection.json'
    matrix = json.loads(matrix_path.read_text()); voices = defaultdict(dict)
    for job in matrix['jobs']:
        voices[job['split']].setdefault(job['speaker_id'], {k: job[k] for k in ('speaker_id', 'reference_audio', 'reference_text', 'reference_sha256')})
    all_voices = [v for group in voices.values() for v in group.values()]
    if len({v['speaker_id'] for v in all_voices}) != len(all_voices) or any(len(g) < 2 for g in voices.values()):
        raise ValueError('Need disjoint actual conditioning groups in every partition')
    for voice in all_voices:
        if sha(Path(voice['reference_audio'])) != voice['reference_sha256']:
            raise ValueError('Conditioning reference bytes changed')
    excluded = set()
    for name in ['data/wordmatrix_v3/selection.json', 'data/human_g2p_bank_v5/selection.json', 'data/development_v1/selection.json', 'data/indicf5_audio8_v1/selection.json']:
        selection = json.loads((ROOT / name).read_text()); excluded.update(j['text'] for j in selection['jobs'])
    metadata_path = ROOT / 'data/human_regions_v5/single_utterance_candidates.json'
    texts = {}; coverage = Counter()
    for row in json.loads(metadata_path.read_text()):
        text = unicodedata.normalize('NFC', row['text']); tokens = words(text)
        if text in excluded or text in texts or not 1.5 <= row['local_seconds'] <= 12 or not 4 <= len(tokens) <= 24:
            continue
        if any(unicodedata.category(c)[0] == 'N' for c in text) or any(c.isalpha() and 'DEVANAGARI' not in unicodedata.name(c, '') for c in text):
            continue
        split = partition('text-source-group:' + row['speaker_id'])
        targets = sorted(set(tokens) & by_word.keys())
        active = [w for w in targets if word_parts[w] == split]
        texts[text] = {'text': text, 'split': split, 'text_kind': 'natural_transcript',
                       'source_text_group': row['speaker_id'], 'source_shard': row['source_shard'], 'source_row': row['source_row'],
                       'all_contrast_words': targets, 'target_words': active}
        coverage.update(active)
    natural_count = len(texts)
    for word in sorted(by_word):
        for carrier in CARRIERS:
            text = carrier.format(word=word)
            if text in excluded or text in texts:
                continue
            texts[text] = {'text': text, 'split': word_parts[word], 'text_kind': 'authored_targeted_carrier',
                           'source_text_group': None, 'all_contrast_words': sorted(set(words(text)) & by_word.keys()), 'target_words': [word]}
    # Actual variants are preserved; first lexicon lookup is not treated as native gold.
    sys.path.insert(0, str(ROOT / '.sources/frontend'))
    data = importlib.import_module('nepali_frontend.data')
    rules = importlib.import_module('nepali_frontend.g2p.post_rules')
    variants = data.lexicon_variants(); ipa_path = ROOT / '.sources/frontend/data/frontend/ipa_map.tsv'
    with ipa_path.open() as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        if reader.fieldnames != ['phone', 'ipa_candidate', 'status', 'notes']:
            raise ValueError('Actual IPA mapping schema differs')
        ipa = {r['phone']: r['ipa_candidate'] for r in reader}
    priors = []; already_native = 0
    executable = Path('/opt/homebrew/bin/espeak-ng').resolve(strict=True)
    version = subprocess.check_output([str(executable), '--version'], text=True)
    if not version.startswith('eSpeak NG text-to-speech: 1.52.0 '):
        raise ValueError('Hindi G2P version differs')
    for word, rows in sorted(by_word.items()):
        promoted = [rules.apply_anusvara_to_phone_list(v)[0] for v in variants[word]]
        paths = {tuple(p for p in v if p != '.') for v in promoted}
        aliases_already_listed = [r['candidate_id'] for r in rows if tuple(r['hindi_schwa_hypothesis']) in paths]
        already_native += len(aliases_already_listed)
        hindi = subprocess.run([str(executable), '-q', '-v', 'inc/hi', '--ipa', '--sep=|', word], capture_output=True, text=True, check=True)
        if not hindi.stdout.strip() or hindi.stderr.strip():
            raise ValueError('Hindi G2P returned empty output or warning for ' + word)
        priors.append({'word': word, 'partition': word_parts[word], 'nepali_project_variants': promoted,
                       'nepali_ipa_variants': [[ipa.get(p) if p != '.' else '.' for p in v] for v in promoted],
                       'hindi_espeak_raw': hindi.stdout, 'candidates': rows,
                       'mined_hindi_alias_already_in_nepali_variants': aliases_already_listed,
                       'actual_pronunciation_gold': None, 'first_variant_is_gold': False})
    jobs = []; text_rows = []
    for row in sorted(texts.values(), key=lambda r: hashlib.sha256(r['text'].encode()).hexdigest()):
        identifier = 'night11_' + hashlib.sha256(row['text'].encode()).hexdigest()[:20]
        text_rows.append(dict(row, id=identifier))
        pool = list(voices[row['split']].values()); offset = int(hashlib.sha256(identifier.encode()).hexdigest()[:8], 16) % len(pool)
        for ordinal in range(2):
            voice = pool[(offset + ordinal) % len(pool)]
            jobs.append({'id': identifier + '_v' + str(ordinal), 'text_id': identifier, **row, **voice,
                         'seed': 0, 'automatic_leakage_label': None})
    report = {'status': 'large_paired_corpus_frozen_before_generation', 'unique_texts': len(text_rows),
              'natural_texts': natural_count, 'targeted_texts': len(text_rows) - natural_count,
              'target_lexemes': len(by_word), 'conditioning_groups': len(all_voices), 'jobs_per_system': len(jobs),
              'planned_audio_clips': len(jobs) * 2, 'text_partition_counts': dict(Counter(r['split'] for r in text_rows)),
              'natural_focal_word_occurrences': sum(coverage.values()),
              'mined_aliases_already_listed_as_nepali_candidate_variants': already_native,
              'deadline_utc': '2026-11-08T02:15:00+00:00',
              'limits': ['Text/conditioning groups and designated focal lexemes are separated; incidental non-focal words can overlap.',
                         'Authored carriers are metalinguistic pronunciation probes; punctuation is intended style, not verified delivered prosody.',
                         'Source model roles and mined dictionary contrasts are weak supervision; actual realized phones/errors are not gold.',
                         'ASR filtering must be calibrated on training data; report rejection/coverage separately by source and text kind.',
                         'Hindi-looking dictionary aliases that also occur in Nepali variant data cannot establish exclusive Hindi errors.']}
    source = Path(__file__)
    write_group(args.out, {'selection.json': {'schema_version': 1, 'experiment_id': 'pocket_bulk_overnight_v11', 'jobs': jobs, 'clips': [],
                                            'texts': text_rows, 'word_partitions': word_parts},
                          'g2p_priors.json': priors, 'report.json': report,
                          'inputs.json': {str(p.relative_to(ROOT)): sha(p) for p in (candidates_path, matrix_path, metadata_path, ipa_path)},
                          'source_snapshot.json': {'path': str(source.relative_to(ROOT)), 'sha256': sha(source), 'source_utf8': source.read_text()}},
                'Large same-text Hindi/Nepali Pocket corpus, two conditioning groups per text, independent conditioning partitions, designated focal-word splits and all native G2P variants retained. Native audio is not copied; this uses text only.',
                ['Pinned source corpus metadata and existing16 verified TTS references', '.venv-indicf5/bin/python -m scripts.build_overnight_corpus --out NEW'])
    print(json.dumps(report))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Scaled corpus freezing failed; no guessed source schemas or phone labels')
        raise
