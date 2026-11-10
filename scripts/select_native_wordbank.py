"""Select real multi-speaker lexical anchors without copying corpus WAVs."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import unicodedata

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
CORPUS = Path('/Volumes/T9/indicvoices_r_ne')


def tokens(text):
    return [''.join(c for c in w if unicodedata.category(c)[0] in {'L', 'M'}) for w in text.split()]


def main():
    out = ROOT / 'data/native_wordbank_v3'
    if not (out / 'README.md').is_file() or (out / 'selection.json').exists():
        raise ValueError('Need a documented new native-word bank')
    development = json.loads((ROOT / 'data/development_v1/selection.json').read_text())
    probe = json.loads((ROOT / 'data/native_probe_v2/selection.json').read_text())
    excluded = {j['speaker_id'] for j in development['jobs']} | {c['speaker_id'] for c in probe['clips']}
    protected = {''.join(tokens(j['text'])) for j in development['jobs']}
    protected |= {''.join(tokens(r['text_devanagari'])) for r in json.loads((ROOT / '.sources/neptts/benchmark/sentences.json').read_text())}
    candidates = json.loads((ROOT / 'data/schwa_candidates_v1/candidates.json').read_text())
    words = {r['word'] for r in candidates}
    rows, coverage = [], defaultdict(dict)
    manifest = CORPUS / 'groups.jsonl'
    for line in manifest.read_text().splitlines():
        row = json.loads(line)
        if set(row) != {'path', 'duration', 'transcript', 'source'}:
            raise ValueError('Native corpus structure changed')
        if not row['source'].startswith('indicvoices_r/'):
            raise ValueError('Unknown native source group namespace')
        speaker = row['source'].removeprefix('indicvoices_r/')
        if speaker in excluded or not 2 <= row['duration'] <= 12 or ''.join(tokens(row['transcript'])) in protected:
            continue
        found = set(tokens(row['transcript'])) & words
        if not found or len(tokens(row['transcript'])) > 32:
            continue
        row = {**row, 'speaker_id': speaker, 'matched_words': sorted(found)}
        index = len(rows); rows.append(row)
        for word in found:
            old = coverage[word].get(speaker)
            if old is None or row['duration'] < rows[old]['duration']:
                coverage[word][speaker] = index
    # Require actual independent source-group coverage. No invented examples for absent lexemes.
    eligible = [w for w, groups in coverage.items() if len(groups) >= 8]
    focus = set(sum(json.loads((ROOT / 'data/wordmatrix_v3/selection.json').read_text())['word_groups'].values(), []))
    selected_words = sorted(eligible, key=lambda w: (w not in focus, -len(coverage[w]), w))[:20]
    if len(selected_words) != 20:
        raise ValueError('Insufficient verified multi-speaker lexical coverage')
    chosen = set()
    word_groups = {}
    for word in selected_words:
        groups = sorted(coverage[word], key=lambda s: hashlib.sha256(('wordbank-v3:' + word + ':' + s).encode()).hexdigest())[:6]
        word_groups[word] = groups
        chosen.update(coverage[word][s] for s in groups)
    jobs, clips = [], []
    for index in sorted(chosen):
        row = rows[index]; path = Path(row['path']).resolve(strict=True)
        if not path.is_relative_to((CORPUS / 'audio').resolve()):
            raise ValueError('Native corpus path outside its declared audio root')
        info = sf.info(path)
        if info.channels != 1 or not 2 <= info.duration <= 12.02:
            raise ValueError('Selected native recording does not match expected mono/duration')
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        identifier = 'native_bank_' + hashlib.sha256(str(path).encode()).hexdigest()[:20]
        jobs.append({'id': identifier, 'text': row['transcript'], 'speaker_id': row['speaker_id'], 'split': 'train',
                     'matched_anchor_words': [w for w, speakers in word_groups.items() if coverage[w].get(row['speaker_id']) == index and row['speaker_id'] in speakers]})
        clips.append({'clip_id': identifier, 'sentence_id': identifier, 'system': 'native_recording',
                      'speaker_id': row['speaker_id'], 'split': 'train', 'text': row['transcript'],
                      'audio': str(path), 'sha256': sha, 'bytes': path.stat().st_size, 'automatic_leakage_label': None})
    result = {'schema_version': 1, 'creator': 'Codex GPT-6 /root', 'jobs': jobs, 'clips': clips,
              'external_audio_roots': [str(CORPUS / 'audio')], 'word_source_groups': word_groups,
              'excluded_source_groups': sorted(excluded), 'all_eligible_word_coverage': {w: len(coverage[w]) for w in sorted(eligible)},
              'corpus_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
              'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'limits': ['Native corpus transcripts/groups are upstream metadata, not verified per-phone gold.',
                         'Bank is for phonetic variation; unmatched phrase contexts cannot define prosodic correctness.',
                         'Old benchmark opaque speaker IDs cannot establish cross-corpus real-person identity.']}
    (out / 'selection.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'anchor_words': len(selected_words), 'utterances': len(clips), 'speaker_groups': len({c['speaker_id'] for c in clips})}))


if __name__ == '__main__':
    main()
