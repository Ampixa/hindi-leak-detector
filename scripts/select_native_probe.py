"""Lock fresh native negative proxies before fitting v2; no human labels inferred."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf

from scripts.select_development_data import normalized

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/Volumes/T9/indicvoices_r_ne/groups.jsonl')
AUDIO = Path('/Volumes/T9/indicvoices_r_ne/audio')


def main() -> None:
    out = ROOT / 'data/native_probe_v2'
    if not (out / 'README.md').is_file() or (out / 'selection.json').exists():
        raise ValueError('Need a documented new selection directory')
    development = json.loads((ROOT / 'data/development_v1/selection.json').read_text())
    excluded = {j['speaker_id'] for j in development['jobs']}
    protected = {normalized(j['text']) for j in development['jobs']}
    protected.update(normalized(j['reference_text']) for j in development['jobs'])
    protected.update(normalized(j['text_devanagari']) for j in json.loads((ROOT / '.sources/neptts/benchmark/sentences.json').read_text()))
    groups = defaultdict(list)
    for number, line in enumerate(SOURCE.open(), 1):
        row = json.loads(line)
        if set(row) != {'path', 'duration', 'transcript', 'source'}:
            raise ValueError(f'Unknown source schema at {number}')
        speaker = row['source'].removeprefix('indicvoices_r/')
        text = normalized(row['transcript'])
        if speaker in excluded or not 6 <= row['duration'] <= 8 or not 12 <= len(text) <= 160:
            continue
        if any(text == p or (min(len(text), len(p)) >= 15 and (text in p or p in text)) for p in protected):
            continue
        path = Path(row['path']).resolve(strict=True)
        if not path.is_relative_to(AUDIO) or not row['source'].startswith('indicvoices_r/'):
            raise ValueError('Source outside inspected native corpus')
        groups[speaker].append(row)
    speakers = sorted(groups, key=lambda s: hashlib.sha256(('native-probe-v2:' + s).encode()).hexdigest())[:32]
    if len(speakers) != 32:
        raise ValueError('Need 32 fresh source speaker groups')
    jobs, clips = [], []
    chosen_texts = set()
    for index, speaker in enumerate(speakers):
        candidates = sorted(groups[speaker], key=lambda r: hashlib.sha256(r['path'].encode()).hexdigest())
        row = next((r for r in candidates if normalized(r['transcript']) not in chosen_texts), None)
        if row is None:
            raise ValueError('Could not select distinct fresh text')
        chosen_texts.add(normalized(row['transcript']))
        path = Path(row['path']); info = sf.info(path)
        if info.channels != 1 or abs(info.duration - row['duration']) > .02:
            raise ValueError('Audio duration/channel mismatch')
        sentence = f'probe_v2_{index:02d}'
        jobs.append({'id': sentence, 'text': row['transcript'], 'speaker_id': speaker,
                     'split': 'reserved_evaluation', 'category': 'fresh_native_negative_proxy'})
        clips.append({'clip_id': hashlib.sha256(('native-probe-v2:' + row['path']).encode()).hexdigest()[:24],
                      'system': 'native_recording', 'sentence_id': sentence, 'speaker_id': speaker,
                      'text': row['transcript'], 'split': 'reserved_evaluation', 'audio': str(path),
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'automatic_leakage_label': None})
    record = {'schema_version': 1, 'creator': 'Codex GPT-6 /root',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'source_manifest': str(SOURCE),
              'source_manifest_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              'external_audio_roots': [str(AUDIO)], 'jobs': jobs, 'clips': clips,
              'purpose': 'Fresh native negative-proxy probe locked before v2 fitting; full and short-window feature views',
              'limits': ['Native corpus identity is a weak negative assumption, not human accent gold.',
                         'Source speaker groups disjoint from development; cross-corpus real-person identity remains unverified.',
                         'Short deterministic windows can begin/end inside a word; not word-level schwa annotations.']}
    (out / 'selection.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'speakers': 32, 'recordings': 32, 'selection_sha256': hashlib.sha256((out / 'selection.json').read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
