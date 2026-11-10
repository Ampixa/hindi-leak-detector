"""Recover human speaker regions and utterance keys via metadata-only Parquet ranges."""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import sqlite3
import sys
import time
import unicodedata

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/human_regions_v5'
REVISION = '5f4495c91d500742a58d1be2ab07d77f73c0acf8'
REPOSITORY = 'ai4bharat/indicvoices_r'
COLUMNS = ['speaker_id', 'state', 'district', 'lang', 'normalized', 'text', 'verbatim', 'duration', 'samples', 'scenario', 'task_name']
LOG = logging.getLogger(__name__)


def fingerprint(text):
    text = unicodedata.normalize('NFC', text)
    canonical = ' '.join(''.join(c if unicodedata.category(c)[0] in {'L', 'M', 'N'} else ' ' for c in text).split())
    return hashlib.sha256(canonical.encode()).hexdigest() if canonical else None


def save(name, value):
    p = OUT / name; tmp = p.with_name(p.name + '.partial')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'); tmp.replace(p)


def main():
    if not (OUT / 'README.md').is_file() or (OUT / 'audit.json').exists():
        raise ValueError('Need a fresh documented metadata recovery group')
    credential = sys.stdin.readline().strip() or None
    info = HfApi(token=credential).dataset_info(REPOSITORY, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ValueError('Human dataset revision changed')
    files = sorted([f for f in info.siblings if f.rfilename.startswith('Nepali/train-') and f.rfilename.endswith('.parquet')], key=lambda f: f.rfilename)
    if len(files) != 139:
        raise ValueError('Unexpected inspected Nepali shard inventory')
    save('source_snapshot.json', {'path': 'scripts/recover_human_regions.py', 'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'source_utf8': Path(__file__).read_text()})
    db = sqlite3.connect(OUT / 'metadata.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS provenance (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    previous = dict(db.execute('SELECT key,value FROM provenance'))
    if previous and previous.get('dataset_revision') != REVISION:
        raise ValueError('Existing metadata database has another source revision')
    db.execute('INSERT OR IGNORE INTO provenance VALUES (?,?)', ('dataset_revision', REVISION))
    db.execute('CREATE TABLE IF NOT EXISTS shards (path TEXT PRIMARY KEY, source_bytes INTEGER, source_lfs_sha256 TEXT, rows INTEGER, projection_sha256 TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS utterances (shard TEXT, row_index INTEGER, speaker_id TEXT, text_sha256 TEXT, duration REAL, samples INTEGER, state TEXT, district TEXT, lang TEXT, scenario TEXT, task_name TEXT, PRIMARY KEY(shard,row_index))')
    db.execute('CREATE INDEX IF NOT EXISTS utterance_key ON utterances(speaker_id,text_sha256)')
    db.commit()
    done = {row[0] for row in db.execute('SELECT path FROM shards')}
    started = datetime.now(timezone.utc).isoformat()

    def fetch(source):
        for attempt in range(3):
            try:
                fs = HfFileSystem(token=credential, skip_instance_cache=True)
                with fs.open('datasets/' + REPOSITORY + '@' + REVISION + '/' + source.rfilename, 'rb', block_size=65536, cache_type='none') as stream:
                    parquet = pq.ParquetFile(stream)
                    schema = parquet.schema_arrow
                    for name in COLUMNS:
                        if name not in schema.names:
                            raise ValueError('Missing inspected metadata column: ' + name)
                    rows = parquet.read(columns=COLUMNS).to_pylist()
                values = []
                for i, row in enumerate(rows):
                    for key in ['speaker_id', 'state', 'district', 'lang', 'normalized', 'text', 'verbatim', 'scenario', 'task_name']:
                        if row[key] is not None and not isinstance(row[key], str):
                            raise ValueError('Metadata string field changed: ' + key)
                    duration = row['duration']
                    if duration is not None and (not isinstance(duration, (int, float)) or not math.isfinite(duration)):
                        raise ValueError('Nonfinite/invalid source duration')
                    values.append((source.rfilename, i, row['speaker_id'], fingerprint(row['normalized'] or row['text'] or row['verbatim'] or ''),
                                   duration, row['samples'], row['state'], row['district'], row['lang'], row['scenario'], row['task_name']))
                lfs = source.lfs
                lfs_sha = lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)
                projected_sha = hashlib.sha256(json.dumps(values, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
                return source.rfilename, source.size, lfs_sha, values, projected_sha
            except Exception as error:
                message = str(error).replace(credential, '[REDACTED]') if credential else str(error)
                message = re.sub(r'https?://\S+', '[remote URL omitted]', message)
                LOG.error('Metadata projection failed %s attempt %s: %s %s', source.rfilename, attempt + 1, type(error).__name__, message)
                if attempt == 2:
                    raise RuntimeError('Metadata recovery failed for ' + source.rfilename + ': ' + type(error).__name__) from None
                time.sleep(1 + attempt)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, source) for source in files if source.rfilename not in done]
        for future in as_completed(futures):
            name, size, lfs_sha, values, projected_sha = future.result()
            with db:
                db.executemany('INSERT INTO utterances VALUES (?,?,?,?,?,?,?,?,?,?,?)', values)
                db.execute('INSERT INTO shards VALUES (?,?,?,?,?)', (name, size, lfs_sha, len(values), projected_sha))
            done.add(name)
            save('status.json', {'status': 'recovering_metadata', 'shards_complete': len(done), 'shards_planned': len(files), 'updated_utc': datetime.now(timezone.utc).isoformat()})
            LOG.info('Recovered metadata %s/%s (%s rows)', len(done), len(files), len(values))
    regions = defaultdict(list)
    for speaker, state, district, lang, count in db.execute('SELECT speaker_id,state,district,lang,COUNT(*) FROM utterances GROUP BY speaker_id,state,district,lang'):
        regions[speaker].append({'state': state, 'district': district, 'lang': lang, 'source_utterances': count})
    corpus = ROOT.parent / 'indicvoices_r_ne/groups.jsonl'
    local = Counter(); singles = []; unmatched = 0; ambiguous = 0
    for line in corpus.read_text().splitlines():
        row = json.loads(line)
        if set(row) != {'path', 'duration', 'transcript', 'source'} or not row['source'].startswith('indicvoices_r/'):
            raise ValueError('Local human corpus schema/source changed')
        speaker = row['source'].removeprefix('indicvoices_r/'); local[speaker] += 1
        matches = list(db.execute('SELECT shard,row_index,state,district,duration FROM utterances WHERE speaker_id=? AND text_sha256=? AND ABS(duration-?)<=0.05', (speaker, fingerprint(row['transcript']), row['duration'])))
        if len(matches) == 1:
            shard, index, state, district, duration = matches[0]
            singles.append({'audio': row['path'], 'speaker_id': speaker, 'text': row['transcript'], 'local_seconds': row['duration'],
                            'source_seconds': duration, 'source_shard': shard, 'source_row': index, 'state': state, 'district': district,
                            'status': 'single_original_utterance_text_duration_candidate_not_audio_checksum_identity'})
        elif matches:
            ambiguous += 1
        else:
            unmatched += 1
    records = []
    for speaker, count in sorted(local.items()):
        reported = regions.get(speaker, [])
        nonempty = {(r['state'], r['district'], r['lang']) for r in reported if r['state'] and r['district'] and r['lang'] == 'ne'}
        records.append({'speaker_id': speaker, 'local_recordings': count, 'reported_regions': reported,
                        'status': 'unique_reported_region' if len(nonempty) == 1 else 'missing_or_conflicting_region'})
    rows = db.execute('SELECT COUNT(*) FROM utterances').fetchone()[0]
    states = dict(db.execute('SELECT state,COUNT(*) FROM utterances GROUP BY state'))
    districts = [{'state': a, 'district': b, 'source_utterances': c} for a, b, c in db.execute('SELECT state,district,COUNT(*) FROM utterances GROUP BY state,district')]
    db.close()
    save('speaker_regions.json', records); save('single_utterance_candidates.json', singles)
    audit = {'creator': 'Codex GPT-6 /root', 'started_utc': started, 'completed_utc': datetime.now(timezone.utc).isoformat(),
             'status': 'human_metadata_recovered_not_accent_labels', 'repository': REPOSITORY, 'revision': REVISION,
             'shards': len(done), 'source_utterances': rows, 'local_source_groups': len(local), 'local_group_recordings': sum(local.values()),
             'local_groups_with_metadata': sum(count for speaker, count in local.items() if speaker in regions),
             'single_utterance_text_duration_candidates': len(singles), 'ambiguous_single_utterance_matches': ambiguous,
             'not_matched_as_single_utterance': unmatched, 'states': states, 'districts': districts,
             'audio_column_read': False, 'full_parquet_payload_checksum_verified': False,
             'historical_local_prep_revision_known': False, 'geography_is_an_accent_label': False,
             'first_language_residence_and_hindi_exposure': 'unknown; never inferred from state/district',
             'limits': ['Source-region fields describe the speaker region reported by the dataset, not current residence or Hindi influence.',
                        'The local staging source can concatenate utterances with fixed 250 ms gaps. Local groups are not automatically natural prosodic units.',
                        'Single-utterance matches require exact normalized text and close duration; their original audio identity is not checksum-verified.',
                        'Speaker IDs are source metadata, not independently verified real-person identity.',
                        'Metadata is fetched at a pinned current revision; the historical corpus extraction revision remains unknown.']}
    save('audit.json', audit)
    save('status.json', {'status': 'complete_metadata_only', 'shards_complete': len(done), 'shards_planned': len(files), 'updated_utc': datetime.now(timezone.utc).isoformat()})
    save('inventory.json', [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                            for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in {'inventory.json', 'acquire.log', 'pid'}])
    print(json.dumps({k: audit[k] for k in ['source_utterances', 'local_source_groups', 'local_groups_with_metadata', 'single_utterance_text_duration_candidates', 'states']}))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
    try:
        main()
    except Exception:
        LOG.exception('Human metadata recovery incomplete; do not use partial regional coverage as accent labels')
        if OUT.is_dir():
            save('status.json', {'status': 'failed_inspect_log_before_resume', 'updated_utc': datetime.now(timezone.utc).isoformat()})
        raise
