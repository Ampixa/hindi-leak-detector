"""Recover selected original utterances by Parquet row group; retain no full shards."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
from pathlib import Path
import shutil
import re
import traceback
import sqlite3
import subprocess
import sys

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfFileSystem

from scripts.recover_human_regions import fingerprint

ROOT = Path(__file__).resolve().parents[1]
REVISION = '5f4495c91d500742a58d1be2ab07d77f73c0acf8'
REPOSITORY = 'ai4bharat/indicvoices_r'
LOG = logging.getLogger(__name__)
OUTPUT_PATH = None


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    global OUTPUT_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-row-groups', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    if args.max_row_groups < 1 or not (out / 'README.md').is_file() or ((out / 'status.json').exists() and not args.resume):
        raise ValueError('Need a documented fresh output directory and positive row-group limit')
    if shutil.disk_usage(out).free < 20 * 1024**3:
        raise RuntimeError('Keep at least 20 GiB free on the data volume')
    OUTPUT_PATH = out
    credential = sys.stdin.readline().strip() or None
    bank = ROOT / 'data/human_g2p_bank_v5/selection.json'
    selection = json.loads(bank.read_text())
    clips = {r['audio']: r for r in selection['clips']}
    candidates = json.loads((ROOT / 'data/human_regions_v5/single_utterance_candidates.json').read_text())
    selected = [r for r in candidates if r['audio'] in clips]
    if len(selected) != len(clips) or len({r['audio'] for r in selected}) != len(selected):
        raise ValueError('Each selected recording must have one original source row')
    by_shard = defaultdict(list)
    for row in selected:
        if sha(Path(row['audio'])) != clips[row['audio']]['sha256']:
            raise ValueError('Staged human waveform changed')
        by_shard[row['source_shard']].append(row)
    metadata = ROOT / 'data/human_regions_v5/metadata.sqlite'
    with sqlite3.connect('file:' + str(metadata) + '?mode=ro', uri=True) as db:
        pin = db.execute('SELECT value FROM provenance WHERE key=?', ('dataset_revision',)).fetchone()
        if pin != (REVISION,):
            raise ValueError('Metadata source revision mismatch')
        shard_identities = {name: {'source_bytes': size, 'source_lfs_sha256': digest} for name, size, digest in db.execute('SELECT path,source_bytes,source_lfs_sha256 FROM shards')}
    audio = out / 'audio'; audio.mkdir(exist_ok=args.resume)
    rows = json.loads((out / 'recordings.json').read_text()) if args.resume and (out / 'recordings.json').exists() else []
    groups = json.loads((out / 'row_groups.json').read_text()) if args.resume and (out / 'row_groups.json').exists() else []
    for row in rows:
        path = (out / row['original_encoded_audio']).resolve(strict=True)
        if not path.is_relative_to(audio.resolve()) or sha(path) != row['original_encoded_sha256'] or row['source_revision'] != REVISION:
            raise ValueError('Previously recovered original payload changed')
        if clips[row['staged_audio']]['sha256'] != row['staged_sha256']:
            raise ValueError('Previously verified staged recording changed')
    recovered = {row['clip_id'] for row in rows}
    if len(recovered) != len(rows):
        raise ValueError('Duplicate previously recovered clip')

    def save(name: str, value: object) -> None:
        path = out / name; temporary = path.with_name(path.name + '.partial')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)

    def status(phase: str) -> None:
        save('status.json', {'status': phase, 'updated_utc': datetime.now(timezone.utc).isoformat(),
                            'original_recordings_recovered': len(rows), 'selected_recordings': len(selected),
                            'audio_row_groups_read': len(groups), 'row_group_limit': args.max_row_groups})

    snapshots = [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()}
                 for p in (Path(__file__), ROOT / 'scripts/recover_human_regions.py', ROOT / 'hindi_detector/wave_identity.py', ROOT / 'scripts/reproduce_human_staging.py')]
    if args.resume and json.loads((out / 'source_snapshot.json').read_text()) != snapshots:
        raise ValueError('Recovery source changed; inspect before resuming')
    save('source_snapshot.json', snapshots)
    plan = {'selection_sha256': sha(bank), 'source_repo': REPOSITORY, 'source_revision': REVISION,
            'candidates': selected}
    if args.resume and json.loads((out / 'plan.json').read_text()) != plan:
        raise ValueError('Recovery input selection changed')
    save('plan.json', plan)
    status('recovering_original_utterance_audio')
    fs = HfFileSystem(token=credential, skip_instance_cache=True)
    for shard, requested in sorted(by_shard.items()):
        if len(groups) >= args.max_row_groups:
            break
        url = 'datasets/' + REPOSITORY + '@' + REVISION + '/' + shard
        with fs.open(url, 'rb', block_size=65536, cache_type='none') as stream:
            parquet = pq.ParquetFile(stream)
            needed = {'audio', 'speaker_id', 'normalized', 'text', 'verbatim', 'duration', 'state', 'district'}
            if not needed <= set(parquet.schema_arrow.names):
                raise ValueError('Original Parquet schema changed')
            offset = 0
            for group_index in range(parquet.num_row_groups):
                group = parquet.metadata.row_group(group_index)
                matching = [r for r in requested if offset <= r['source_row'] < offset + group.num_rows
                            and clips[r['audio']]['clip_id'] not in recovered]
                if matching and len(groups) < args.max_row_groups:
                    compressed = sum(group.column(j).total_compressed_size for j in range(group.num_columns)
                                     if group.column(j).path_in_schema.startswith('audio.'))
                    if compressed > 512 * 1024**2:
                        raise ValueError('Audio row group exceeds the inspected memory bound')
                    LOG.info('Reading %s group %s (%s compressed bytes, %s selected recordings)', shard, group_index, compressed, len(matching))
                    table = parquet.read_row_group(group_index, columns=sorted(needed))
                    for candidate in matching:
                        row = table.slice(candidate['source_row'] - offset, 1).to_pylist()[0]
                        text = row['normalized'] or row['text'] or row['verbatim'] or ''
                        if row['speaker_id'] != candidate['speaker_id'] or fingerprint(text) != fingerprint(candidate['text']):
                            raise ValueError('Original source row no longer matches selected speaker/text')
                        if (row['state'], row['district']) != (candidate['state'], candidate['district']):
                            raise ValueError('Original region metadata changed')
                        field = row['audio']
                        if not isinstance(field, dict) or not isinstance(field.get('bytes'), bytes):
                            raise ValueError('Expected inspected embedded original audio bytes')
                        payload = field['bytes']; info = sf.info(io.BytesIO(payload))
                        identifier = clips[candidate['audio']]['clip_id']
                        destination = audio / (identifier + '.source')
                        destination.write_bytes(payload)
                        if info.channels not in {1, 2} or info.samplerate <= 0:
                            raise ValueError('Original audio differs from inspected mono/stereo containers; raw payload retained for review')
                        # Keep original channels untouched. Reproduce the inspected
                        # conversion in its existing native runtime as a separate check.
                        result = subprocess.run(['/Users/k2/madhyam/pocket-tts/.venv/bin/python', '-m',
                                                 'scripts.reproduce_human_staging', '--original', str(destination),
                                                 '--staged', candidate['audio']], cwd=ROOT,
                                                capture_output=True, text=True)
                        if result.returncode != 0:
                            raise RuntimeError('Native staging reproduction failed: ' + result.stderr[-3000:])
                        reproduction = json.loads(result.stdout)
                        record = {'clip_id': identifier, 'source_repo': REPOSITORY, 'source_revision': REVISION,
                                  'source_shard': shard, 'source_row': candidate['source_row'], 'row_group': group_index,
                                  'source_shard_identity': shard_identities[shard], 'full_parquet_hash_verified': False,
                                  'original_encoded_audio': str(destination.relative_to(out)),
                                  'original_encoded_bytes': len(payload), 'original_encoded_sha256': hashlib.sha256(payload).hexdigest(),
                                  'original_audio_format': info.format, 'original_audio_subtype': info.subtype,
                                  'original_channels': info.channels, 'original_sample_rate': info.samplerate,
                                  'source_audio_path_field': field.get('path'), 'speaker_id': row['speaker_id'],
                                  'text': text, 'reported_state': row['state'], 'reported_district': row['district'],
                                  'staged_audio': candidate['audio'], 'staged_sha256': sha(Path(candidate['audio'])),
                                  'recipe_reproduction': reproduction,
                                  'native_acceptability_label': None, 'hindi_leakage_label': None}
                        rows.append(record)
                        recovered.add(identifier)
                        save('recordings.json', rows)
                    groups.append({'shard': shard, 'row_group': group_index, 'rows': group.num_rows,
                                   'compressed_audio_column_bytes': compressed, 'selected_recordings': len(matching)})
                    save('row_groups.json', groups); status('recovering_original_utterance_audio')
                    del table
                offset += group.num_rows
    report = {'status': 'selected_original_audio_recovered_not_pronunciation_labels',
              'source_repo': REPOSITORY, 'source_revision': REVISION, 'recovered_recordings': len(rows),
              'planned_recordings': len(selected), 'all_selected_recovered': len(rows) == len(selected),
              'row_groups_read': len(groups), 'compressed_audio_column_bytes_read': sum(r['compressed_audio_column_bytes'] for r in groups),
              'retained_original_audio_bytes': sum(r['original_encoded_bytes'] for r in rows),
              'byte_identical_staging_reproductions': sum(r['recipe_reproduction']['encoded_staged_bytes_identical'] for r in rows),
              'native_acceptability_accuracy': None, 'hindi_leakage_accuracy': None,
              'limits': ['Only selected embedded audio payloads are retained, never full Parquet shards or unrelated recordings.',
                         'Encoded source payload SHA-256 is measured; whole source-shard hash is publisher metadata, not independently downloaded/verified.',
                         'Waveform correspondence does not establish accent, native acceptability or real-person identity.',
                         'All current verified source regions are West Bengal; Nepal-region coverage remains unverified.']}
    save('report.json', report); status('original_audio_recovery_complete_with_explicit_coverage')
    save('inventory.json', [{'path': str(p.relative_to(out)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                            for p in sorted(out.rglob('*')) if p.is_file() and p.name not in {'inventory.json', 'worker.log', 'pid'}])
    print(json.dumps(report))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
    try:
        main()
    except Exception as error:
        # Avoid credentials and signed URLs in remote exception strings.
        message = re.sub(r'https?://\S+', '[remote URL omitted]', str(error))
        message = re.sub(r'hf_[A-Za-z0-9]+', '[credential omitted]', message)
        frames = [{'file': f.filename, 'line': f.lineno, 'function': f.name} for f in traceback.extract_tb(error.__traceback__)]
        LOG.error('Original-audio recovery stopped: %s: %s; frames=%s', type(error).__name__, message, frames)
        if OUTPUT_PATH is not None:
            (OUTPUT_PATH / 'failure.json').write_text(json.dumps({'type': type(error).__name__, 'message': message, 'frames': frames, 'updated_utc': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')
            (OUTPUT_PATH / 'failure_inventory.json').write_text(json.dumps([
                {'path': str(p.relative_to(OUTPUT_PATH)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                for p in sorted(OUTPUT_PATH.rglob('*')) if p.is_file() and p.name not in {'worker.log', 'failure_inventory.json'}], indent=2) + '\n')
        raise SystemExit(1) from None
