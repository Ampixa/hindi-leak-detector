"""Restore only the verified, retired Audio8 baseline required by this job."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import shutil
import sys

ROOT = Path('/home/ampixa/hindi-detector-wordmatrix-v3')
BENCH = Path('/home/ampixa/tts_bench_v0')
REVISION = 'f4a96a2708d651f95202ecebba025c514935cb77'
LOG = logging.getLogger(__name__)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    from huggingface_hub import HfApi, hf_hub_download

    token = sys.stdin.readline().strip() or None
    parent = ROOT / 'models'
    if not (parent / 'README.md').is_file():
        raise ValueError('Document the restored model consumer before acquisition')
    old_path = BENCH / 'receipts/audio8_acquisition_inventory.json'
    old = json.loads(old_path.read_text())
    expected = [r for r in old['files'] if r['path'].startswith('models/audio8/')]
    if len(expected) != 14 or any(Path(r['path']).parent.as_posix() != 'models/audio8' for r in expected):
        raise ValueError('Inspected baseline inventory structure changed')
    destination = parent / 'audio8'
    info = HfApi(token=token).model_info('ampixa/audio8-nepali', revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ValueError('HF revision mismatch')
    siblings = {s.rfilename: s for s in info.siblings}
    rows = []
    for item in expected:
        name = Path(item['path']).name
        if name not in siblings or siblings[name].size != item['bytes']:
            raise ValueError('Remote and original inventory disagree')
        path = destination / name
        if not path.exists():
            if shutil.disk_usage(ROOT).free - item['bytes'] < 18 * 1024**3:
                raise RuntimeError('Restoration would cross the native 18 GiB disk floor')
            path = Path(hf_hub_download('ampixa/audio8-nepali', name, revision=REVISION, local_dir=destination, token=token))
        if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            raise ValueError('Restored file differs from original benchmark bytes: ' + name)
        rows.append({'name': name, 'bytes': item['bytes'], 'sha256': item['sha256']})
        LOG.info('Restored and verified %s (%s bytes)', name, item['bytes'])
    record = {'creator': 'Codex GPT-6 /root', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'source_repo': 'ampixa/audio8-nepali', 'source_revision': REVISION,
              'original_inventory_sha256': digest(old_path), 'files': rows,
              'bytes': sum(r['bytes'] for r in rows), 'reason': 'Shared benchmark baseline directory was retired while this job waited; restore task-owned pinned inputs.',
              'driver_sha256': digest(Path(__file__))}
    (parent / 'restoration.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'files': len(rows), 'bytes': record['bytes'], 'verified': True}))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Audio8 restoration failed; no unverified replacement permitted')
        raise
