"""Fetch pinned Allosaurus research assets without changing a live environment."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request

from hindi_detector.artifacts import write_group

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'd9f1adaf47d3b3765b41f4177da62a051516d636'
MODEL_BYTES = 41497736


def fetch(url):
    with urllib.request.urlopen(url, timeout=90) as response:
        data = response.read()
    return data


def unpack(data, destination, prefix):
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        members = archive.getmembers()
        for member in members:
            parts = Path(member.name).parts
            if not parts or parts[0] != prefix or '..' in parts or Path(member.name).is_absolute():
                raise ValueError('Unexpected archive member: ' + member.name)
            if not (member.isfile() or member.isdir()):
                raise ValueError('Refusing archive link/device: ' + member.name)
        archive.extractall(destination, filter='data')


def main():
    out = ROOT / 'runtime/independent_phone_v64'
    if out.exists():
        raise FileExistsError(out)
    if not (ROOT / '.venv-indicf5/bin/python').samefile(sys.executable):
        raise ValueError('Use verified k2 inference interpreter; do not change its packages')
    uv = Path('/Users/k2/.local/bin/uv')
    if not uv.is_file():
        raise FileNotFoundError(uv)
    write_group(out, {'protocol.json': {
        'source_revision': REVISION, 'model': 'uni2005', 'source_model_accuracy_claim': None,
        'purpose': 'Independent encoder phone evidence on physically controlled edits and Swar examples, not automatic gold.',
        'installation': 'Target-only missing dependencies; project interpreter packages unchanged; no source import from production.',
        'redistribution': 'Code GPL-3.0. Checkpoint-specific terms must be read from the release; no external redistribution.',
        'production_changed': False}},
        'Independent research phone audit. source/ contains pinned upstream source including LICENSE; '
        'models/uni2005 contains the release phone recognizer and inventory; deps/ contains only missing '
        'runtime dependencies, with dist-info licences/metadata. cache/ is the controlled uv download/build '
        'cache (disposable after successful installation). setup.log records dependency operations; '
        'downloads.json records upstream URLs/revisions, archive byte counts/hashes; inventory.json '
        'enumerates every retained file size/SHA/role. No hidden model downloader is called. '
        'Current consumer independent vowel-presence audit, owner Ampixa/user; review2026-11-10. '
        'Models/data are not uploaded to HF; this copy is k2-only. Checkpoint release-specific licence '
        'not yet verified; internal research only and no redistribution. Keep compact audit outputs; '
        'raw model/deps/cache disposable24h after this consumer closes unless it becomes a named active dependency.',
        ['.venv-indicf5/bin/python -m scripts.prepare_independent_phone_audit'])
    downloads = []
    source_url = 'https://api.github.com/repos/xinjli/allosaurus/tarball/' + REVISION
    source = fetch(source_url)
    with tarfile.open(fileobj=io.BytesIO(source), mode='r:gz') as archive:
        prefix = archive.getmembers()[0].name.split('/')[0]
    unpack(source, out / 'source', prefix)
    downloads.append({'role': 'source', 'url': source_url, 'revision': REVISION,
                      'bytes': len(source), 'sha256': hashlib.sha256(source).hexdigest(), 'archive_root': prefix})
    model_url = 'https://github.com/xinjli/allosaurus/releases/download/v1.0/uni2005.tar.gz'
    model = fetch(model_url)
    if len(model) != MODEL_BYTES:
        raise ValueError('Observed GitHub release asset byte count changed')
    unpack(model, out / 'models', 'uni2005')
    downloads.append({'role': 'phone_model', 'url': model_url, 'bytes': len(model),
                      'sha256': hashlib.sha256(model).hexdigest(), 'hf_repository': None})
    (out / 'downloads.json').write_text(json.dumps(downloads, indent=2) + '\n')
    environment = os.environ.copy()
    environment['UV_CACHE_DIR'] = str(out / 'cache')
    command = [str(uv), 'pip', 'install', '--python', sys.executable, '--target', str(out / 'deps'), '--no-deps',
               'resampy==0.4.3', 'panphon==0.22.2', 'editdistance==0.8.1', 'unicodecsv==0.14.1', 'munkres==1.1.4']
    with (out / 'setup.log').open('w') as stream:
        subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
    records = []
    for path in sorted(out.rglob('*')):
        if path.is_file() and path.name != 'inventory.json':
            records.append({'path': str(path.relative_to(out)), 'bytes': path.stat().st_size,
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'role': path.relative_to(out).parts[0]})
    (out / 'inventory.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps({'status': 'prepared', 'created_utc': datetime.now(timezone.utc).isoformat(),
                      'directory': str(out), 'source_path': str(out / 'source' / prefix),
                      'retained_bytes': sum(row['bytes'] for row in records), 'files': len(records)}))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Independent phone preparation failed; production environment unchanged')
        raise
