"""Acquire only the pinned small phoneme renderer, never its training checkpoint."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = 'ampixa/real-nepali-v0.2-kala'
REVISION = '45b7d3052a0ad01d5741213ff041e65db0829f5a'
LOG = logging.getLogger(__name__)


def main() -> None:
    out = ROOT / 'models/kala_controls'
    if not (out / 'README.md').is_file() or (out / 'inventory.json').exists():
        raise ValueError('Need a documented fresh renderer group')
    with urllib.request.urlopen('https://huggingface.co/api/models/' + REPOSITORY + '/revision/' + REVISION + '?blobs=true', timeout=30) as response:
        info = json.load(response)
    if info['sha'] != REVISION or info['cardData']['license'] != 'cc-by-sa-4.0':
        raise ValueError('Inspected renderer revision/licence changed')
    remote = {r['rfilename']: r for r in info['siblings']}
    files = []
    for name, local in [('real_nepali_v02_kala.fp32.onnx', 'real_nepali_v02_kala.fp32.onnx'),
                        ('config.json', 'config.json'), ('speaker_id_map.json', 'speaker_id_map.json'),
                        ('README.md', 'MODEL_CARD.md')]:
        expected = remote[name]
        if shutil.disk_usage(out).free - expected['size'] < 20 * 1024**3:
            raise RuntimeError('Renderer acquisition would cross the20GiB data-volume floor')
        path = out / local
        if path.exists():
            raise FileExistsError('Inspect partial acquisition before retrying')
        url = 'https://huggingface.co/' + REPOSITORY + '/resolve/' + REVISION + '/' + name
        temporary = path.with_name(path.name + '.partial')
        with urllib.request.urlopen(url, timeout=90) as response, temporary.open('wb') as handle:
            while block := response.read(4 * 1024**2):
                handle.write(block)
        body = temporary.read_bytes(); digest = hashlib.sha256(body).hexdigest()
        blob = hashlib.sha1(b'blob ' + str(len(body)).encode() + b'\0' + body).hexdigest()
        if len(body) != expected['size'] or ('lfs' in expected and digest != expected['lfs']['sha256']) or ('lfs' not in expected and blob != expected['blobId']):
            raise ValueError('Pinned renderer file identity mismatch: ' + name)
        temporary.replace(path)
        files.append({'path': local, 'source_name': name, 'bytes': len(body), 'sha256': digest,
                      'publisher_identity': expected, 'source_url': url})
        LOG.info('Acquired verified renderer input %s (%s bytes)', name, len(body))
    record = {'creator': 'Codex GPT-6 /root', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'repository': REPOSITORY, 'revision': REVISION, 'license': 'CC-BY-SA4.0', 'files': files,
              'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'purpose': 'Auxiliary same-voice phoneme interventions; requested phones are not verified human error labels.',
              'training_checkpoint_downloaded': False, 'new_hf_upload': False}
    (out / 'inventory.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'verified_files': len(files), 'bytes': sum(f['bytes'] for f in files)}))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Pinned renderer acquisition incomplete; inspect retained partial files before retry')
        raise
