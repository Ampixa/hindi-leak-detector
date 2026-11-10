"""Download one pinned publisher checkpoint for a second acoustic check."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import traceback
import urllib.request

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha

ROOT = Path(__file__).resolve().parents[1]
REPO = 'facebook/wav2vec2-lv-60-espeak-cv-ft'
REVISION = 'ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4'


def main():
    out = ROOT/'models/phoneme_probe_lv60_v20'
    if out.exists():
        raise FileExistsError(out)
    api = json.load(urllib.request.urlopen('https://huggingface.co/api/models/'+REPO+'/revision/'+REVISION+'?blobs=true', timeout=30))
    if api['sha'] != REVISION:
        raise ValueError('Inspected publisher model revision changed')
    names = ['config.json','preprocessor_config.json','pytorch_model.bin','special_tokens_map.json','tokenizer_config.json','vocab.json','README.md']
    members = {r['rfilename']: r for r in api['siblings']}
    if any(n not in members for n in names):
        raise ValueError('Inspected publisher file inventory changed')
    write_group(out, {'publisher_api.json': api, 'download_source.json': {
        'sha256': sha(Path(__file__)), 'source_utf8': Path(__file__).read_text()}},
        'One pinned second acoustic checkpoint, LV60 backbone versus current XLSR53, for cross-checking actual '
        'word/site evidence. config/preprocessor/tokenizer/vocab files and pytorch_model.bin are publisher '
        'artifacts; PUBLISHER.md is the original model card. inventory.json pins every retained file, publisher '
        'LFS SHA and local bytes/SHA. download_status.json records partial/failed progress explicitly. '
        'This is a second model, not independent pronunciation gold; both checkpoints use the same research '
        'family/CommonVoice. Publisher licence apache-2.0 per API/card. HF source '+REPO+'@'+REVISION+'. '
        'Current consumer actual Swar word cross-check; status active, review2026-11-09; remove this weight '
        'after its audit closes unless adopted as a named inference dependency. No duplicate global HF cache '
        'or model re-upload. Restore from exact revision and verify LFS SHA before use.',
        ['.venv-indicf5/bin/python -m scripts.download_phone_crosscheck'])

    def status(phase, **extra):
        p = out/'download_status.json'; temp = p.with_suffix('.partial')
        temp.write_text(json.dumps({'phase': phase, 'pid': os.getpid(),
            'updated_utc': datetime.now(timezone.utc).isoformat(), **extra}, indent=2)+'\n'); temp.replace(p)

    try:
        inventory = []
        for name in names:
            info = members[name]
            path = out/('PUBLISHER.md' if name=='README.md' else name)
            temp = path.with_suffix(path.suffix+'.partial')
            digest = hashlib.sha256(); count = 0; published = info.get('lfs', {}).get('sha256')
            url = 'https://huggingface.co/'+REPO+'/resolve/'+REVISION+'/'+name
            status('downloading_pinned_publisher_file', file=name, expected_bytes=info['size'])
            with urllib.request.urlopen(url, timeout=60) as response, temp.open('xb') as target:
                while True:
                    chunk = response.read(4*1024*1024)
                    if not chunk:
                        break
                    count += len(chunk); digest.update(chunk); target.write(chunk)
                    status('downloading_pinned_publisher_file', file=name, downloaded_bytes=count, expected_bytes=info['size'])
            actual = digest.hexdigest()
            if count != info['size'] or published is not None and actual != published:
                raise ValueError('Publisher model size/LFS digest differs for '+name)
            temp.replace(path)
            inventory.append({'path': path.name, 'publisher_path': name, 'url': url,
                'bytes': count, 'sha256': actual, 'publisher_lfs_sha256': published})
        (out/'inventory.json').write_text(json.dumps({'repo': REPO, 'revision': REVISION,
            'publisher_licence': api.get('cardData',{}).get('license'), 'files': inventory}, indent=2)+'\n')
        status('complete_verified_phone_checkpoint')
    except Exception:
        status('failed_explicitly', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Pinned second acoustic download failed explicitly')
        raise
