"""Build the listening lab from actual word-evidence receipts, never mock scores."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not (args.out / 'README.md').is_file() or (args.out / 'index.json').exists():
        raise ValueError('Need a documented new lab data directory')
    groups = defaultdict(list)
    inputs, identities = [], set()
    for directory in args.evidence:
        source = directory / 'manifest.json'
        manifest = json.loads(source.read_text())
        inputs.append({'path': str(source.resolve()), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
        for row in manifest['items']:
            if row['status'] != 'ok_unreviewed_word_evidence':
                continue
            if row['clip_id'] in identities:
                raise ValueError('Duplicate clip across word-evidence groups')
            identities.add(row['clip_id'])
            path = (directory / row['detail']).resolve(strict=True)
            if not path.is_relative_to(directory.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != row['detail_sha256']:
                raise ValueError('Word evidence identity mismatch')
            data = json.loads(path.read_text())
            if data['audio_sha256'] != row['audio_sha256']:
                raise ValueError('Audio/evidence identity mismatch')
            groups[row['sentence_id']].append(data)
    index = []
    inventory = []
    for identifier, systems in sorted(groups.items()):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', identifier):
            raise ValueError('Unsafe sentence identifier')
        texts = {s['text'] for s in systems}
        words = [[w['word'] for w in s['words']] for s in systems]
        if len(texts) != 1 or any(w != words[0] for w in words[1:]):
            raise ValueError('Matched word/text sequences disagree across systems')
        path = args.out / (identifier + '.json')
        path.write_text(json.dumps({'id': identifier, 'text': systems[0]['text'], 'systems': systems}, ensure_ascii=False, allow_nan=False) + '\n')
        index.append({'id': identifier, 'text': systems[0]['text'], 'systems': len(systems), 'words': len(words[0])})
        inventory.append({'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size})
    (args.out / 'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2) + '\n')
    inventory.append({'path': 'index.json', 'sha256': hashlib.sha256((args.out / 'index.json').read_bytes()).hexdigest(), 'bytes': (args.out / 'index.json').stat().st_size})
    (args.out / 'manifest.json').write_text(json.dumps({'creator': 'Codex GPT-6 /root',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'input_manifests': inputs,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'files': inventory}, indent=2) + '\n')
    print(json.dumps({'sentences': len(index), 'word_rows': sum(sum(len(s['words']) for s in systems) for systems in groups.values())}))


if __name__ == '__main__':
    main()
