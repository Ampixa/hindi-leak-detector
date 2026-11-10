"""Pin the small OpenVoice converter for controlled voice-confound experiments."""
from __future__ import annotations

import hashlib
import ast
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_REVISION = '74a1d147b17a8c3092dd5430504bd83ef6c7eb23'
MODEL_REVISION = 'f36e7edfe1684461a8343844af60babc2efbb727'


def get_json(url):
    with urllib.request.urlopen(url, timeout=40) as response:
        return json.load(response)


def git_blob(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def main() -> None:
    source, model = ROOT / '.sources/openvoice', ROOT / 'models/openvoice_v2'
    if not all((d / 'README.md').is_file() for d in [source, model]):
        raise ValueError('Source/model retention READMEs required')
    tree = get_json('https://api.github.com/repos/myshell-ai/OpenVoice/git/trees/' + CODE_REVISION + '?recursive=1')
    available = {entry['path']: entry for entry in tree['tree'] if entry['type'] == 'blob'}
    required = ['LICENSE', 'openvoice/models.py', 'openvoice/modules.py', 'openvoice/attentions.py',
                'openvoice/commons.py', 'openvoice/mel_processing.py']
    if 'openvoice/__init__.py' in available:
        required.append('openvoice/__init__.py')
    records = []
    position = 0
    while position < len(required):
        name = required[position]; position += 1
        if name not in available:
            raise ValueError('Inspected converter core path absent: ' + name)
        destination = source / name
        destination.parent.mkdir(exist_ok=True, parents=True)
        url = 'https://raw.githubusercontent.com/myshell-ai/OpenVoice/' + CODE_REVISION + '/' + name
        if destination.exists():
            data = destination.read_bytes()
        else:
            with urllib.request.urlopen(url, timeout=40) as response:
                data = response.read()
        if git_blob(data) != available[name]['sha']:
            raise ValueError('Source Git blob verification failed')
        if not destination.exists():
            destination.write_bytes(data)
        if name.endswith('.py'):
            for node in ast.walk(ast.parse(data)):
                modules = []
                if isinstance(node, ast.ImportFrom) and node.module == 'openvoice':
                    modules = ['openvoice.' + alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('openvoice.'):
                    modules = [node.module]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names if alias.name.startswith('openvoice.')]
                for module in modules:
                    dependency = module.replace('.', '/') + '.py'
                    if dependency not in available:
                        raise ValueError('Local source import needs explicit acquisition support: ' + module)
                    if dependency not in required:
                        required.append(dependency)
        records.append({'directory': 'source', 'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'git_blob': available[name]['sha']})
    info = get_json('https://huggingface.co/api/models/myshell-ai/OpenVoiceV2/revision/' + MODEL_REVISION + '?blobs=true')
    if info['sha'] != MODEL_REVISION or info.get('cardData', {}).get('license') != 'mit':
        raise ValueError('Unexpected converter revision/licence')
    files = {row['rfilename']: row for row in info['siblings']}
    for name, local in [('converter/config.json', 'config.json'), ('converter/checkpoint.pth', 'checkpoint.pth'), ('README.md', 'upstream_model_card.md')]:
        expected = files[name]; destination = model / local
        url = 'https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/' + MODEL_REVISION + '/' + name
        if not destination.exists():
            with urllib.request.urlopen(url, timeout=90) as response, destination.open('wb') as out:
                while chunk := response.read(4 * 1024 * 1024):
                    out.write(chunk)
        data = destination.read_bytes(); sha = hashlib.sha256(data).hexdigest()
        if len(data) != expected['size']:
            raise ValueError('Converter artifact size mismatch')
        if 'lfs' in expected:
            if sha != expected['lfs']['sha256']:
                raise ValueError('Converter LFS checksum mismatch')
        elif git_blob(data) != expected['blobId']:
            raise ValueError('Converter small-file Git blob mismatch')
        records.append({'directory': 'model', 'path': local, 'bytes': len(data), 'sha256': sha, 'url': url})
    inventory = {'creator': 'Codex GPT-6 /root', 'created_utc': datetime.now(timezone.utc).isoformat(),
                 'code_revision': CODE_REVISION, 'model_revision': MODEL_REVISION, 'license': 'MIT', 'files': records,
                 'purpose': 'Voice-confound counterfactuals only; never an accent-model input or human-label substitute'}
    (model / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    (source / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    print(json.dumps({'verified_members': len(records), 'checkpoint_bytes': (model / 'checkpoint.pth').stat().st_size}))


if __name__ == '__main__':
    main()
