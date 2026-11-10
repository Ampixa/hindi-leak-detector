"""Acquire the official alignment checkpoint into its documented task cache."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    directory = ROOT / 'models/mms_fa'
    if not (directory / 'README.md').is_file():
        raise ValueError('Model retention/licence README required')
    bundle = torchaudio.pipelines.MMS_FA
    if torchaudio.__version__.split('+')[0] != '2.6.0':
        raise ValueError('Acquisition expects the inspected TorchAudio 2.6.0 bundle')
    path = directory / 'model.pt'
    if path.exists():
        raise FileExistsError('Inspect existing full/partial checkpoint before acquisition')
    torch.hub.download_url_to_file(bundle._path, str(path), progress=True)
    record = {'creator': 'Codex GPT-6 /root', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'url': bundle._path, 'publisher_revision': None, 'publisher_sha256': None,
              'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'size_bytes': path.stat().st_size,
              'file': path.name, 'torch': torch.__version__, 'torchaudio': torchaudio.__version__,
              'sample_rate': bundle.sample_rate, 'dictionary': bundle.get_dict(),
              'bundle_documentation': bundle.__doc__,
              'uroman': importlib.metadata.version('uroman'), 'pyworld': importlib.metadata.version('pyworld'),
              'use': 'Alignment only; not Nepali/Hindi pronunciation truth'}
    # Verify the actual model loads from these exact bytes and no default cache.
    model = bundle.get_model(dl_kwargs={'model_dir': str(directory), 'file_name': path.name, 'weights_only': True})
    record['parameters'] = sum(p.numel() for p in model.parameters())
    del model
    (directory / 'inventory.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: record[k] for k in ['size_bytes', 'sha256', 'parameters', 'uroman', 'pyworld']}))


if __name__ == '__main__':
    main()
