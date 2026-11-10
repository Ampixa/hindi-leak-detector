"""Data preparation only: separate held-out features from optimizer inputs."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import time

import numpy as np

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT/'reports/vowel_controls_v19'
    out = base/'training_partitions'
    if out.exists():
        raise FileExistsError(out)
    while True:
        state = json.loads((base/'status.json').read_text())
        if state['phase'] == 'failed_explicitly':
            raise RuntimeError('Vowel control producer failed; no silent partial fallback')
        if state['phase'] == 'complete_actual_vowel_controls':
            break
        os.kill(state['pid'], 0); time.sleep(5)
    report = json.loads((base/'report.json').read_text())
    if sha(base/'arrays.npz') != report['array_sha256'] or sha(base/'index.json') != report['index_sha256']:
        raise ValueError('Completed control identity changed')
    index = json.loads((base/'index.json').read_text())
    with np.load(base/'arrays.npz', allow_pickle=False) as data:
        features, labels = data['features'], data['labels']
    if len(features) != len(index) or not np.isfinite(features).all():
        raise ValueError('Wrong control matrix geometry')
    write_group(out, {'inputs.json': {'source_arrays_sha256': report['array_sha256'],
        'source_index_sha256': report['index_sha256']}, 'source_snapshot.json': {
            'sha256': sha(Path(__file__)), 'source_utf8': Path(__file__).read_text()}},
        'Data-only partition preparation before fitting. arrays_train.npz/arrays_development.npz/arrays_test.npz '
        'contain the exact rows of their frozen source split; inventory.json pins actual bytes/SHA/counts. '
        'The optimizer reads only train/development until model and thresholds are frozen. No model fitting '
        'or selection occurs in this preparation process. These arrays duplicate the compact parent matrix '
        'for separation of held-out consumption. Current consumer vowel head fitting; review2026-11-09, '
        'retire redundant arrays after final reproduction review. No HF upload; source rights unchanged.',
        ['.venv-indicf5/bin/python -m scripts.prepare_vowel_partitions'])
    inventory = []
    for part in ('train', 'development', 'test'):
        mask = np.array([r['split'] == part for r in index])
        path = out/('arrays_'+part+'.npz')
        np.savez_compressed(path, features=features[mask], labels=labels[mask])
        inventory.append({'path': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path), 'rows': int(mask.sum())})
    (out/'inventory.json').write_text(json.dumps(inventory, indent=2)+'\n')
    print(json.dumps(inventory))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Vowel data partition preparation failed explicitly')
        raise
