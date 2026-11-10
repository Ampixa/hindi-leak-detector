"""Pinned local dual-G2P access for arbitrary new Nepali words."""
from __future__ import annotations

from dataclasses import asdict
import importlib
import json
from pathlib import Path
import subprocess
import sys

from hindi_detector.corrected_synthesis import sha


class DualG2P:
    def __init__(self, root: Path):
        plan = root/'data/swar_phone_plan_v18'
        pins = json.loads((plan/'inputs.json').read_text())
        for row in pins['frontend_resources']:
            if sha(root/row['path']) != row['sha256']:
                raise ValueError('Pinned Nepali frontend changed: ' + row['path'])
        for row in pins['hindi_resources']:
            if sha(Path(row['path'])) != row['sha256']:
                raise ValueError('Pinned Hindi G2P resource changed')
        frontend = (root/'.sources/frontend').resolve(strict=True)
        sys.path.insert(0, str(frontend))
        try:
            self.ph = importlib.import_module('nepali_frontend.g2p.phonemizer')
        finally:
            sys.path.pop(0)
        if not Path(self.ph.__file__).resolve().is_relative_to(frontend):
            raise ValueError('Unexpected frontend already imported')
        self.variants = self.ph.data.lexicon_variants()
        self.ipa = json.loads((plan/'ipa_map.json').read_text())
        self.cache = {r['word']: r for r in json.loads((plan/'words.json').read_text())}
        self.exe = Path(pins['hindi_resources'][0]['path'])
        if self.exe.name != 'espeak-ng':
            raise ValueError('Pinned Hindi executable inventory order differs')
        self.runtime = pins['hindi_runtime']

    def word(self, word):
        if word in self.cache:
            return self.cache[word]
        native = self.ph.phonemize_word(word, style='spoken_nepali')
        result = subprocess.run([str(self.exe), '-q', '-v', 'inc/hi', '--ipa', '--sep=|', word],
                                capture_output=True, text=True, check=True)
        if not result.stdout.strip() or result.stderr.strip():
            raise RuntimeError('Hindi G2P returned empty output or unexplained warning for ' + word)
        record = {'word': word, 'nepali_default': asdict(native),
                  'nepali_project_variants': self.variants.get(word, [native.phones]),
                  'hindi_units': result.stdout.strip().split('|'), 'hindi_raw_stdout': result.stdout,
                  'g2p_is_realization_gold': False}
        self.cache[word] = record
        return record
