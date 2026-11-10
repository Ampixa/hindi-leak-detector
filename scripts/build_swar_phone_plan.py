"""Fetch actual dual-G2P hypotheses for every word in the frozen Swar corpus."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import csv
import importlib
import json
import logging
from pathlib import Path
import re
import subprocess
import sys
import unicodedata

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT/'data/swar_speech_v18'
    out = ROOT/'data/swar_phone_plan_v18'
    if out.exists():
        raise FileExistsError(out)
    selection = json.loads((source/'selection.json').read_text())
    frontend = (ROOT/'.sources/frontend').resolve(strict=True)
    sys.path.insert(0, str(frontend))
    ph = importlib.import_module('nepali_frontend.g2p.phonemizer')
    if not Path(ph.__file__).resolve().is_relative_to(frontend):
        raise ValueError('Unexpected frontend import')
    variants = ph.data.lexicon_variants()
    with (frontend/'data/frontend/ipa_map.tsv').open() as f:
        rows = csv.DictReader(f, delimiter='\t')
        if rows.fieldnames != ['phone', 'ipa_candidate', 'status', 'notes']:
            raise ValueError('Actual IPA map schema changed')
        ipa = {r['phone']: r['ipa_candidate'] for r in rows}
    exe = Path('/opt/homebrew/bin/espeak-ng').resolve(strict=True)
    version = subprocess.check_output([str(exe), '--version'], text=True)
    if not version.startswith('eSpeak NG text-to-speech: 1.52.0 '):
        raise ValueError('Uninspected Hindi G2P runtime')
    match = re.search(r'Data at:\s*(.+)', version)
    if not match:
        raise ValueError('Hindi engine data location unavailable')
    data = Path(match.group(1).strip()).resolve(strict=True)
    resources = [exe, *[data/n for n in ['hi_dict', 'lang/inc/hi', 'phondata', 'phontab', 'phonindex', 'intonations']]]
    words = sorted({''.join(c for c in token if unicodedata.category(c)[0] in {'L', 'M'})
                    for row in selection for token in unicodedata.normalize('NFC', row['text']).split()})
    words = [w for w in words if w]
    records = []
    for word in words:
        native = ph.phonemize_word(word, style='spoken_nepali')
        result = subprocess.run([str(exe), '--path='+str(data.parent), '-q', '-v', 'inc/hi', '--ipa', '--sep=|', word],
                                capture_output=True, text=True, check=True)
        if not result.stdout.strip() or result.stderr.strip():
            raise ValueError('Unexplained Hindi G2P output for ' + word)
        all_native = variants.get(word, [native.phones])
        records.append({'word': word, 'nepali_default': asdict(native),
            'nepali_project_variants': all_native, 'hindi_raw_stdout': result.stdout,
            'hindi_units': result.stdout.strip().split('|'), 'g2p_is_realization_gold': False})
    paths = [*frontend.rglob('*.py'), frontend/'data/frontend/candidates_lexicon.tsv', frontend/'data/frontend/ipa_map.tsv']
    write_group(out, {'words.json': records, 'ipa_map.json': ipa,
        'inputs.json': {'created_utc': datetime.now(timezone.utc).isoformat(),
            'selection_sha256': sha(source/'selection.json'), 'words': len(records), 'hindi_runtime': version,
            'hindi_resources': [{'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size} for p in resources],
            'frontend_resources': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'bytes': p.stat().st_size} for p in paths]},
        'regressions.json': [
            {'id': 'real_yt_yt01_005', 'source': 'swar', 'word': 'हैन', 'reported_pronunciation': 'hain',
             'label_source': 'User native-listener report in this conversation', 'split': 'development_regression'},
            {'id': 'real_yt_yt01_008', 'source': 'swar', 'word': 'किनकि', 'reported_pronunciation': 'kinki',
             'label_source': 'User native-listener report in this conversation', 'split': 'development_regression'}],
        'source_snapshot.json': {'sha256': sha(Path(__file__)), 'source_utf8': Path(__file__).read_text()}},
        'Frozen generic word hypotheses for actual Swar speech. words.json preserves all actual frontend lexicon '
        'variants and Hindi engine output; IPA mappings and resource SHA are explicit. G2P predictions are not '
        'observed realization labels. regressions.json contains only the two user-confirmed examples, for '
        'development regression; no untouched-test claim. Other four picks are unlabeled, not assumed errors. '
        'No waveform/model copy or HF upload. Current consumer word/site phonetic model; review2026-11-09.',
        ['.venv-indicf5/bin/python -m scripts.build_swar_phone_plan',
         'Actual96-case Swar corpus, local verified frontend and pinned eSpeak1.52 Hindi G2P'])
    print(json.dumps({'words': len(records), 'cases': len(selection)}))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Actual Swar phone-plan creation failed explicitly')
        raise
