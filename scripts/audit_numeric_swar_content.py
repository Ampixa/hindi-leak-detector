"""Decode every excluded numeric-containing triple before considering text repair."""
from __future__ import annotations

import json
import logging
from pathlib import Path
import unicodedata

from hindi_detector.artifacts import write_group
from hindi_detector.asr_gate import CpuContentGate
from hindi_detector.corrected_synthesis import sha
from scripts.calibrate_asr_gate import audio
from scripts.check_query_content import check

ROOT = Path(__file__).resolve().parents[1]


def main():
    pack = ROOT / 'data/swar_speech_v18'
    out = ROOT / 'reports/numeric_swar_content_v53'
    if out.exists():
        raise FileExistsError(out)
    inventory = json.loads((pack / 'inventory.json').read_text())
    selection = {row['id']: row for row in json.loads((pack / 'selection.json').read_text())
                 if any(unicodedata.category(c)[0] == 'N' for c in row['text'])}
    assets = [row for row in inventory['assets'] if row['id'] in selection and row['kind'] in {'real', 'swar', 'a450'}]
    if len(selection) != 10 or len(assets) != 30:
        raise ValueError('Actual excluded-case schema/count changed')
    write_group(out, {'inputs.json': {'inventory_sha256': sha(pack / 'inventory.json'),
        'selection_sha256': sha(pack / 'selection.json'), 'cases': 10, 'assets': assets},
        'source_snapshot.json': {'sha256': sha(Path(__file__)), 'source_utf8': Path(__file__).read_text()}},
        'Lexical-content audit of all10 numeric-containing Swar cases previously unscored. '
        'items.jsonl exact source WAV SHA and three actual ASR hypotheses. runtime.json '
        'model/source pins. report.json completion; no text expansion/repair or Hindi-error '
        'labels automatically inferred. Candidate spoken forms require explicit corroboration '
        'across Native and Swar readings; disagreements stay unscored, never guessed from '
        'numeric spelling. Consumer widen review coverage safely, review2026-11-10. Compact '
        'receipts retained with task; no new audio/model copies/HF upload; inherited rights.',
        ['.venv-asr-gate/bin/python -m scripts.audit_numeric_swar_content'])
    gate = CpuContentGate(ROOT, threads=2)
    (out / 'runtime.json').write_text(json.dumps(gate.runtime, indent=2) + '\n')
    for asset in assets:
        result = check(gate, audio(pack, {'audio': asset['path'], 'sha256': asset['sha256']}), selection[asset['id']]['text'])
        record = {**asset, 'original_text': selection[asset['id']]['text'], **result}
        with (out / 'items.jsonl').open('a') as f:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    report = {'status': 'complete_numeric_content_audit', 'cases': 10, 'clips': 30,
              'expanded_texts_approved': 0, 'actual_hindi_error_accuracy': None}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Numeric-case lexical audit failed explicitly; no source transcript changes')
        raise
