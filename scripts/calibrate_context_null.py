"""Calibrate unknown-context uncertainty on human development shams only."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from hindi_detector.artifacts import write_group
from hindi_detector.consonant_context import NonSyllabicTable
from hindi_detector.context_detector import load
from hindi_detector.corrected_synthesis import sha
from hindi_detector.dual_runtime import PhonePair
from hindi_detector.intrinsic_vowels import measure
from hindi_detector.paired_local_vowels import vector
from hindi_detector.phone_lattice import VOWELS, units_for_project
from hindi_detector.pronunciation_sites import windows
from hindi_detector.task_vowel_detector import decisions, score
from scripts.audit_context_controls import context_template, MODES
from scripts.extract_corrected_training import read_rows
from scripts.score_swar_sites import load_item

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / 'reports/context_fallback_v61b'
    if out.exists():
        raise FileExistsError(out)
    model = ROOT / 'reports/context_fallback_v61/model.pt'
    networks, original = load(model)
    controls = ROOT / 'reports/vowel_controls_v19'
    primary = ROOT / 'reports/swar_corpus_features_v18'
    secondary = ROOT / 'reports/swar_phone_crosscheck_v20'
    source = read_rows(controls / 'items.jsonl')
    rows = [row for row in source if row['split'] == 'development' and row['label'] == 0 and row['kind'] == 'vowel_deletion']
    if len(rows) != 53 or any(row['word'] in {'हैन', 'किनकि'} for row in rows):
        raise ValueError('Inspected no-change development split/count changed')
    observed = {row['audio_sha256']: row for row in read_rows(ROOT / 'reports/intrinsic_physical_v24/items.jsonl')}
    write_group(out, {'protocol.json': {'source_model_sha256': sha(model), 'null_source_items_sha256': sha(controls / 'items.jsonl'),
        'null_rows': len(rows), 'selection': 'Every old-development deletion sham; no positives or held-out test data loaded.',
        'effect': 'min(primary_relative_margin_min,secondary_relative_margin_min), the minimum query-minus-reference margin over all three pads in both probes.',
        'floor': 'Maximum no-change effect over all53 source shams and four context masks, plus.00001; deletion strong decisions below floor become uncertain.',
        'weights_and_quality_unchanged': True, 'v62_scores_opened': False,
        'labels': 'Known unchanged speech with matched2ms fade joins, never generator/voice identity or real accent gold.'},
        'source_snapshot.json': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()}
                               for p in (Path(__file__), ROOT / 'hindi_detector/context_calibrated.py')]},
        'Native-human no-change effect calibration for unknown consonant contexts. '
        'items.jsonl actual source/sham SHA,114 features, old score and cross-probe minimum '
        'effect across four context masks. model.pt keeps v61 head weights/normalizations '
        'bit-exact and adds a measured deletion uncertainty floor. No positive/test waveforms '
        'or v62 prediction scores enter calibration. frozen_selection.json model/source '
        'identity/time before new evaluation. No raw waveform/frame/encoder copies. '
        'Current consumer conservative fallback admission, review2026-11-10; compact '
        'evidence retained while consumed. Not promoted by this calibration, no Hindi-error '
        'accuracy/HF upload; source/MMS/PanPhon rights inherited.',
        ['.venv-indicf5/bin/python -m scripts.calibrate_context_null'])
    runtime = PhonePair(ROOT, original)
    table = NonSyllabicTable(ROOT / 'data/panphon_context_v58')
    ipa = json.loads((ROOT / 'data/swar_phone_plan_v18/ipa_map.json').read_text())
    lookup = [{row['id']: row for row in read_rows(directory / 'items.jsonl') if row['kind'] == 'real'} for directory in (primary, secondary)]
    cache = {}
    effects, records = [], []
    for row in rows:
        if row['key'] not in cache:
            cache[row['key']] = [load_item(directory, receipts[row['key']]) for directory, receipts in zip((primary, secondary), lookup, strict=True)]
        reference_items = cache[row['key']]
        old = observed[row['audio_sha256']]
        if old['word'] != row['word'] or old['unit_index'] != row['unit_index']:
            raise ValueError('Null control phonetic identity differs')
        original_word = reference_items[0]['detail']['words'][row['ordinal']]
        path = controls / row['audio']
        if sha(path) != row['audio_sha256']:
            raise ValueError('Known no-change control bytes changed')
        samples, rate = sf.read(path, dtype='float32')
        if rate != 16000 or samples.ndim != 1 or abs(len(samples) / 16000 - reference_items[0]['detail']['seconds']) > 1 / 16000:
            raise ValueError('Deletion sham altered utterance length or format')
        lp = runtime.encode(samples)
        for mode in MODES:
            features = []
            for probe, vocabulary in enumerate(runtime.vocabularies):
                phones = old['evidence']['native_target_phones']
                units, projections = units_for_project(phones, ipa, vocabulary)
                strict = ({'any_vowel': frozenset(vocabulary[t] for t in VOWELS if t in vocabulary)},
                          {'phones': phones, 'units': units, 'projections': projections},
                          {'kind': 'vowel_deletion', 'index': row['unit_index']})
                compiled, variant, contrast, _ = context_template(strict, ipa, vocabulary, table, mode)
                reference = reference_items[probe]
                query = {'detail': {'words': [original_word], 'phone_frame_seconds': len(samples) / 16000 / len(lp[probe])}, 'phone_log_probs': lp[probe]}
                q, qe = measure(compiled, variant, contrast, windows(query, 0))
                r, re = measure(compiled, variant, contrast, windows(reference, row['ordinal']))
                features.append(vector(q, r, qe, re, 'vowel_deletion'))
            x = np.concatenate(features)[None].astype(np.float32)
            p = score(networks, original, x)
            effect = float(min(x[0, 9], x[0, 66]))
            effects.append(effect)
            record = {'word': row['word'], 'source_case': row['key'], 'mode': mode,
                'audio_sha256': row['audio_sha256'], 'features': x[0].tolist(), 'effect': effect,
                'uncalibrated_score': float(p[0]), 'uncalibrated_decision': decisions(original, x, p)[0],
                'actual_hindi_error_label': None}
            records.append(record)
            with (out / 'items.jsonl').open('a') as f:
                f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    floor = max(0., max(effects)) + .00001
    state = copy.deepcopy(original)
    state.update(schema='context_task_vowels_v61b', null_effect_floor={'vowel_deletion': floor},
                 null_source_receipts_sha256=sha(out / 'items.jsonl'), calibrated_from_sha256=sha(model))
    state['code_sha256']['hindi_detector/context_calibrated.py'] = sha(ROOT / 'hindi_detector/context_calibrated.py')
    torch.save(state, out / 'model.pt')
    frozen = {'created_utc': datetime.now(timezone.utc).isoformat(), 'model_sha256': sha(out / 'model.pt'),
              'model_bytes': (out / 'model.pt').stat().st_size, 'null_effect_floor': floor, 'v62_scores_opened': False}
    (out / 'frozen_selection.json').write_text(json.dumps(frozen, indent=2) + '\n')
    result = {'status': 'complete_null_calibrated_context_candidate', **frozen,
        'independent_null_shams': len(rows), 'repeated_context_views': len(records),
        'uncalibrated_sham_flags': sum(record['uncalibrated_decision'] == 'strong_vowel_change' for record in records),
        'null_effect_quantiles': dict(zip(['minimum', 'median', 'p95', 'maximum'], np.quantile(effects, [0, .5, .95, 1]).tolist(), strict=True)),
        'weights_unchanged': True, 'quality_decisions_unchanged': True, 'production_changed': False,
        'actual_hindi_error_accuracy': None}
    (out / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Human no-change calibration failed explicitly; do not promote or retune held-out scores')
        raise
