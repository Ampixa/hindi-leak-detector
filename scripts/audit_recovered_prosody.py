"""Check whether the recovered source-trained prosody branch transfers to Swar.

This audit never promotes source probabilities into Hindi-error probabilities.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import numpy as np
import torch

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.local_features import PROSODY_NAMES
from hindi_detector.structured_detector import context, load_model
from scripts.extract_corrected_training import read_rows

ROOT = Path(__file__).resolve().parents[1]


def features(directory, row):
    path = (directory / row['detail']).resolve(strict=True)
    if not path.is_relative_to(directory.resolve()) or sha(path) != row['detail_sha256']:
        raise ValueError('Prosody detail identity differs')
    detail = json.loads(path.read_text())
    values = np.array([[word['prosody'][name] for name in PROSODY_NAMES] for word in detail['words']], np.float32)
    if values.shape != (len(detail['words']), 29) or not np.isfinite(values).all():
        raise ValueError('Unsupported retained prosody feature schema')
    return detail, values


def main():
    out = ROOT / 'reports/recovered_prosody_audit_v48'
    if out.exists():
        raise FileExistsError(out)
    model = ROOT / 'reports/corrected_training_v16/final/model.pt'
    network, state = load_model(model)
    historical = ROOT / 'reports/structured_evidence_v15'
    source = json.loads((historical / 'source_snapshot.json').read_text())
    pinned = next(row for row in source if row['path'] == 'hindi_detector/structured_detector.py')
    if sha(ROOT / pinned['path']) != pinned['sha256']:
        raise ValueError('Recovered contextual prosody construction changed')
    if sha(historical / 'runtime.json') != state['feature_runtime_sha256']:
        raise ValueError('Recovered feature runtime changed')
    # Independently confirm dictionary order against the old persisted numeric
    # matrix, rather than inferring the29 columns from a current implementation.
    old_row = next(row for row in read_rows(historical / 'items.jsonl') if row['status'] == 'ok_structured_features')
    _, old_values = features(historical, old_row)
    old_path = historical / old_row['features']
    if sha(old_path) != old_row['features_sha256']:
        raise ValueError('Historical numeric feature identity differs')
    with np.load(old_path, allow_pickle=False) as arrays:
        if not np.array_equal(old_values, arrays['prosody'].astype(np.float32)):
            raise ValueError('Current prosody dictionary order differs from recovered numeric model inputs')
    primary = ROOT / 'reports/swar_corpus_features_v18'
    teacher = ROOT / 'reports/swar_teacher_features_v21'
    queries = {(row['id'], row['kind']): row for row in read_rows(primary / 'items.jsonl')}
    references = defaultdict(list)
    for row in read_rows(teacher / 'items.jsonl'):
        references[row['text_id']].append(row)
    content = {row['id']: row['content_quality']['decision'] for row in read_rows(ROOT / 'reports/swar_teacher_content_v21/items.jsonl')}
    normalization = state['normalization']['prosody']
    if np.asarray(normalization['mean']).shape != (300,) or state['dimensions']['prosody_dimension'] != 300:
        raise ValueError('Recovered prosody normalization/context geometry differs')
    threshold = state['operating_points']['prosody']['high']
    records = []
    per_word = defaultdict(list)
    with torch.inference_mode():
        for key, rows in sorted(references.items()):
            q = {kind: features(primary, queries[key, kind]) for kind in ('real', 'swar', 'a450')}
            for reference in rows:
                rd, rv = features(teacher, reference)
                for kind, (qd, qv) in q.items():
                    if [word['word'] for word in qd['words']] != [word['word'] for word in rd['words']]:
                        raise ValueError('Same-text prosody word identity differs')
                    delta = context(qv) - context(rv)
                    x = np.concatenate([delta, np.abs(delta)], axis=1)
                    z = np.clip((x - normalization['mean']) / normalization['scale'], -8, 8).astype(np.float32)
                    scores = network.prosody_head(network.prosody(torch.from_numpy(z))).sigmoid().numpy().reshape(-1)
                    if not np.isfinite(scores).all():
                        raise ValueError('Nonfinite recovered prosody output')
                    for ordinal, (word, score) in enumerate(zip(qd['words'], scores, strict=True)):
                        record = {'id': key, 'source': kind, 'word': word['word'], 'word_ordinal': ordinal,
                            'reference_id': reference['id'], 'reference_content_decision': content[reference['id']],
                            'source_score': float(score), 'source_threshold_flag': bool(score >= threshold),
                            'duration_relative_difference': float(qv[ordinal, 0] - rv[ordinal, 0]),
                            'pitch_change_difference_st': float(qv[ordinal, 9] - rv[ordinal, 9]),
                            'actual_hindi_error_probability': None}
                        records.append(record)
                        if content[reference['id']] == 'pass':
                            per_word[key, kind, ordinal].append(record)
    consensus = []
    for (key, kind, ordinal), rows in sorted(per_word.items()):
        flags = {row['source_threshold_flag'] for row in rows}
        consensus.append({'id': key, 'source': kind, 'word_ordinal': ordinal, 'word': rows[0]['word'],
            'content_pass_references': len(rows), 'decision': 'source_shift' if flags == {True} else 'no_source_shift' if flags == {False} else 'reference_sensitive',
            'mean_source_score': float(np.mean([row['source_score'] for row in rows]))})
    paired = {(row['id'], row['source'], row['word_ordinal']): row for row in consensus}
    deltas = [row['mean_source_score'] - paired[row['id'], 'real', row['word_ordinal']]['mean_source_score']
              for row in consensus if row['source'] == 'swar']
    examples = [row for row in consensus if (row['id'], row['word']) in {
        ('real_yt_yt01_005', 'हैन'), ('real_yt_yt01_008', 'किनकि'), ('real_yt_yt03_018', 'तलको'), ('real_yt_yt05_036', 'किनकि')}]
    report = {'status': 'complete_recovered_prosody_transfer_audit', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'model_sha256': sha(model), 'branch': 'prosody_only', 'input_dimension': 300,
        'no_phonetic_or_voice_embedding_inputs': True, 'threshold': threshold,
        'decisions': dict(Counter(row['source'] + '/' + row['decision'] for row in consensus)),
        'swar_minus_real_median_source_score': float(np.median(deltas)),
        'swar_greater_than_real_fraction': float(np.mean(np.array(deltas) > 0)),
        'development_examples': examples, 'actual_hindi_error_accuracy': None,
        'promotion': 'not_promoted_source_supervision_is_not_Hindi_prosody_gold',
        'limits': ['Scores indicate similarity to old teacher/IndicF5 source supervision, never calibrated Hindi intonation errors.',
            'Recorded-human shifts against generated references expose normal/reference variation, not Native-error false-positive labels.',
            'Two reference voices and shared word alignment do not establish speaker invariance.',
            'Phone class/aspiration and discourse context must be handled before attributing pitch changes to Hindi prosody.']}
    write_group(out, {'report.json': report, 'comparisons.json': records, 'consensus.json': consensus,
        'input_inventory.json': [{'path': str(path.relative_to(ROOT)), 'sha256': sha(path)} for path in (
            model, primary / 'items.jsonl', teacher / 'items.jsonl', ROOT / 'reports/swar_teacher_content_v21/items.jsonl')],
        'source_snapshot.json': {'sha256': sha(Path(__file__)), 'source_utf8': Path(__file__).read_text()}},
        'Recover the existing corrected-source prosody branch on all actual usable same-text '
        'recordings/teacher references, with verified historical29-column numeric feature ABI '
        'and unchanged300-dimensional contextual recipe. report.json records source-style transfer '
        'and examples, comparisons.json all scores, consensus.json content-pass-reference stability. '
        'No model fitting, threshold selection, Hindi-error labels or changes to the served vowel '
        'model. Current consumer prosody-component admission decision; next review2026-11-10. '
        'Retain compact evidence while goal active; no audio/weight copies, NOT on HF, inherited rights.',
        ['.venv-indicf5/bin/python -m scripts.audit_recovered_prosody'])
    print(json.dumps({k: report[k] for k in ('decisions', 'swar_minus_real_median_source_score', 'swar_greater_than_real_fraction', 'promotion')}))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Recovered prosody transfer audit failed explicitly')
        raise
