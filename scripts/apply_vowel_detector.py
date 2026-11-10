"""Apply frozen vowel head to actual Swar picks and the remaining speech cases."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import numpy as np

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.vowel_detector import decisions, load, score
from scripts.extract_corrected_training import read_rows

ROOT = Path(__file__).resolve().parents[1]


def main():
    model = ROOT/'reports/vowel_detector_v19'
    sites = ROOT/'reports/swar_sites_v18'
    feature_group = ROOT/'reports/swar_corpus_features_v18'
    out = model/'actual_swar'
    if out.exists():
        raise FileExistsError(out)
    report = json.loads((model/'report.json').read_text())
    if sha(model/'model.pt') != report['model_sha256']:
        raise ValueError('Selected model identity changed')
    network, state = load(model/'model.pt')
    for path, key in [(ROOT/'data/swar_phone_plan_v18/words.json', 'phone_plan_sha256'),
                      (ROOT/'hindi_detector/phone_lattice.py', 'projection_code_sha256'),
                      (ROOT/'hindi_detector/pronunciation_sites.py', 'site_code_sha256')]:
        if sha(path) != state[key]:
            raise ValueError('Frozen site-model dependency changed: ' + key)
    sr = json.loads((sites/'report.json').read_text())
    if sha(sites/'arrays.npz') != sr['array_sha256'] or sha(sites/'index.json') != sr['index_sha256']:
        raise ValueError('Actual site matrix identity changed')
    if json.loads((ROOT/'reports/vowel_controls_v19/runtime.json').read_text())['phone'] != json.loads((feature_group/'runtime.json').read_text())['phone']:
        raise ValueError('Control and actual-speech acoustic encoders differ')
    with np.load(sites/'arrays.npz', allow_pickle=False) as data:
        values = data['features']
    records = read_rows(sites/'items.jsonl')
    if len(records) != len(values) or any(r['row_index'] != i for i, r in enumerate(records)):
        raise ValueError('Actual site row ordering differs')
    probabilities = score(network, state, values)
    result_decisions = decisions(state, values, probabilities)
    by_case = defaultdict(list)
    for row, probability, decision in zip(records, probabilities, result_decisions, strict=True):
        by_case[row['id'], row['source']].append({**row, 'controlled_vowel_change_score': float(probability),
            'decision': decision, 'actual_hindi_error_probability': None,
            'interpretation': 'Hindi-compatible vowel contrast' if decision == 'strong_vowel_change' else 'No strong verified contrast decision'})
    selection = json.loads((ROOT/'data/swar_speech_v18/selection.json').read_text())
    receipts = {(r['id'], r['kind']): r for r in read_rows(feature_group/'items.jsonl')}
    cases = []
    for row in selection:
        outputs = {}
        for source in ('real', 'swar', 'a450'):
            receipt = receipts.get((row['id'], source))
            if receipt is None:
                continue
            dp = feature_group/receipt['detail']
            if sha(dp) != receipt['detail_sha256']:
                raise ValueError('Actual word detail pin differs')
            detail = json.loads(dp.read_text())
            sites_by_word = defaultdict(list)
            for r in by_case[row['id'], source]:
                sites_by_word[r['word_ordinal']].append(r)
            words = []
            for ordinal, w in enumerate(detail['words']):
                observed = sites_by_word[ordinal]
                words.append({**w, 'sites': observed,
                    'decision': 'strong_vowel_change' if any(s['decision']=='strong_vowel_change' for s in observed)
                                else 'uncertain' if any(s['decision']=='uncertain' for s in observed)
                                else 'no_strong_vowel_change' if observed else 'untested_site',
                    'actual_hindi_error_probability': None})
            outputs[source] = {'audio': receipt['path'], 'audio_sha256': receipt['sha256'],
                'words': words, 'seconds': detail['seconds'], 'tracks': detail['tracks']}
        cases.append({**row, 'analysis': outputs,
            'coverage': 'scored' if outputs else 'unscored_numeric_transcript_needs_verification'})
    regressions = []
    for label in json.loads((ROOT/'data/swar_phone_plan_v18/regressions.json').read_text()):
        case = next(r for r in cases if r['id'] == label['id'])
        words = [w for w in case['analysis'][label['source']]['words'] if w['word'] == label['word']]
        regressions.append({**label, 'found': bool(words),
            'flagged': any(w['decision']=='strong_vowel_change' for w in words),
            'site_decisions': [{'kind': s['site']['kind'], 'score': s['controlled_vowel_change_score'],
                               'decision': s['decision'], 'mean_relative_margin': s['features'][8]} for w in words for s in w['sites']]})
    counts = Counter((source, w['decision']) for case in cases for source, reading in case['analysis'].items() for w in reading['words'])
    summary = {'created_utc': datetime.now(timezone.utc).isoformat(), 'model_sha256': report['model_sha256'],
        'cases_planned': len(cases), 'cases_scored': sum(bool(c['analysis']) for c in cases),
        'user_picks_scored': sum(c['user_pick'] and bool(c['analysis']) for c in cases),
        'regressions': regressions, 'regressions_passed': sum(r['flagged'] for r in regressions),
        'word_decisions': {'/'.join(k): v for k,v in counts.items()},
        'actual_hindi_error_accuracy': None, 'user_examples_are_development_regressions': True,
        'limits': ['Only two user-confirmed real errors; other actual-word flags remain candidate errors.',
                  'Scores are calibrated on physical vowel changes, not calibrated Hindi-error probabilities.',
                  'Real versus itself cannot establish a native false-positive rate; independent intact/sham controls are reported separately.',
                  'No Hindi-specific intonation correctness benchmark supplied.']}
    write_group(out, {'report.json': summary, 'cases.json': cases, 'source_snapshot.json': {
        'sha256': sha(Path(__file__)), 'source_utf8': Path(__file__).read_text()}},
        'Frozen-model actual Swar application after controlled held-out evaluation. cases.json contains every '
        'actual word interval, measured phonetic sequence, site evidence/decision and verified audio identity; '
        'unscored numeric cases and untested words remain explicit. report.json records the two user development '
        'regressions and all source decision counts, without assigning gold labels to the remaining picks. '
        'No audio copy or HF upload. Current consumer morning word-level user review; review2026-11-09. '
        'Retain compact actual results and selected model; source rights and MMS research restriction apply.',
        ['.venv-indicf5/bin/python -m scripts.apply_vowel_detector',
         'Frozen model/control evaluation; actual same-text human/Swar/Audio8 audio pins'])
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Actual Swar vowel model application failed explicitly')
        raise
