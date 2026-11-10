"""Measure generic vowel contrasts on actual same-text human/TTS cases."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import time
import traceback

import numpy as np

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.pronunciation_sites import (FEATURE_NAMES, compile_word, evidence_for_site,
    paired_site_features, select_reference_structure, windows)
from scripts.extract_corrected_training import read_rows

ROOT = Path(__file__).resolve().parents[1]


def load_item(group, receipt):
    values = {}
    for name in ('detail', 'features'):
        path = (group/receipt[name]).resolve(strict=True)
        if not path.is_relative_to(group.resolve()) or sha(path) != receipt[name+'_sha256']:
            raise ValueError('Actual frame-evidence hash/path changed')
        if name == 'detail':
            values['detail'] = json.loads(path.read_text())
        else:
            with np.load(path, allow_pickle=False) as arrays:
                values['phone_log_probs'] = arrays['phone_log_probs']
    values['receipt'] = receipt
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=Path, default=ROOT/'reports/swar_corpus_features_v18')
    parser.add_argument('--out', type=Path, default=ROOT/'reports/swar_sites_v18')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    plan = ROOT/'data/swar_phone_plan_v18'
    write_group(args.out, {'inputs.json': {'feature_group': str(args.features),
        'word_plan_sha256': sha(plan/'words.json'), 'ipa_map_sha256': sha(plan/'ipa_map.json'),
        'kind': 'Measured competing pronunciation evidence, not source labels or calibrated error probabilities'},
        'source_snapshot.json': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()}
            for p in (Path(__file__), ROOT/'hindi_detector/pronunciation_sites.py', ROOT/'hindi_detector/phone_lattice.py')]},
        'Actual word/site evidence. items.jsonl records every measured target/alternative path at three fixed '
        'boundary recipes, same-text human reference, native-variant/context selection, feature values and explicit '
        'unavailable reasons. arrays.npz stores18 numeric features per measured site, without lexical/voice/source '
        'identity in feature columns. report.json counts coverage only. Features have no Hindi-error labels; two '
        'user examples are development regressions. Review2026-11-09; current consumer generic pronunciation model. '
        'No new audio/model or HF upload; source rights unchanged.',
        ['.venv-indicf5/bin/python -m scripts.score_swar_sites --features GROUP --out NEW'])

    def status(phase, **extra):
        p = args.out/'status.json'
        temp = p.with_suffix('.partial')
        temp.write_text(json.dumps({'phase': phase, 'pid': os.getpid(),
            'updated_utc': datetime.now(timezone.utc).isoformat(), **extra}, indent=2)+'\n')
        temp.replace(p)

    try:
        status('waiting_for_actual_speech_features')
        while True:
            state = json.loads((args.features/'status.json').read_text())
            if state['phase'] == 'failed_explicitly':
                raise RuntimeError('Actual speech feature extraction failed')
            if state['phase'] == 'complete_actual_picks_features':
                break
            os.kill(state['pid'], 0)
            time.sleep(5)
        runtime = json.loads((args.features/'runtime.json').read_text())
        vocab = runtime['vocabulary']
        ipa = json.loads((plan/'ipa_map.json').read_text())
        candidates, unsupported = {}, {}
        for row in json.loads((plan/'words.json').read_text()):
            try:
                candidates[row['word']] = compile_word(row, ipa, vocab)
            except ValueError as error:
                unsupported[row['word']] = str(error)
        receipts = read_rows(args.features/'items.jsonl')
        lookup = {(r['id'], r['kind']): r for r in receipts}
        keys = sorted({r['id'] for r in receipts})
        speakers = {r['id']: r['voice'] for r in json.loads((ROOT/'data/swar_speech_v18/selection.json').read_text())}
        arrays, index = [], []
        counts = Counter()
        for number, key in enumerate(keys):
            if (key, 'real') not in lookup:
                raise ValueError('Same-text human reference is missing')
            reference = load_item(args.features, lookup[key, 'real'])
            queries = {kind: load_item(args.features, lookup[key, kind]) for kind in ('real', 'swar', 'a450') if (key, kind) in lookup}
            expected = [w['word'] for w in reference['detail']['words']]
            if any([w['word'] for w in q['detail']['words']] != expected for q in queries.values()):
                raise ValueError('Actual same-text aligned word identity differs')
            for ordinal, word in enumerate(expected):
                if word not in candidates:
                    counts['unsupported_word_occurrences'] += 1
                    continue
                compiled = candidates[word]
                try:
                    rcrops = windows(reference, ordinal)
                    variant, reduced = select_reference_structure(compiled, rcrops)
                except ValueError as error:
                    logging.warning('Explicit unrepresentable reference %s/%s: %s', key, word, error)
                    counts['unrepresentable_reference_occurrences'] += 1
                    continue
                for contrast in variant['contrasts']:
                    try:
                        revidence = evidence_for_site(compiled, variant, reduced, contrast, rcrops)
                        for kind, query in queries.items():
                            qevidence = evidence_for_site(compiled, variant, reduced, contrast, windows(query, ordinal))
                            features = paired_site_features(qevidence, revidence, contrast)
                            record = {'row_index': len(arrays), 'id': key, 'source': kind, 'word': word,
                                'word_ordinal': ordinal, 'word_start': query['detail']['words'][ordinal]['start'],
                                'word_end': query['detail']['words'][ordinal]['end'], 'variant_id': variant['variant_id'],
                                'native_phones': variant['phones'], 'site': {k: v for k, v in contrast.items() if k != 'alternative_unit'},
                                'query': qevidence, 'reference': revidence, 'features': features.tolist(),
                                'speaker_id': speakers[key],
                                'actual_hindi_error_label': None}
                            arrays.append(features)
                            index.append({k: record[k] for k in ('row_index', 'id', 'source', 'word', 'word_ordinal', 'site', 'speaker_id')})
                            with (args.out/'items.jsonl').open('a') as f:
                                f.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
                            counts[kind+'/'+contrast['kind']] += 1
                    except ValueError as error:
                        counts['unrepresentable_site'] += 1
                        logging.warning('Explicit site exclusion %s/%s/%s: %s', key, word, contrast['kind'], error)
            status('measuring_actual_word_sites', cases=number+1, total=len(keys), site_observations=len(arrays))
        matrix = np.stack(arrays)
        np.savez_compressed(args.out/'arrays.npz', features=matrix)
        (args.out/'index.json').write_text(json.dumps(index, ensure_ascii=False)+'\n')
        report = {'status': 'measured_actual_word_sites', 'cases': len(keys), 'feature_names': FEATURE_NAMES,
            'feature_dimension': matrix.shape[1], 'site_observations': len(matrix), 'coverage': dict(counts),
            'compiled_word_families': len(candidates), 'unsupported_word_families': unsupported,
            'array_sha256': sha(args.out/'arrays.npz'), 'index_sha256': sha(args.out/'index.json'),
            'runtime_sha256': sha(args.features/'runtime.json'), 'actual_hindi_error_accuracy': None}
        (args.out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        status('complete_actual_site_measurement', site_observations=len(arrays), cases=len(keys))
    except Exception:
        status('failed_explicitly', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Actual pronunciation-site measurement failed explicitly')
        raise
