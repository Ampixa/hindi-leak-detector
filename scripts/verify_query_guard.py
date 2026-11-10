"""End-to-end guard/abstention checks after the CLI summary-scope repair."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import subprocess
import sys

import numpy as np

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / 'reports/query_guard_verification_v44'
    if out.exists():
        raise FileExistsError(out)
    cases = {row['id']: row for row in json.loads((ROOT / 'reports/task_vowel_release_v31/actual_swar/cases.json').read_text())}
    requests = json.loads((ROOT / 'reports/query_guard_verification_v42/plan.json').read_text())
    requests = [{**row, 'content_guard': True} for row in requests]
    for key in ('real_yt_yt06_045', 'real_yt_yt10_074'):
        case = cases[key]
        requests.append({'name': key + '_reference_abstention',
            'query': str(ROOT / 'data/swar_speech_v18' / case['analysis']['swar']['audio']),
            'reference': None, 'text': case['text'], 'expected': 'unscored_reference_quality', 'content_guard': False})
    write_group(out, {'plan.json': requests,
        'repair.json': {'previous_attempt': 'reports/query_guard_verification_v42',
            'failure': 'Result JSON saved, but stdout summary referenced a variable local to main and exited1. Prior outputs are not counted as successful CLI executions.',
            'fix': 'Summary derives reference_mode from result dictionary; regression unit covers all3 result statuses.',
            'fixed_cli_sha256': sha(ROOT / 'scripts/detect_nepali_leakage.py')},
        'source_snapshot.json': {'sha256': sha(Path(__file__)), 'source_utf8': Path(__file__).read_text()}},
        'Final guard/abstention execution verification. Each request result has actual exit0 JSON '
        'and exact model/input/ASR/source receipts; *.log subprocess output. report.json checks '
        'known flags, unchanged phonetic features, deliberately mismatched transcript rejection '
        'and two no-content-qualified-reference abstentions. Original v42 failure retained. '
        'Current consumer runnable CLI handoff; review2026-11-10, compact conclusions retained '
        'with release, logs expire24h after consumer closes. No waveform/model copies/HF upload.',
        ['.venv-indicf5/bin/python -m scripts.verify_query_guard'])
    results = []
    for request in requests:
        directory = out / request['name']
        cmd = [sys.executable, '-m', 'scripts.detect_nepali_leakage', '--query', request['query'],
               '--text', request['text'], '--out', str(directory)]
        if request['content_guard']:
            cmd.append('--check-query-content')
        if request['reference']:
            cmd.extend(['--reference', request['reference']])
        else:
            cmd.append('--cached-only')
        with (out / (request['name'] + '.log')).open('xb') as stream:
            subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=300)
        result = json.loads((directory / 'result.json').read_text())
        passed = result['status'] == request['expected']
        delta = None
        if request['expected'] == 'flagged_vowel':
            passed = result['status'] == 'scored_vowel_evidence' and request['word'] in result['word_flags']
            batch = cases[request['name']]['analysis']['swar']['words']
            cold = result['per_reference'][0]['words']
            delta = 0.
            for a, b in zip(batch, cold, strict=True):
                if a['word'] != b['word']:
                    raise ValueError('Content guard changed canonical word identity')
                key = lambda s: (s['site']['kind'], s['site']['index'], tuple(s['native_phones']))
                av, bv = {key(s): s for s in a['sites']}, {key(s): s for s in b['sites']}
                if set(av) != set(bv):
                    raise ValueError('Content guard changed measured site coverage')
                for k in av:
                    delta = max(delta, float(np.max(np.abs(np.array(av[k]['features']) - np.array(bv[k]['features'])))))
            passed = passed and delta <= 1e-5
        elif result['word_flags'] or result['words']:
            raise ValueError('Abstention must not masquerade as a scored clean utterance')
        results.append({'name': request['name'], 'status': result['status'], 'exit_code': 0,
            'content_decision': result.get('query_content_check', {}).get('decision') if result.get('query_content_check') else None,
            'word_flags': result['word_flags'], 'max_feature_difference': delta,
            'passed': passed, 'result_sha256': sha(directory / 'result.json')})
        (out / 'partial_results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    report = {'status': 'complete_query_guard_verification', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'checks': results, 'passed': sum(row['passed'] for row in results), 'total': len(results),
        'actual_hindi_error_accuracy': None, 'model_sha256': sha(ROOT / 'reports/task_vowel_release_v31/model.pt')}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Guard verification failed explicitly; original attempts retained')
        raise
