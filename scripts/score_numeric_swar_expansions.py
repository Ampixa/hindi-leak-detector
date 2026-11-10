"""Score only explicitly ASR-corroborated numeric expansions; keep original text."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import subprocess
import traceback
import unicodedata

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from scripts.extract_corrected_training import read_rows

ROOT = Path(__file__).resolve().parents[1]
# These are auditable transcript repairs, never prediction rules or error labels.
# The matching conditions were chosen after reading every actual decoder receipt.
REPAIRS = {
    'real_yt_yt08_060': {'replace': {'१३': 'तेह्र'}, 'counts': {'१३': 1}, 'phrases': [(['तेह्र हप्ता'], 1)]},
    'real_yt_yt09_069': {'replace': {'३': 'तीन'}, 'counts': {'३': 2}, 'phrases': [(['तिन हजार', 'तीन हजार'], 2)]},
    'real_yt_yt10_073': {'replace': {'१२': 'बाह्र', '१५': 'पन्ध्र'}, 'counts': {'१२': 1, '१५': 1},
                          'phrases': [(['बाह्र हजार'], 1), (['पन्ध्र हजार'], 1)]},
    'real_yt_yt12_092': {'replace': {'१५': 'पन्ध्र'}, 'counts': {'१५': 1}, 'phrases': [(['पन्ध्र करोड'], 1)]},
}


def corroborated(row, repair):
    for decoder in ('zipformer_ne', 'conformer_ne'):
        text = row['hypotheses'][decoder]['text']
        for variants, count in repair['phrases']:
            if sum(text.count(variant) for variant in variants) != count:
                return False
    return True


def main():
    out = ROOT / 'reports/numeric_swar_expansions_v54'
    if out.exists():
        raise FileExistsError(out)
    pack = ROOT / 'data/swar_speech_v18'
    source = ROOT / 'reports/numeric_swar_content_v53'
    selection = {row['id']: row for row in json.loads((pack / 'selection.json').read_text())}
    audits = {(row['id'], row['kind']): row for row in read_rows(source / 'items.jsonl')}
    assets = {(row['id'], row['kind']): row for row in json.loads((pack / 'inventory.json').read_text())['assets']}
    jobs = []
    for key, repair in REPAIRS.items():
        original = selection[key]['text']
        text = original
        for old, new in repair['replace'].items():
            if original.count(old) != repair['counts'][old]:
                raise ValueError('Original numeric token occurrence changed')
            text = text.replace(old, new)
        if any(unicodedata.category(c)[0] == 'N' for c in text):
            raise ValueError('Numeric text remains after explicit expansion')
        qualified = [kind for kind in ('real', 'swar', 'a450') if corroborated(audits[key, kind], repair)]
        if not {'real', 'swar'}.issubset(qualified):
            raise ValueError('Real and Swar numbers not corroborated by both recognizers')
        jobs.append({'id': key, 'original_text': original, 'analysis_text': text, 'qualified_sources': qualified,
                     'repair': repair, 'meaning': 'ASR-corroborated numeric lexical value, not human-verified phone/error gold.'})
    write_group(out, {'plan.json': jobs, 'inputs.json': {'audit_items_sha256': sha(source / 'items.jsonl'),
        'source_selection_sha256': sha(pack / 'selection.json'), 'model_sha256': sha(ROOT / 'reports/task_vowel_release_v31/model.pt')},
        'source_snapshot.json': {'sha256': sha(Path(__file__)), 'source_utf8': Path(__file__).read_text()}},
        'Explicit four-case spoken-number expansions corroborated independently by both '
        'Nepali decoders on real and Swar audio. plan.json preserves exact original text, '
        'replacement occurrences and audit rules; six ambiguous cases remain excluded. '
        'Each id/source directory contains cold model inference result/source receipts; '
        'id_source.log child stdout/errors; completed.json final summary. Existing audio, '
        'page source text, head weights/thresholds unchanged. Comparison uses real same-text '
        'audio, not an unverified generated reference. A source without corroborated number '
        'reading is omitted explicitly. No new Hindi-error labels or accuracy estimate. '
        'Consumer broaden actual word review, owner Ampixa/user, review2026-11-10; compact '
        'metrics retained with active review, logs expire24h after closure. No HF upload; '
        'inherited rights.',
        ['.venv-indicf5/bin/python -m scripts.score_numeric_swar_expansions'])
    completed = []
    try:
        for job in jobs:
            reference = pack / assets[job['id'], 'real']['path']
            for kind in job['qualified_sources']:
                asset = assets[job['id'], kind]
                query = pack / asset['path']
                if sha(query) != asset['sha256'] or sha(reference) != assets[job['id'], 'real']['sha256']:
                    raise ValueError('Original source WAV changed')
                result_out = out / job['id'] / kind
                command = [str(ROOT / '.venv-indicf5/bin/python'), '-u', '-m', 'scripts.detect_nepali_leakage',
                    '--query', str(query), '--reference', str(reference), '--text', job['analysis_text'],
                    '--check-query-content', '--out', str(result_out)]
                with (out / (job['id'] + '_' + kind + '.log')).open('xb') as log:
                    process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    temp = out / 'status.partial'
                    temp.write_text(json.dumps({'phase': 'scoring_expanded_numeric_case', 'pid': os.getpid(),
                        'child_pid': process.pid, 'command': command, 'completed': len(completed),
                        'updated_utc': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')
                    temp.replace(out / 'status.json')
                    code = process.wait()
                if code != 0:
                    raise RuntimeError('Cold numeric expansion child failed explicitly:' + job['id'] + '/' + kind)
                result = json.loads((result_out / 'result.json').read_text())
                if result['status'] != 'scored_vowel_evidence':
                    raise ValueError('Expanded numeric query rejected content/reference check')
                completed.append({'id': job['id'], 'kind': kind, 'result': str((result_out / 'result.json').relative_to(out)),
                    'sha256': sha(result_out / 'result.json'), 'word_flags': result['word_flags'],
                    'query_content_decision': result['query_content_check']['decision']})
        report = {'status': 'complete_expanded_numeric_inference', 'cases': len(jobs), 'outputs': len(completed),
                  'results': completed, 'actual_hindi_error_accuracy': None, 'live_review_updated': False}
        for name in ('completed.json', 'status.json'):
            (out / name).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(report, ensure_ascii=False))
    except Exception:
        (out / 'status.json').write_text(json.dumps({'phase': 'failed_explicitly', 'error': traceback.format_exc()}, indent=2) + '\n')
        raise


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Explicit numeric expansion scoring failed; source texts and live review remain unchanged')
        raise
