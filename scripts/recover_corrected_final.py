"""Finish the inspected source-control run without treating it as accent gold."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import subprocess
import traceback

import numpy as np

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from scripts.audit_native_variation import score_cached_audit
from scripts.extract_corrected_training import read_rows
from scripts.finalize_corrected_training import main as finalize
from scripts.train_corrected_detector import complete_groups, fit

ROOT = Path(__file__).resolve().parents[1]


def put_status(directory, phase, **extra):
    p = directory / 'status.json'
    temp = p.with_suffix('.partial')
    temp.write_text(json.dumps({'phase': phase, 'pid': os.getpid(),
        'updated_utc': datetime.now(timezone.utc).isoformat(), **extra}, indent=2) + '\n')
    temp.replace(p)


def main():
    training = ROOT / 'reports/corrected_training_v16'
    base = ROOT / 'reports/structured_evidence_v15'
    native = ROOT / 'reports/native_variation_audit_v16'
    out = training / 'recovery_v17'
    previous = {str(p.relative_to(ROOT)): json.loads(p.read_text()) for p in
        [training/'status.json', base/'status.json', native/'status.json', training/'finalizer_status.json']}
    if any(s['phase'] != 'failed_explicitly' for s in previous.values()):
        raise ValueError('Only the inspected failed pipeline may recover')
    if (training/'final').exists():
        raise FileExistsError('Final fit already exists; never overwrite reserved evaluation')
    sources = [Path(__file__), ROOT/'scripts/extract_corrected_training.py',
               ROOT/'hindi_detector/splice_quality.py', ROOT/'scripts/audit_native_variation.py',
               ROOT/'scripts/finalize_corrected_training.py']
    write_group(out, {'previous_failure_states.json': previous, 'source_snapshot.json': [
        {'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()} for p in sources]},
        'Recovery of inspected zero-context splice boundary, with previous exact failure states preserved. '
        'Existing good features and seven completed CPU groups are reused under checksum validation. '
        'Only unseen final text groups continue; expected unusable splice boundaries are explicitly quarantined. '
        'No fitting settings changed, no reserved scores had previously been produced. Source-control metrics '
        'are auxiliary to the user actual Swar pronunciation task. launch/log live under training group. '
        'Review2026-11-09; keep compact failure/recovery decisions and selected final model; logs disposable24h after review.',
        ['.venv-indicf5/bin/python -m scripts.recover_corrected_final',
         'Original24-layer teacher/IndicF5 source and encoder/feature pins unchanged'])
    try:
        pins = json.loads((base/'inputs.json').read_text())
        for path, key in [(ROOT/'data/teacher_indicf5_v14/model_pair.json', 'model_pair_sha256'),
                          (ROOT/'data/teacher_indicf5_v14/selection.json', 'plan_sha256'),
                          (ROOT/'data/pocket_bulk_plan_v11/g2p_priors.json', 'prior_sha256'),
                          (ROOT/'data/schwa_candidates_v1/candidates.json', 'candidate_sha256'),
                          (ROOT/'.sources/frontend/data/frontend/ipa_map.tsv', 'ipa_map_sha256')]:
            if sha(path) != pins[key]:
                raise ValueError('Original frozen input changed: ' + key)
        put_status(training, 'recovering_inspected_boundary_failure')
        subprocess.run([str(ROOT/'.venv-indicf5/bin/python'), '-u', '-m',
            'scripts.extract_corrected_training', '--out', str(base), '--resume-failed'], cwd=ROOT, check=True)
        if json.loads((base/'status.json').read_text())['phase'] != 'complete_corrected_structured_evidence':
            raise RuntimeError('Resumed extraction is not complete')
        partitions = complete_groups(base)
        recovered = defaultdict(list)
        for row in read_rows(base/'cpu_recovery/items.jsonl'):
            recovered[row['text_id']].append(row)
        for tid, rows in recovered.items():
            if len(rows) != 6 or sum(r['kind'] == 'physical' for r in rows) != 2:
                raise ValueError('Incomplete cached whole-unit CPU recovery')
            if any(r['status'] != 'ok_structured_features' for r in rows):
                raise ValueError('Failed recovered feature reached final fit')
            partitions[rows[0]['row']['split']][tid] = rows
        counts = {p: len(v) for p, v in partitions.items()}
        contrast = {r['word'] for r in json.loads((ROOT/'data/pocket_bulk_plan_v11/g2p_priors.json').read_text())}
        put_status(training, 'fitting_final_corrected_candidate', partition_texts=counts)
        report = fit(base, training/'final', partitions, contrast, final=True, pairwise=True)
        quality = {'recovered_texts': len(recovered), 'recovered_ids': sorted(recovered),
            'original_feature_errors_retained': True, 'encoder_policy': 'Each recovered quartet and physical pair entirely CPU; other groups original MPS.',
            'boundary_quarantines': read_rows(base/'text_exclusions.jsonl'),
            'physical_content_quarantines': len({r['text_id'] for r in read_rows(base/'physical_items.jsonl') if r['feature_status'] == 'pair_content_quarantined'}),
            'final_model_sha256': report['model_sha256'], 'previous_failures': str(out/'previous_failure_states.json')}
        (training/'final/data_quality_addendum.json').write_text(json.dumps(quality, indent=2) + '\n')
        put_status(training, 'complete_corrected_training_and_reserved_evaluation', partition_texts=counts,
                   model_sha256=report['model_sha256'])
        put_status(native, 'scoring_frozen_final_model_from_verified_native_cache')
        requests = {r['clip_id']: r for r in read_rows(native/'content_items.jsonl')}
        cache = {}
        for receipt in read_rows(native/'feature_items.jsonl'):
            if receipt['status'] != 'ok':
                continue
            row = requests[receipt['clip_id']]
            if row['content_quality']['decision'] != 'pass' or row['sha256'] != receipt['source_audio_sha256']:
                raise ValueError('Native cache source/content pin differs')
            paths = [native/'features'/receipt[k] for k in ('detail', 'features')]
            if any(sha(p) != receipt[k+'_sha256'] for k, p in zip(('detail', 'features'), paths)):
                raise ValueError('Native cached feature hash differs')
            with np.load(paths[1], allow_pickle=False) as arrays:
                values = {k: arrays[k] for k in ('phonetics', 'prosody', 'temporal_phone')}
            if any(not np.isfinite(v).all() for v in values.values()):
                raise ValueError('Nonfinite cached native arrays')
            cache[row['clip_id']] = {'row': row, 'detail': json.loads(paths[0].read_text()), **values}
        contexts = {c['id']: c for c in json.loads((ROOT/'data/native_variation_v16/selection.json').read_text())['contexts']}
        variation = score_cached_audit(native, contexts, cache, training/'final')
        put_status(native, 'complete_native_variation_audit', human_recordings=variation['human_recordings_scored'])
        finalize(resume_failed=True)
        (out/'report.json').write_text(json.dumps({'status': 'complete_recovered_source_control_run',
            'model_sha256': report['model_sha256'], 'partition_texts': counts,
            'actual_hindi_error_accuracy': None, 'auxiliary_only': True}, indent=2) + '\n')
    except Exception:
        put_status(training, 'failed_explicitly', recovery_error=traceback.format_exc())
        raise


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Inspected source-control recovery failed explicitly')
        raise
