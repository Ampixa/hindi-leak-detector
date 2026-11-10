"""Report the frozen quality follow-up with all-row and content-pass counts."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import numpy as np

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.evaluation import binary_metrics
from scripts.extract_corrected_training import read_rows
from scripts.prepare_release_review import wilson

ROOT = Path(__file__).resolve().parents[1]


def metrics(rows):
    y = np.array([row['label'] for row in rows])
    pred = np.array([row['release']['decision'] == 'strong_vowel_change' for row in rows], float)
    if not len(y):
        return None
    m = binary_metrics(y, pred, .5)
    return {'counts': m, 'recall_interval_95_wilson': wilson(m['tp'], m['positive_labels']),
            'sham_flag_interval_95_wilson': wilson(m['fp'], m['negative_labels'])}


def main():
    out = ROOT / 'reports/quality_followup_summary_v38'
    if out.exists():
        raise FileExistsError(out)
    old = ROOT / 'reports/fresh_gap_audit_v32'
    new = ROOT / 'reports/release_quality_gap_audit_v38'
    content = ROOT / 'reports/minimal_quality_content_v38/result.json'
    plan = ROOT / 'data/minimal_quality_audit_v38/selection.json'
    intrinsic = ROOT / 'reports/intrinsic_quality_audit_v38/report.json'
    digest = sha(ROOT / 'reports/task_vowel_release_v31/model.pt')
    for directory in (old, new):
        if json.loads((directory / 'report.json').read_text())['model_sha256'] != digest:
            raise ValueError('Quality audit checkpoint differs')
    rows = [read_rows(directory / 'items.jsonl') for directory in (old, new)]
    rows = [[row for row in group if row['kind'] == 'vowel_quality'] for group in rows]
    if {row['word'] for row in rows[0]} & {row['word'] for row in rows[1]}:
        raise ValueError('Follow-up word entered earlier quality cohort')
    screen = json.loads(content.read_text())
    decisions = {row['id']: row['decision'] for row in screen['rows']}
    selection = json.loads(plan.read_text())
    if len(decisions) != len(selection['jobs']) or set(decisions) != {job['id'] for job in selection['jobs']}:
        raise ValueError('Incomplete content-screened request inventory')
    paired = defaultdict(list)
    for row in rows[1]:
        paired[row['job_id']].append(row)
    passed = []
    for job, pair in paired.items():
        if sorted(row['label'] for row in pair) != [0, 1]:
            raise ValueError('Physical audit pair incomplete')
        positive = next(row for row in pair if row['label'] == 1)
        if decisions[job] == 'pass' and decisions[positive['donor']['job_id']] == 'pass':
            passed.extend(pair)
    combined = rows[0] + rows[1]
    grouped = defaultdict(list)
    for row in combined:
        if row['label'] == 1:
            grouped[row['word']].append(row)
    # Preserve the word as the resampling unit; two voices are not independent
    # lexical families. The same conditioning groups are shared across cohorts.
    rng = np.random.default_rng(20261109)
    names = sorted(grouped)
    bootstrap = []
    for _ in range(4000):
        sample = [row for word in rng.choice(names, len(names), replace=True) for row in grouped[word]]
        bootstrap.append(np.mean([row['release']['decision'] == 'strong_vowel_change' for row in sample]))
    report = {'status': 'complete_frozen_quality_followup', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'model_sha256': digest, 'new_word_families_requested': len(selection['words']), 'new_readings': len(selection['jobs']),
        'content_decisions': dict(Counter(decisions.values())),
        'new_physical_eligible_pairs': len(rows[1]) // 2, 'new_physical_word_families': len({row['word'] for row in rows[1]}),
        'earlier_physical': metrics(rows[0]), 'new_physical_all_eligible': metrics(rows[1]),
        'new_physical_content_pass_both_source_and_donor': metrics(passed),
        'pooled_physical': metrics(combined), 'pooled_word_families': len(names),
        'pooled_recall_word_cluster_bootstrap_95': np.quantile(bootstrap, [.025, .975]).tolist(),
        'bootstrap_recipe': '4000 resamples of word families,seed20261109; shared2 conditioning groups are not new independent speaker cohorts',
        'actual_hindi_error_accuracy': None, 'model_or_threshold_updates': False,
        'limits': ['Only18 of256 new canonical readings met the full source/donor/segmentation eligibility rules;128 requested words are not128 observed-error examples.',
            'Phone-probe qualification favors clear realizations and shares acoustic biases with the detector.',
            'Known waveform replacements establish a local edit, not actual Hindi origin, naturalness or Hindi intonation.',
            'The two audits use new words/carriers but share two previously heldout conditioning groups.',
            'Zero observed sham flags does not establish zero Native-error false positives; real reference-mode variation flags remain separately reported.']}
    write_group(out, {'report.json': report,
        'input_inventory.json': [{'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'bytes': path.stat().st_size}
            for path in (old / 'items.jsonl', new / 'items.jsonl', content, plan, intrinsic, ROOT / 'reports/quality_audit_launch_v38/protocol.json')],
        'source_snapshot.json': {'sha256': sha(Path(__file__)), 'source_utf8': Path(__file__).read_text()}},
        'Compact frozen quality follow-up summary. report.json keeps earlier/new/pooled controlled '
        'counts, exact content screening, ASR-pass-both subset, word-family uncertainty and selection '
        'limits; input_inventory.json pins all original evidence. No new labels, source/model/audio '
        'copies or tuning. Current consumer morning release review; review2026-11-10. Retain compact '
        'decisions while release consumed; raw controls/logs expire24h after consumer closes. NOT '
        'on HF; inherited source/probe/MMS research restrictions apply.',
        ['.venv-indicf5/bin/python -m scripts.summarize_quality_followup'])
    print(json.dumps({k: report[k] for k in ('new_physical_eligible_pairs', 'new_physical_content_pass_both_source_and_donor', 'pooled_physical', 'pooled_word_families', 'pooled_recall_word_cluster_bootstrap_95')}))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Quality follow-up aggregation failed explicitly')
        raise
