"""Measure fixed-component transfer; never turn source roles into error gold."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from hindi_detector.artifacts import write_group
from hindi_detector.real_word_transfer import human_reference_rank, prosody_difference

LOG = logging.getLogger(__name__)
BRANCHES = ('ctc_site', 'pooled_phone')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(values: list[float]) -> dict:
    if not values:
        return {'n': 0, 'median': None, 'p10': None, 'p90': None, 'minimum': None, 'maximum': None}
    a = np.asarray(values, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite measured value')
    return {'n': len(values), 'median': float(np.median(a)), 'p10': float(np.quantile(a, .1)),
            'p90': float(np.quantile(a, .9)), 'minimum': float(a.min()), 'maximum': float(a.max())}


def matched_style(query: dict, reference: dict) -> dict:
    if query['text'] != reference['text'] or query['word']['word'] != reference['word']['word'] or query['word']['word_index'] != reference['word']['word_index']:
        raise ValueError('Cannot compare style across mismatched text or word positions')
    if reference['system'] != 'pocket_nepali_v1c':
        raise ValueError('Preferred reference must be the user-selected Pocket checkpoint')
    return prosody_difference(query['word']['prosody'], reference['word']['prosody'])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--application', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    application = args.application
    report = json.loads((application / 'result/report.json').read_text())
    plan = json.loads((application / 'plan.json').read_text())
    status = json.loads((application / 'status.json').read_text())
    if report['failed_clips'] or status['status'] != 'real_word_application_complete_with_explicit_coverage':
        raise ValueError('Summarize only a complete application without failed clips')
    if report['models_sha256'] != plan['models_sha256'] or sha(root / 'reports/intervention_heads_v7/selection/models.json') != plan['models_sha256']:
        raise ValueError('Frozen component identity changed')
    rows = [json.loads(line) for line in (application / 'items.jsonl').read_text().splitlines()]
    if len(rows) != plan['clips'] or len(rows) != report['clips'] or len({r['clip_id'] for r in rows}) != len(rows):
        raise ValueError('Incomplete or duplicate application clip identities')
    entries, identities = [], set()
    for row in rows:
        if row['status'] != 'ok_unreviewed_real_word_application' or row['actual_hindi_error_label'] is not None:
            raise ValueError('Failed row or unexpected inferred gold')
        for word in row['words']:
            key = (row['clip_id'], word['word_index'], word['candidate_id'])
            if key in identities:
                raise ValueError('Repeated application word identity')
            identities.add(key)
            entries.append({**{k: row[k] for k in ('clip_id', 'source', 'system', 'speaker_id', 'sentence_id', 'prompt_id', 'split', 'text', 'audio_sha256')}, 'word': word})
    if len(entries) != plan['candidate_word_occurrences'] or len(entries) != report['candidate_word_occurrences']:
        raise ValueError('Actual and planned candidate-word counts differ')
    human = defaultdict(list); preferred = {}; other_voices = defaultdict(list)
    for entry in entries:
        word = entry['word']
        if entry['source'] == 'regional_humans':
            human[word['candidate_id']].append(entry)
        if entry['system'] == 'pocket_nepali_v1c':
            key = (entry['sentence_id'], word['word_index'], word['candidate_id'])
            if key in preferred:
                raise ValueError('Duplicate same-job preferred word')
            preferred[key] = entry
            if entry['prompt_id'] is not None:
                other_voices[(entry['prompt_id'], word['word_index'], word['candidate_id'])].append(entry)
    output = []; by_system = defaultdict(list); by_word = defaultdict(list)
    paired = defaultdict(list)
    for entry in entries:
        word = entry['word']; component = word['component']
        branches = {}
        for branch in BRANCHES:
            score = component['branches'][branch]
            refs = [dict(r['word']['component']['branches'][branch], speaker_id=r['speaker_id']) for r in human[word['candidate_id']]]
            branches[branch] = {'request_summary': score,
                                'human_reference': human_reference_rank(score, refs, exclude_speaker_id=entry['speaker_id'])}
        ref = preferred.get((entry['sentence_id'], word['word_index'], word['candidate_id']))
        same_job = None
        if ref is not None and ref['clip_id'] != entry['clip_id']:
            same_job = {'reference_clip_id': ref['clip_id'], 'same_reference_group': ref['speaker_id'] == entry['speaker_id'],
                        'prosody_difference': matched_style(entry, ref), 'request_score_deltas': {}}
            for branch in BRANCHES:
                q = component['branches'][branch]['baseline_request_score']
                r = ref['word']['component']['branches'][branch]['baseline_request_score']
                delta = q - r if q is not None and r is not None else None
                same_job['request_score_deltas'][branch] = delta
                if delta is not None:
                    for grouping in (entry['split'], 'all_splits'):
                        paired[(entry['system'], grouping, branch)].append(delta)
        refs = other_voices.get((entry['prompt_id'], word['word_index'], word['candidate_id']), []) if entry['prompt_id'] is not None else []
        refs = [r for r in refs if r['speaker_id'] != entry['speaker_id']]
        style = [matched_style(entry, r) for r in refs]
        cross_voice = {'reference_source_groups': len({r['speaker_id'] for r in refs}),
                       'same_full_text_required': True, 'same_source_group_excluded': True,
                       'distances': {name: describe([r[name] for r in style if r[name] is not None])
                                     for name in ('relative_word_duration_delta', 'relative_pause_after_delta', 'pitch_shape_rmse_st', 'relative_pitch_mean_delta_st', 'pitch_change_delta_st')},
                       'actual_hindi_intonation_error': None}
        item = {**{k: entry[k] for k in ('clip_id', 'system', 'source', 'speaker_id', 'sentence_id', 'split', 'audio_sha256')},
                **{k: word[k] for k in ('word', 'word_index', 'candidate_id', 'start', 'end')},
                'component_status': component['status'], 'branches': branches,
                'matched_pocket': same_job, 'other_voice_pocket_style': cross_voice,
                'actual_hindi_error_probability': None, 'native_acceptability': None}
        output.append(item); by_system[entry['system']].append(item); by_word[word['word']].append(item)

    def summarize(items):
        result = {'candidate_word_occurrences': len(items), 'branches': {}}
        for branch in BRANCHES:
            summaries = [r['branches'][branch]['request_summary'] for r in items]
            supported = [r for r in summaries if r['baseline_request_score'] is not None]
            result['branches'][branch] = {
                'baseline_supported': len(supported), 'all_seven_windows_supported': sum(r['all_windows_supported'] for r in summaries),
                'request_class_changes_across_windows': sum(r['request_class_changes'] is True for r in summaries),
                'baseline_score_at_least_half': sum(r['baseline_request_score'] >= .5 for r in supported),
                'baseline_request_scores': describe([r['baseline_request_score'] for r in supported]),
                'descriptive_human_rank_available': sum(r['branches'][branch]['human_reference']['status'] == 'descriptive_human_reference_rank_only' for r in items)}
        return result

    ordering = []
    for (system, split, branch), values in sorted(paired.items()):
        ordering.append({'system': system, 'split': split, 'branch': branch, 'available_pairs': len(values),
                         'query_request_score_above_pocket': sum(v > 0 for v in values),
                         'ties': sum(v == 0 for v in values), 'score_deltas': describe(values),
                         'is_actual_hindi_accuracy': False})
    summary = {'status': 'real_word_transfer_measured_unvalidated_error_detection',
               'clips': len(rows), 'candidate_word_occurrences': len(entries),
               'source_clip_counts': dict(Counter(r['source'] for r in rows)),
               'overall': summarize(output), 'systems': {k: summarize(v) for k, v in sorted(by_system.items())},
               'words': {k: summarize(v) for k, v in sorted(by_word.items())}, 'matched_pocket_ordering': ordering,
               'application_items_sha256': sha(application / 'items.jsonl'), 'application_plan_sha256': sha(application / 'plan.json'),
               'models_sha256': plan['models_sha256'], 'actual_hindi_accuracy': None, 'actual_prosody_accuracy': None,
               'limits': ['All scores remain synthetic request-class evidence; none is a calibrated Hindi error probability.',
                          'System roles are user-specified comparators, not gold for individual words.',
                          'Human source groups exclude the query group; repeated occurrences do not increase speaker support.',
                          'Human bank covers West Bengal districts, not verified Nepal regions, and has no acceptability/error labels.',
                          'Cross-voice style distances require identical full text; register centering is not proof of voice invariance.',
                          'Repeated words, conditions and source reference groups are dependent observations; no confidence interval assumes clip independence.',
                          'All splits were inspected descriptively; this application is not pristine held-out real-error validation.']}
    source_files = [Path(__file__), root / 'hindi_detector/real_word_transfer.py']
    write_group(args.out, {'report.json': summary, 'comparisons.json': output,
                         'source_snapshot.json': [{'path': str(p.relative_to(root)), 'sha256': sha(p), 'source_utf8': p.read_text()} for p in source_files]},
                'Measure actual-word coverage, crop sensitivity, observed human score support and text-matched Pocket style without inventing Hindi/prosody gold.',
                [str(application.resolve()), 'Application items SHA ' + sha(application / 'items.jsonl'),
                 '.venv-indicf5/bin/python -m scripts.summarize_real_word_transfer --application reports/real_word_transfer_v8 --out NEW'])
    print(json.dumps({k: summary[k] for k in ('status', 'clips', 'candidate_word_occurrences', 'overall')}))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Actual-word summary failed; no missing observations or source labels imputed')
        raise
