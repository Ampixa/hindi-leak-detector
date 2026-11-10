"""One frozen held-out physical-control evaluation; never selects a model."""
from __future__ import annotations

from collections import Counter
import json
import logging
from pathlib import Path

import numpy as np

from hindi_detector.anchored_detector import load, score
from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from scripts.extract_corrected_training import read_rows

ROOT = Path(__file__).resolve().parents[1]
POSITIVE = {'P1', 'P1w', 'P1u'}
FAMILIES = POSITIVE | {'S_original', 'S0', 'S1', 'S_world', 'S2'}


def family_summary(rows):
    """Keep abstentions in the denominator, including positive misses."""
    if not rows:
        raise ValueError('Cannot summarize an absent control family')
    measured = [r for r in rows if r['status'] == 'measured']
    flagged = sum(r['candidate'] for r in measured)
    negative_decisions = len(measured) - flagged
    positive = bool(rows[0]['label'])
    if any(bool(r['label']) != positive for r in rows):
        raise ValueError('Mixed labels within a physical control family')
    return {'attempted': len(rows), 'measured': len(measured), 'abstained': len(rows) - len(measured),
            'flagged': flagged, 'flag_rate': flagged / len(rows),
            'abstention_rate': 1 - len(measured) / len(rows),
            'correct_decision_rate': (flagged if positive else negative_decisions) / len(rows),
            'recall': flagged / len(rows) if positive else None,
            'false_flag_rate': None if positive else flagged / len(rows)}


def cluster_interval(rows):
    groups = sorted({r['phone_collision_group'] for r in rows})
    totals = np.array([sum(r['phone_collision_group'] == g for r in rows) for g in groups])
    flags = np.array([sum(r['phone_collision_group'] == g and r['status'] == 'measured' and r['candidate'] for r in rows) for g in groups])
    rng = np.random.default_rng(7101)
    choices = rng.integers(len(groups), size=(1000, len(groups)))
    values = flags[choices].sum(1) / totals[choices].sum(1)
    return {'unit': 'exact phonetic-collision group; all conditioning voices and source variants resampled together',
            'groups': len(groups), 'seed': 7101, 'replicates': 1000,
            'flag_rate_95_percentile_interval': np.quantile(values, [.025, .975]).tolist()}


def annotated(row, probability, candidate):
    result = {k: row[k] for k in ('pair_id', 'job_id', 'word_id', 'word', 'voice', 'phone_collision_group', 'split', 'family', 'label', 'status')}
    result.update(score=probability, candidate=candidate)
    if row['status'] != 'measured':
        result['reason'] = row.get('reason', 'No measured features')
    return result


def main():
    out = ROOT / 'reports/anchored_test_v74'
    if out.exists():
        raise FileExistsError('Do not reopen/overwrite the frozen test: ' + str(out))
    fit_dir = ROOT / 'reports/anchored_training_v71'
    freeze = json.loads((fit_dir / 'frozen_selection.json').read_text())
    external = json.loads((fit_dir / 'external_freeze.json').read_text())
    if not external['verified'] or any(sha(fit_dir / name) != digest for name, digest in external['files_sha256'].items()):
        raise ValueError('External pre-test freeze differs')
    if sha(fit_dir / 'model.pt') != freeze['model_sha256'] or sha(fit_dir / 'protocol.json') != freeze['protocol_sha256']:
        raise ValueError('Frozen model/protocol identity changed')
    network, state = load(fit_dir / 'model.pt')
    controls = ROOT / 'reports/native_main_controls_test_v71'
    report = json.loads((controls / 'report.json').read_text())
    if report['status'] != 'complete_reviewed_local_controls' or sha(controls / 'items.jsonl') != report['items_sha256']:
        raise ValueError('Incomplete/changed held-out controls')
    runtime = json.loads((controls / 'runtime.json').read_text())
    training_runtime = json.loads((ROOT / 'reports/native_main_controls_train_v71/runtime.json').read_text())
    if runtime != training_runtime:
        raise ValueError('Held-out feature runtime differs from training')
    content_path = ROOT / 'reports/native_main_content_test_v71/result.json'
    content = json.loads(content_path.read_text())
    sources = {r['id']: r for r in content['rows']}
    accepted = set(content['accepted_ids'])
    selection = json.loads((ROOT / 'data/native_main_v71/selection.json').read_text())
    planned = {j['id']: j for j in selection['jobs'] if j['split'] == 'test' and j['requested_vowel_change_class'] == 0}
    if set(sources) != set(planned):
        raise ValueError('Held-out source screening does not cover the canonical test partition exactly')
    seen = set()
    rows, rejected = [], []
    for row in read_rows(controls / 'items.jsonl'):
        identity = (row['pair_id'], row['family'])
        if identity in seen or row['split'] != 'test' or row['job_id'] not in planned:
            raise ValueError('Duplicate, mixed or unplanned test row')
        seen.add(identity)
        if row['family'] not in FAMILIES or row['label'] != int(row['family'] in POSITIVE):
            raise ValueError('Unexpected physical family/label')
        if row['source_audio_sha256'] != sources[row['job_id']]['sha256']:
            raise ValueError('Source content receipt belongs to another waveform')
        if row.get('audio') and sha(controls / row['audio']) != row['audio_sha256']:
            raise ValueError('Held-out intervention waveform changed')
        if row['job_id'] not in accepted:
            rejected.append({key: row[key] for key in ('pair_id', 'word', 'family')})
            continue
        if row['status'] not in {'measured', 'abstain'} or (row['status'] == 'measured') != (row['features'] is not None):
            raise ValueError('Invalid attempted measurement status')
        rows.append(row)
    fitted = json.loads((fit_dir / 'fitting_rows.json').read_text())
    for key in ('pair_id', 'word_id', 'voice', 'phone_collision_group'):
        if {r[key] for r in fitted} & {r[key] for r in rows}:
            raise ValueError('Held-out identity overlaps train/development: ' + key)
    for identity in {r['pair_id'] for r in rows}:
        group = [r for r in rows if r['pair_id'] == identity]
        if not {'P1', 'P1w', 'P1u', 'S0', 'S_world'} <= {r['family'] for r in group}:
            raise ValueError('Missing attempted mandatory test family')
    predictions, baselines = [], {'both_gaps_empty': [], 'both_min_margins_positive': []}
    for row in rows:
        p, flag = None, None
        if row['status'] == 'measured':
            p = float(score(network, state, np.array(row['features'])[None])[0])
            flag = p >= state['threshold']
        predictions.append(annotated(row, p, flag))
        x = row['features']
        for name in baselines:
            decision = None if x is None else (x[11] == 0 and x[23] == 0 if name == 'both_gaps_empty' else x[1] > 0 and x[13] > 0)
            baselines[name].append(annotated(row, None, decision))
    families = {}
    for family in sorted(FAMILIES):
        group = [r for r in predictions if r['family'] == family]
        if not group:
            raise ValueError('Held-out family missing entirely: ' + family)
        families[family] = {**family_summary(group), 'cluster_interval': cluster_interval(group)}
    criteria = json.loads((fit_dir / 'protocol.json').read_text())['admission_before_test']
    p1 = [r for r in predictions if r['family'] == 'P1']
    gates = {'minimum_test_P1_sources': len({r['pair_id'] for r in p1}) >= criteria['minimum_test_P1_sources'],
             'minimum_test_lexemes': len({r['word_id'] for r in p1}) >= criteria['minimum_test_lexemes'],
             'minimum_test_conditioning_groups': len({r['voice'] for r in p1}) >= criteria['minimum_test_conditioning_groups'],
             'P1_recall': families['P1']['recall'] >= criteria['P1_recall'],
             'P1w_recall': families['P1w']['recall'] >= criteria['P1w_recall'],
             'S0_false_flag_rate': families['S0']['false_flag_rate'] <= criteria['S0_false_flag_rate'],
             'other_retained_vowel_controls': all(families[f]['false_flag_rate'] <= criteria['other_retained_vowel_control_false_flag_rate'] for f in FAMILIES - POSITIVE - {'S0'}),
             'abstentions': all(r['abstention_rate'] <= criteria['maximum_abstention_per_family'] for r in families.values()),
             'P1u_boundary_robustness': families['P1']['recall'] - families['P1u']['recall'] <= criteria['P1u_recall_drop_vs_P1']}
    matched = {r['pair_id'] for r in predictions if r['family'] == 'S2'}
    paired = {f: family_summary([r for r in predictions if r['family'] == f and r['pair_id'] in matched]) for f in ('S2', 'P1w')}
    baseline_reports = {name: {f: family_summary([r for r in values if r['family'] == f]) for f in sorted(FAMILIES)} for name, values in baselines.items()}
    natural = []
    request_dir = ROOT / 'reports/native_requests_test_v74'
    request_report = json.loads((request_dir / 'report.json').read_text())
    if sha(request_dir / 'items.jsonl') != request_report['items_sha256']:
        raise ValueError('Unfiltered request features changed')
    request_rows = read_rows(request_dir / 'items.jsonl')
    if {r['job_id'] for r in request_rows} != {j['id'] for j in selection['jobs'] if j['split'] == 'test'}:
        raise ValueError('Unfiltered request diagnostic does not cover every held-out request')
    for row in request_rows:
        if row['realized_error_label'] is not None or row['split'] != 'test':
            raise ValueError('Natural requests acquired invented gold labels')
        p = None if row['features'] is None else float(score(network, state, np.array(row['features'])[None])[0])
        natural.append({k: row[k] for k in ('job_id', 'pair_id', 'word', 'word_id', 'voice', 'requested_change_class', 'status')} |
                       {'score': p, 'candidate': None if p is None else p >= state['threshold'], 'realized_error_label': None})
    natural_summary = {str(c): {'requests': sum(r['requested_change_class'] == c for r in natural),
        'measured': sum(r['requested_change_class'] == c and r['status'] == 'measured' for r in natural),
        'flagged': sum(r['requested_change_class'] == c and r['candidate'] is True for r in natural), 'accuracy': None} for c in (0, 1)}
    result = {'status': 'completed_frozen_held_out_physical_test', 'model_sha256': sha(fit_dir / 'model.pt'),
              'threshold': state['threshold'], 'families': families, 'admission_gates': gates,
              'physical_control_admission_passed': all(gates.values()), 'production_changed': False,
              'test_sources': len(p1), 'test_lexemes': len({r['word_id'] for r in p1}),
              'conditioning_groups': len({r['voice'] for r in p1}), 'matched_shortened_vs_removed': paired,
              'unfiltered_requested_readings': natural_summary, 'actual_hindi_error_accuracy': None,
              'limits': ['Physical edits are not real Hindi leakage annotations.', 'All conditioning groups come from one approved Native TTS teacher; no human speaker-generalization claim.',
                         'Source eligibility conditions on audible vowel evidence, so selected-control specificity is not unfiltered Native specificity.',
                         'Phonetic collision groups are disjoint; morphological root independence was not established.',
                         'Natural requested edits have null realized-error labels; source/text request identity is never pronunciation gold.']}
    write_group(out, {'report.json': result, 'predictions.json': predictions, 'simple_baselines.json': baseline_reports,
        'unfiltered_requests.json': natural, 'source_content_exclusions.json': rejected,
        'inputs.json': {'model': str(fit_dir / 'model.pt'), 'model_sha256': sha(fit_dir / 'model.pt'), 'external_freeze_sha256': sha(fit_dir / 'external_freeze.json'),
            'test_items_sha256': sha(controls / 'items.jsonl'), 'source_content_sha256': sha(content_path),
            'unfiltered_requests_sha256': sha(request_dir / 'items.jsonl'), 'script_sha256': sha(Path(__file__))}},
        'Single locked test of local vowel omission after reviewed training and external model/threshold freeze. report.json contains predeclared admission and all attempted family denominators; predictions.json exact decisions, including abstentions; '
        'simple_baselines.json fixed empty-gap/margin rules; unfiltered_requests.json unlabeled canonical/edited request diagnostics; source_content_exclusions.json only unedited lexical-content failures; inputs.json immutable identity chain. '
        'No test-based retuning or live-model promotion. Owner Ampixa/user; review2026-11-10, active announcement/research evidence. Retain compact evidence while benchmark consumes it; raw audio remains in controls corpus, NOT on HF; inherited research rights.',
        ['.venv-indicf5/bin/python -m scripts.evaluate_anchored_detector'])
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Frozen held-out evaluation failed explicitly; no promotion or threshold changes')
        raise
