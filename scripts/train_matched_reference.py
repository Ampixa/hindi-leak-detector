"""Fit interpretable matched-word source heads with genuinely disjoint references."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from hindi_detector.artifacts import write_group
from hindi_detector.evaluation import binary_metrics
from hindi_detector.native_reference import PHONE_NAMES, PROSODY_COMPARISON_NAMES, SOURCE_LABELS, pair_with_reference
from hindi_detector.word_model import fit_word_head, load_bags, predict_bag, vectors

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)
QUERY_DIRS = {'indicf5_stock': 'reports/human_matched_f5_evidence_v9',
              'pocket_nepali_v1c': 'reports/human_matched_pocket_evidence_v9'}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def focal_view(bag: dict) -> dict:
    indices = bag['focal_indices']
    if not indices:
        raise ValueError('No planned target-word occurrences in matched transcript')
    return {**bag, 'words': [bag['words'][i] for i in indices],
            **{name: bag[name][indices] for name in ('phonetics', 'prosody', 'phone_mask')}}


def available_pairs(bags: list[dict], branch: str) -> tuple[list[dict], list[str]]:
    grouped = defaultdict(list)
    for bag in bags:
        grouped[bag['sentence_id']].append(bag)
    kept = []; excluded = []
    for job, group in grouped.items():
        if len(group) != 2 or {b['system'] for b in group} != set(SOURCE_LABELS):
            raise ValueError('Both query systems required in every matched cell')
        if all(vectors(focal_view(b), branch)[1].any() for b in group):
            kept.extend(group)
        else:
            excluded.append(job)
    if not kept:
        raise ValueError('No fully supported paired bags for ' + branch)
    return kept, excluded


def log_loss(model: dict, bags: list[dict]) -> float:
    labels = np.array([b['class_id'] for b in bags])
    scores = np.clip([predict_bag(model, focal_view(b))[0] for b in bags], 1e-10, 1 - 1e-10)
    return float(-np.mean(labels * np.log(scores) + (1 - labels) * np.log1p(-scores)))


def predictions(model: dict, bags: list[dict]) -> list[dict]:
    output = []
    for bag in bags:
        focal, _ = predict_bag(model, focal_view(bag)); _, scores = predict_bag(model, bag)
        output.append({**{k: bag[k] for k in ('clip_id', 'audio_sha256', 'sentence_id', 'speaker_id', 'system', 'class_id', 'split', 'reference_clip_id', 'reference_audio_sha256', 'reference_speaker_id', 'reference_kind', 'context_id', 'component_id', 'focal_indices')},
                       'focal_bag_source_evidence': focal,
                       'words': [{'word': w['word'], 'start': w['start'], 'end': w['end'], 'source_evidence': value,
                                  'actual_hindi_error_probability': None} for w, value in zip(bag['words'], scores, strict=True)]})
    return output


def evaluate(rows: list[dict]) -> dict:
    labels = np.array([r['class_id'] for r in rows]); scores = np.array([r['focal_bag_source_evidence'] for r in rows])
    groups = defaultdict(dict)
    for row in rows:
        group = groups[row['sentence_id']]
        if row['system'] in group:
            raise ValueError('Duplicate source prediction for a paired job')
        group[row['system']] = row['focal_bag_source_evidence']
    if any(set(g) != set(SOURCE_LABELS) for g in groups.values()):
        raise ValueError('Incomplete paired source evaluation')
    deltas = [g['indicf5_stock'] - g['pocket_nepali_v1c'] for g in groups.values()]
    # Resample independent metadata components, not repeated clips/conditions.
    components = defaultdict(list)
    for i, row in enumerate(rows):
        components[row['component_id']].append(i)
    keys = sorted(components); random = np.random.default_rng(0); bootstrap = []
    for _ in range(2000):
        chosen = random.choice(keys, len(keys), replace=True)
        indices = [i for key in chosen for i in components[key]]
        bootstrap.append(float(np.mean((scores[indices] >= .5) == labels[indices])))
    return {'weak_source_metrics': binary_metrics(labels, scores), 'paired_jobs': len(groups),
            'f5_source_score_above_pocket': sum(v > 0 for v in deltas), 'ties': sum(v == 0 for v in deltas),
            'median_pair_delta': float(np.median(deltas)), 'independent_metadata_components': len(keys),
            'component_bootstrap_source_accuracy_95_interval': np.quantile(bootstrap, [.025, .975]).tolist(),
            'actual_hindi_accuracy': None, 'actual_prosody_accuracy': None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--reference-kind', choices=['human', 'pocket'], required=True)
    parser.add_argument('--reference-evidence', type=Path, required=True)
    parser.add_argument('--reference-pack', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh immutable matched-reference fitting group')
    plan = json.loads(args.plan.read_text()); selection_path = ROOT / 'data/human_matched_v9/selection.json'
    selection = json.loads(selection_path.read_text())
    if sha(selection_path) != plan['selection_sha256']:
        raise ValueError('Frozen context/partition metadata changed')
    jobs = {j['id']: j for j in selection['jobs']}; assignments = {j['job_id']: j for j in plan['jobs']}
    component_for_context = {c: component['id'] for component in plan['components'] for c in component['contexts']}
    manifests = {}; row_for_job = {}; input_hashes = {}
    for system, name in QUERY_DIRS.items():
        directory = ROOT / name; manifest = json.loads((directory / 'manifest.json').read_text())
        if manifest['failures'] or manifest['successes'] != len(jobs) or len(manifest['items']) != len(jobs):
            raise ValueError('Need complete actual query word evidence before fitting')
        if any(r['system'] != system for r in manifest['items']):
            raise ValueError('Query source registry differs')
        manifests[system] = manifest
        row_for_job[system] = {r['sentence_id']: r for r in manifest['items']}
        if set(row_for_job[system]) != set(jobs):
            raise ValueError('Generated query jobs do not match frozen plan')
        input_hashes[name] = sha(directory / 'manifest.json')
    reference_manifest = json.loads((args.reference_evidence / 'manifest.json').read_text())
    if reference_manifest['failures']:
        raise ValueError('Reference evidence contains failed clips')
    for manifest in manifests.values():
        if any(manifest[k] != reference_manifest[k] for k in ('phone_encoder_revision', 'phone_vocabulary', 'prosody_feature_names', 'prosody_code_sha256', 'alignment_sha256')):
            raise ValueError('Reference/query acoustic feature recipe differs')
    vocabulary = reference_manifest['phone_vocabulary']
    reference_ids = {}
    if args.reference_kind == 'human':
        if args.reference_pack is not None:
            raise ValueError('Human reference identities are frozen in the original context plan')
        reference_ids = {c['id']: c['native_clip_id'] for c in selection['contexts']}
    else:
        if args.reference_pack is None:
            raise ValueError('Pocket references require the actual generated pack')
        rp = json.loads((args.reference_pack / 'selection.json').read_text()); rjobs = {j['id']: j for j in rp['jobs']}
        if rp['experiment_id'] != 'native_reference_pocket_v10' or len(rp['clips']) != len(selection['contexts']):
            raise ValueError('Need complete separately conditioned Pocket references')
        for clip in rp['clips']:
            rjob = rjobs[clip['sentence_id']]
            context = rjob['source_context_id']
            if context in reference_ids or rjob['statistical_partition'] != plan['context_partitions'][context] or clip['system'] != 'pocket_nepali_v1c':
                raise ValueError('Pocket reference/context split differs')
            reference_ids[context] = clip['clip_id']
        input_hashes[str(args.reference_pack / 'selection.json')] = sha(args.reference_pack / 'selection.json')
    reference_rows = {r['clip_id']: r for r in reference_manifest['items']}
    if not set(reference_ids.values()) <= set(reference_rows):
        raise ValueError('Planned native/Pocket reference missing from evidence')
    input_hashes[str(args.reference_evidence / 'manifest.json')] = sha(args.reference_evidence / 'manifest.json')
    # Assert query AND reference source-group isolation before reading any NPZ.
    source_groups = defaultdict(set); target_words = defaultdict(set)
    for assignment in assignments.values():
        if not assignment['primary']:
            continue
        split = assignment['partition']; job = jobs[assignment['job_id']]
        source_groups[split].update([job['speaker_id'], reference_rows[reference_ids[assignment['context_id']]]['speaker_id']])
        target_words[split].update(job['target_words'])
    if (source_groups['train'] | source_groups['development']) & source_groups['reserved_evaluation']:
        raise ValueError('Held-out query/reference source group leaks into fitting or selection')
    for left, right in (('train', 'development'), ('train', 'reserved_evaluation'), ('development', 'reserved_evaluation')):
        if target_words[left] & target_words[right]:
            raise ValueError('Target lexeme leaks across frozen partitions')

    def load(partitions: set[str], *, primary: bool = True) -> list[dict]:
        chosen = [a for a in assignments.values() if a['partition'] in partitions and a['primary'] == primary]
        reference_clips = {reference_ids[a['context_id']] for a in chosen}
        native, _ = load_bags(args.reference_evidence, clip_ids=reference_clips)
        native = {b['clip_id']: b for b in native}
        bags = []
        for system, name in QUERY_DIRS.items():
            ids = {row_for_job[system][a['job_id']]['clip_id'] for a in chosen}
            queries, _ = load_bags(ROOT / name, clip_ids=ids)
            for query in queries:
                job = jobs[query['sentence_id']]; a = assignments[job['id']]
                if query['speaker_id'] != job['speaker_id'] or query['text'] != job['text']:
                    raise ValueError('Query voice/text differs from frozen conditioning job')
                reference = native[reference_ids[a['context_id']]]
                if args.reference_kind == 'human' and reference['audio_sha256'] != job['native_audio_sha256']:
                    raise ValueError('Human anchor waveform changed')
                paired = pair_with_reference(query, reference, vocabulary, a['partition'], reference_kind=args.reference_kind)
                indices = [i for i, w in enumerate(paired['words']) if w['word'] in job['target_words']]
                if {paired['words'][i]['word'] for i in indices} != set(job['target_words']):
                    raise ValueError('Planned target word missing from actual aligned reading')
                paired.update(focal_indices=indices, context_id=a['context_id'], component_id=component_for_context[a['context_id']])
                bags.append(paired)
        return bags

    sources = [Path(__file__), ROOT / 'hindi_detector/native_reference.py', ROOT / 'hindi_detector/word_model.py', ROOT / 'hindi_detector/local_features.py', ROOT / 'hindi_detector/evaluation.py']
    write_group(args.out, {'inputs.json': {'plan_sha256': sha(args.plan), 'reference_kind': args.reference_kind, 'manifest_hashes': input_hashes,
                                        'source_groups': {k: sorted(v) for k, v in source_groups.items()}, 'target_words': {k: sorted(v) for k, v in target_words.items()}},
                          'source_snapshot.json': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()} for p in sources]},
                'Fit low-dimensional same-text reference heads using focal-bag checkpoint roles, without inventing pronunciation/prosody-error gold. Only planned targets enter fitting/selection.',
                [str(args.plan.resolve()), str(args.reference_evidence.resolve()), 'Exact invocation and input hashes retained; test features read only after selection artifact.'])
    train = load({'train'}); dev = load({'development'})
    models, choices = {}, {}
    for branch in ('phonetics', 'prosody', 'combined'):
        fitting, excluded_train = available_pairs(train, branch); selecting, excluded_dev = available_pairs(dev, branch)
        trials = []
        for l2 in plan['l2_grid']:
            model = fit_word_head([focal_view(b) for b in fitting], branch, l2, source_priors={'indicf5_stock': .5, 'pocket_nepali_v1c': .5}, negative_word_weight=0.)
            loss = log_loss(model, selecting); trials.append((loss, l2, model))
            LOG.info('%s %s L2=%s development source loss=%.6f', args.reference_kind, branch, l2, loss)
        loss, l2, model = min(trials, key=lambda item: (item[0], item[1]))
        model.update(reference_recipe='relative_word_comparison_v10', reference_kind=args.reference_kind,
                     phone_feature_names=PHONE_NAMES, prosody_feature_names=PROSODY_COMPARISON_NAMES,
                     phone_encoder_revision=reference_manifest['phone_encoder_revision'],
                     phone_vocabulary_sha256=hashlib.sha256(json.dumps(vocabulary, sort_keys=True).encode()).hexdigest(),
                     source_labels=SOURCE_LABELS, score_meaning='Balanced weak IndicF5-versus-Pocket source evidence from same-text word comparisons; not actual Hindi/pronunciation/prosody-error probability.',
                     actual_hindi_accuracy=None)
        models[branch] = model
        choices[branch] = {'chosen_l2': l2, 'development_log_loss': loss,
                           'trials': [{'l2': b, 'development_log_loss': a} for a, b, _ in trials],
                           'training_query_bags': len(fitting), 'development_query_bags': len(selecting),
                           'excluded_training_jobs': excluded_train, 'excluded_development_jobs': excluded_dev}
    write_group(args.out / 'selection', {'models.json': models, 'choices.json': choices, 'freeze.json': {'test_features_read': False}},
                'Freeze all branch weights/hyperparameters using training/development only before any reserved query/reference features are loaded.',
                ['Parent inputs.json and frozen plan; default0.5 source threshold, no test tuning.'])
    test = load({'reserved_evaluation'}); reports = {}; all_predictions = {}
    for branch, model in models.items():
        supported, excluded = available_pairs(test, branch); scored = predictions(model, supported)
        reports[branch] = {'reserved_source_evaluation': evaluate(scored), 'excluded_test_jobs': excluded}
        all_predictions[branch] = scored
    # Diagnostic conditioning changes are read only after weights/test report are fixed.
    diagnostics = load({'train', 'development', 'reserved_evaluation'}, primary=False)
    primary = train + dev + test
    for branch, model in models.items():
        supported, _ = available_pairs(primary + diagnostics, branch)
        scored = predictions(model, supported)
        by_context = defaultdict(list)
        for row in scored:
            by_context[(row['context_id'], row['system'])].append(row['focal_bag_source_evidence'])
        deltas = [abs(v[0] - v[1]) for v in by_context.values() if len(v) == 2]
        if len(deltas) != len(selection['contexts']) * 2:
            raise ValueError('Conditioning diagnostic lost a paired context/source')
        reports[branch]['conditioning_group_sensitivity'] = {'context_source_pairs': len(deltas), 'median_absolute_score_change': float(np.median(deltas)), 'p90_absolute_score_change': float(np.quantile(deltas, .9)), 'voice_identity_preservation_verified': False}
        all_predictions[branch + '_all_conditioning_groups'] = scored
    report = {'status': 'matched_reference_heads_trained_source_accuracy_only', 'reference_kind': args.reference_kind,
              'branches': reports, 'plan_sha256': sha(args.plan), 'models_sha256': sha(args.out / 'selection/models.json'),
              'primary_context_counts': plan['context_counts'], 'feature_dimensions': {'phonetics': len(PHONE_NAMES), 'prosody': len(PROSODY_COMPARISON_NAMES), 'combined': len(PHONE_NAMES) + len(PROSODY_COMPARISON_NAMES)},
              'test_acoustic_features_read_after_weights_frozen': True, 'actual_hindi_accuracy': None, 'actual_native_acceptability_accuracy': None, 'actual_prosody_accuracy': None,
              'limits': ['Labels are actual checkpoint sources under user-specified roles, not reviewed word/intonation-error labels.',
                         'Only32 selected lexemes,63 contexts and one held-out query conditioning group support the primary source test.',
                         'Target lexemes, human context source groups, and test query/reference conditioning groups are separated; incidental carrier lexemes are not claimed disjoint.',
                         'Pitch availability is explicit; relative feature normalization is not proof of complete voice invariance.',
                         'Phone-posterior distances do not solve unsupported dental/other phoneme distinctions or certify a schwa realization.',
                         'Word-level local scores remain weak evidence; no word-error accuracy or production reliability is established.',
                         'Human reference data were inspected in earlier experiments; this is not untouched human-error validation.']}
    write_group(args.out / 'evaluation', {'report.json': report, 'predictions.json': all_predictions},
                'Evaluate frozen same-text reference heads on held-out targets/human groups/query and reference conditioning groups; retain per-word scores and conditioning sensitivity separately from actual Hindi accuracy.',
                ['../selection/models.json SHA ' + report['models_sha256'], 'Parent inputs/source snapshot and frozen metadata plan'])
    print(json.dumps(report))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Matched-reference fitting/evaluation failed; do not substitute source labels for real error validation')
        raise
