"""Freeze metadata-only target/speaker splits before fitting native-reference heads."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

from hindi_detector.artifacts import write_group
from hindi_detector.native_reference import PHONE_NAMES, PROSODY_COMPARISON_NAMES, SOURCE_LABELS, metadata_partitions

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    path = ROOT / 'data/human_matched_v9/selection.json'
    selection = json.loads(path.read_text())
    if selection['experiment_id'] != 'human_matched_v9':
        raise ValueError('Need the actual verified exact-human-context plan')
    contexts = selection['contexts']; jobs = selection['jobs']
    voices = list(dict.fromkeys(j['speaker_id'] for j in jobs))
    if len(voices) != 2 or len(jobs) != len(contexts) * 2:
        raise ValueError('Need both original frozen TTS reference groups')
    parts = metadata_partitions(contexts)
    context_map = {r['id']: r for r in contexts}
    assigned = []
    for job in jobs:
        context = context_map[job['source_context_id']]
        if any(job[k] != context[k] for k in ('text', 'target_words', 'native_clip_id', 'native_speaker_id')):
            raise ValueError('Context/job metadata disagree')
        split = parts['context_partitions'][context['id']]
        expected_voice = voices[1] if split == 'reserved_evaluation' else voices[0]
        assigned.append({'job_id': job['id'], 'context_id': context['id'], 'partition': split,
                         'native_clip_id': job['native_clip_id'], 'native_speaker_id': job['native_speaker_id'],
                         'tts_reference_group': job['speaker_id'], 'target_words': job['target_words'],
                         'primary': job['speaker_id'] == expected_voice})
    source_files = [Path(__file__), ROOT / 'hindi_detector/native_reference.py']
    files = {'plan.json': {**parts, 'jobs': assigned, 'selection_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                          'source_roles': SOURCE_LABELS, 'training_tts_reference_group': voices[0],
                          'test_tts_reference_group': voices[1], 'phone_feature_names': PHONE_NAMES,
                          'prosody_feature_names': PROSODY_COMPARISON_NAMES,
                          'l2_grid': [.001, .01, .1, 1.], 'selection_rule': 'Minimum development focal-bag source log loss; default0.5 threshold, no test tuning.',
                          'training_word_policy': 'Only planned target-word occurrences contribute to fitting and selection; other words are inference diagnostics.',
                          'limits': ['These are weak checkpoint-source roles, not real Hindi/pronunciation/prosody gold.',
                                     'Target lexemes and human source groups are disjoint; incidental carrier words are not claimed disjoint.',
                                     'The new test voice does not enter fitting or development, including reference features: all reference speech is human.',
                                     'The human corpus and earlier models were already inspected; this is not pristine independent real-error validation.']},
             'source_snapshot.json': [{'path': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'source_utf8': p.read_text()} for p in source_files]}
    write_group(args.out, files, 'Freeze native-reference experiment partitions/features using metadata only before new head fitting. No acoustic feature or source score is read by this planner.',
                ['Exact-context selection SHA ' + files['plan.json']['selection_sha256'],
                 '.venv-indicf5/bin/python -m scripts.build_native_reference_plan --out NEW'])
    print(json.dumps({'contexts': parts['context_counts'], 'components': len(parts['components']), 'primary_jobs': sum(j['primary'] for j in assigned),
                      'phone_features': len(PHONE_NAMES), 'prosody_features': len(PROSODY_COMPARISON_NAMES)}))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Native-reference split freezing failed; no target/source group overlaps silently accepted')
        raise
