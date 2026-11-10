"""Plan separate Pocket reference voices without exposing held-out feature data."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

from hindi_detector.artifacts import write_group

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--split-plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    split = json.loads(args.split_plan.read_text())
    selection = json.loads((ROOT / 'data/human_matched_v9/selection.json').read_text())
    if hashlib.sha256((ROOT / 'data/human_matched_v9/selection.json').read_bytes()).hexdigest() != split['selection_sha256']:
        raise ValueError('Frozen context plan differs')
    excluded = {j['speaker_id'] for j in selection['jobs']} | {c['native_speaker_id'] for c in selection['contexts']}
    matrix_path = ROOT / 'data/wordmatrix_v3/selection.json'
    matrix = json.loads(matrix_path.read_text()); available = {}
    for job in matrix['jobs']:
        if job['speaker_id'] not in excluded:
            available.setdefault(job['speaker_id'], {k: job[k] for k in ('speaker_id', 'reference_audio', 'reference_sha256', 'reference_text')})
    if len(available) < 2:
        raise ValueError('Need two additional disjoint actual TTS reference groups')
    voices = list(available.values())[:2]
    for voice in voices:
        if hashlib.sha256(Path(voice['reference_audio']).read_bytes()).hexdigest() != voice['reference_sha256']:
            raise ValueError('Existing TTS conditioning reference changed')
    jobs = []; words = {}
    for context in selection['contexts']:
        partition = split['context_partitions'][context['id']]
        voice = voices[1] if partition == 'reserved_evaluation' else voices[0]
        for word in context['target_words']:
            words.setdefault(word, []).append(context['id'])
        jobs.append({'id': 'pref10_' + context['id'], 'source_context_id': context['id'],
                     'text': context['text'], 'target_words': context['target_words'], 'split': 'development',
                     'statistical_partition': partition, 'seed': 0, **voice,
                     'native_clip_id': context['native_clip_id'], 'automatic_leakage_label': None})
    report = {'experiment_id': 'native_reference_pocket_v10', 'contexts': len(selection['contexts']),
              'jobs_per_tts': len(jobs), 'word_families': len(words), 'reference_groups': 2,
              'training_development_reference_group': voices[0]['speaker_id'], 'test_reference_group': voices[1]['speaker_id'],
              'split_plan_sha256': hashlib.sha256(args.split_plan.read_bytes()).hexdigest(),
              'purpose': 'Provide a separate-voice Pocket reading as an inference reference; both test query and reference voice groups remain outside fitting/development.',
              'actual_hindi_accuracy': None}
    write_group(args.out, {'selection.json': {'schema_version': 1, 'experiment_id': 'native_reference_pocket_v10',
                                            'jobs': jobs, 'clips': [], 'contexts': selection['contexts'], 'words': words},
                          'report.json': report,
                          'source_snapshot.json': {'path': str(Path(__file__).relative_to(ROOT)), 'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'source_utf8': Path(__file__).read_text()}},
                'Freeze separate Pocket reference generation for a deployable same-text comparison model, without extra human/audio/model copies or gold inferred from source roles.',
                [str(args.split_plan.resolve()), 'Conditioning-source selection SHA ' + hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
                 '.venv-indicf5/bin/python -m scripts.build_pocket_reference_plan --split-plan reports/native_reference_plan_v10/plan.json --out NEW'])
    print(json.dumps(report))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Separate-voice Pocket reference planning failed')
        raise
