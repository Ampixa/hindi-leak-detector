"""Freeze TTS readings of actual verified human contexts for pronunciation/prosody."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path

from hindi_detector.artifacts import write_group

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    bank_path = ROOT / 'data/human_g2p_bank_v5/word_bank.json'
    selection_path = ROOT / 'data/human_g2p_bank_v5/selection.json'
    original = ROOT / 'data/human_original_audio_v6'
    original_report = json.loads((original / 'report.json').read_text())
    if not original_report['all_selected_recovered'] or original_report['byte_identical_staging_reproductions'] != original_report['planned_recordings']:
        raise ValueError('Need complete byte-identical source correspondence before planning prosody anchors')
    verified = json.loads((original / 'recordings.json').read_text())
    verified_by_path = {r['staged_audio']: r for r in verified}
    if len(verified_by_path) != len(verified):
        raise ValueError('Duplicate recovered human source')
    bank = json.loads(bank_path.read_text()); selection = json.loads(selection_path.read_text())
    clips = {r['audio']: r for r in selection['clips']}
    matrix_path = ROOT / 'data/wordmatrix_v3/selection.json'
    matrix = json.loads(matrix_path.read_text())
    voices = {}
    for split in ('development', 'reserved_evaluation'):
        job = next(j for j in matrix['jobs'] if j['split'] == split)
        if job['speaker_id'] in voices or job['speaker_id'] in {c['speaker_id'] for c in clips.values()}:
            raise ValueError('Reference groups overlap each other or the observed human bank')
        reference = {k: job[k] for k in ('speaker_id', 'reference_audio', 'reference_text', 'reference_sha256')}
        if sha(Path(reference['reference_audio'])) != reference['reference_sha256']:
            raise ValueError('Existing verified TTS reference changed')
        voices[reference['speaker_id']] = reference
    contexts = {}; chosen_by_word = {}; excluded = []
    for family in bank:
        examples = []; districts = set(); people = set()
        for example in family['human_examples']:
            raw = verified_by_path[example['audio']]
            clip = clips[example['audio']]
            recipe = raw['recipe_reproduction']; correlation = recipe['source_channel_correlation']
            if correlation is not None and (not math.isfinite(correlation) or correlation < .999):
                excluded.append({'word': family['word'], 'clip_id': clip['clip_id'], 'reason': 'Stereo-channel review flag; excluded from this prosody-anchor plan, not a pronunciation label.'})
                continue
            if example['district'] in districts or example['speaker_id'] in people:
                continue
            if not recipe['encoded_staged_bytes_identical'] or raw['staged_sha256'] != clip['sha256'] or raw['text'] != clip['text'] or raw['speaker_id'] != clip['speaker_id']:
                raise ValueError('Human context text/source correspondence changed')
            if sha(Path(clip['audio'])) != clip['sha256']:
                raise ValueError('Human context waveform changed')
            payload = (original / raw['original_encoded_audio']).resolve(strict=True)
            if not payload.is_relative_to(original.resolve()) or sha(payload) != raw['original_encoded_sha256']:
                raise ValueError('Pinned original payload checksum/path changed')
            identifier = 'humancontext_' + hashlib.sha256(clip['clip_id'].encode()).hexdigest()[:16]
            if identifier not in contexts:
                contexts[identifier] = {'id': identifier, 'text': clip['text'], 'native_clip_id': clip['clip_id'],
                                       'native_audio': clip['audio'], 'native_sha256': clip['sha256'],
                                       'native_speaker_id': clip['speaker_id'], 'reported_state': example['state'],
                                       'reported_district': example['district'], 'target_words': [],
                                       'original_encoded_sha256': raw['original_encoded_sha256'],
                                       'source_revision': raw['source_revision'], 'source_shard': raw['source_shard'], 'source_row': raw['source_row']}
            if family['word'] not in contexts[identifier]['target_words']:
                contexts[identifier]['target_words'].append(family['word'])
            examples.append(identifier); districts.add(example['district']); people.add(example['speaker_id'])
            if len(examples) == 2:
                break
        if len(examples) != 2:
            raise ValueError('Fewer than two verified, channel-consistent, different-district contexts for ' + family['word'])
        chosen_by_word[family['word']] = examples
    jobs = []
    for context in contexts.values():
        for ordinal, voice in enumerate(voices.values()):
            identifier = context['id'] + '_ref' + str(ordinal)
            jobs.append({'id': identifier, 'text': context['text'], 'split': 'development', 'seed': 0, **voice,
                         'source_context_id': context['id'], 'native_clip_id': context['native_clip_id'],
                         'native_speaker_id': context['native_speaker_id'], 'native_audio_sha256': context['native_sha256'],
                         'target_words': context['target_words'], 'automatic_leakage_label': None})
    report = {'status': 'verified_human_context_tts_plan_frozen_before_generation', 'word_families': len(chosen_by_word),
              'contexts': len(contexts), 'reference_groups': len(voices), 'jobs_per_tts': len(jobs),
              'intended_tts_systems': ['indicf5_stock', 'pocket_nepali_v1c'],
              'planned_generated_clips': len(jobs) * 2, 'channel_review_exclusions': excluded,
              'selection_policy': 'First two distinct-speaker/district examples per word in the fixed bank order, with byte-identical original correspondence and no stereo review flag; deduplicate contexts; use first development and reserved reference groups from the old matrix.',
              'limits': ['No word-error/native-acceptability labels are supplied by speakers, region or TTS checkpoint.',
                         'All resulting data are exploratory development evidence; the human bank has already been inspected.',
                         'Identical text permits physical contour comparison, but human delivery and discourse intent remain unreviewed.',
                         'Two TTS reference groups are nuisance controls, not proof of voice invariance.',
                         'Pocket is the preferred style comparator; disagreement with that style is not necessarily a Hindi error.']}
    sources = [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p)} for p in (bank_path, selection_path, original / 'report.json', original / 'recordings.json', matrix_path)]
    source = Path(__file__)
    write_group(args.out, {'selection.json': {'schema_version': 1, 'experiment_id': 'human_matched_v9', 'jobs': jobs, 'clips': [],
                                            'contexts': list(contexts.values()), 'words': chosen_by_word},
                          'report.json': report, 'inputs.json': sources,
                          'source_snapshot.json': {'path': str(source.relative_to(ROOT)), 'sha256': sha(source), 'source_utf8': source.read_text()}},
                'Freeze actual human utterances and separate TTS reference groups so the next comparison covers observed word realizations and identical-context prosody, not just artificial carriers. No additional native audio copies.',
                ['Pinned original human identity report and selected payloads verified at planning time',
                 '.venv-indicf5/bin/python -m scripts.build_human_matched_pack --out NEW'])
    print(json.dumps(report))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Human-context plan failed; no presumed source identity or invented native labels')
        raise
