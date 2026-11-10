"""Attach verified transfer evidence and fixed matrix examples to the private UI."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import logging
from pathlib import Path

from hindi_detector.artifacts import write_group

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--application', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.summary / 'report.json').read_text())
    if summary['application_items_sha256'] != sha(args.application / 'items.jsonl'):
        raise ValueError('Transfer comparisons and actual speech application disagree')
    for group in (args.summary, args.base):
        if group == args.summary:
            # write_group records checksums in README; verify this exact current file below.
            continue
        manifest = json.loads((group / 'manifest.json').read_text())
        for member in manifest['files']:
            path = (group / member['path']).resolve(strict=True)
            if not path.is_relative_to(group.resolve()) or sha(path) != member['sha256']:
                raise ValueError('Base review case checksum/path mismatch')
    comparisons = json.loads((args.summary / 'comparisons.json').read_text())
    comparison_lookup = {(r['clip_id'], r['word_index'], r['candidate_id']): r for r in comparisons}
    if len(comparison_lookup) != len(comparisons):
        raise ValueError('Duplicate summary word identity')
    transfer = defaultdict(list)
    for line in (args.application / 'items.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['status'] != 'ok_unreviewed_real_word_application':
            raise ValueError('Cannot render failed actual speech as completed evidence')
        for word in row['words']:
            comparison = comparison_lookup.pop((row['clip_id'], word['word_index'], word['candidate_id']))
            if comparison['audio_sha256'] != row['audio_sha256'] or any(comparison[k] != word[k] for k in ('word', 'start', 'end')):
                raise ValueError('Application/comparison word or audio identity changed')
            if any(comparison['branches'][b]['request_summary'] != s for b, s in word['component']['branches'].items()):
                raise ValueError('Summary and application component values differ')
            transfer[(row['clip_id'], word['word'], word['start'], word['end'])].append(
                {'candidate_id': word['candidate_id'], 'component': word['component'], 'comparison': comparison})
    if comparison_lookup:
        raise ValueError('Orphaned transfer comparisons')

    def annotate(case):
        case = deepcopy(case)
        for system in case['systems']:
            for word in system['words']:
                values = transfer.get((system['clip_id'], word['word'], word['start'], word['end']), [])
                for value in values:
                    if value['comparison']['audio_sha256'] != system['audio_sha256']:
                        raise ValueError('Review waveform differs from scored waveform')
                if values:
                    word['schwa_transfer'] = values
        return case

    index = json.loads((args.base / 'index.json').read_text())
    files = {r['id'] + '.json': annotate(json.loads((args.base / (r['id'] + '.json')).read_text())) for r in index}
    selection = json.loads((ROOT / 'data/pocket_wordmatrix_v4/selection.json').read_text())
    first_prompt = {}; chosen = defaultdict(list)
    for job in selection['jobs']:
        first_prompt.setdefault(job['focus_word'], job['prompt_id'])
        if job['prompt_id'] == first_prompt[job['focus_word']] and len(chosen[job['focus_word']]) < 2:
            if any(j['speaker_id'] == job['speaker_id'] for j in chosen[job['focus_word']]):
                raise ValueError('Fixed example plan repeats reference speaker group')
            chosen[job['focus_word']].append(job)
    if len(chosen) != sum(len(words) for words in selection['word_groups'].values()) or any(len(v) != 2 for v in chosen.values()):
        raise ValueError('Fixed review selection must represent every matrix word with two references')
    wanted = {j['id']: j for group in chosen.values() for j in group}
    systems = defaultdict(list)
    for name in ('reports/wordlab_matrix_f5_v3', 'reports/wordlab_pocket_matrix_v4'):
        directory = ROOT / name; manifest = json.loads((directory / 'manifest.json').read_text())
        if manifest['failures']:
            raise ValueError('Matrix evidence contains extraction failures')
        for row in manifest['items']:
            if row['sentence_id'] not in wanted:
                continue
            path = (directory / row['detail']).resolve(strict=True)
            if not path.is_relative_to(directory.resolve()) or sha(path) != row['detail_sha256']:
                raise ValueError('Matrix detail identity/path mismatch')
            detail = json.loads(path.read_text())
            if detail['audio_sha256'] != row['audio_sha256'] or detail['text'] != wanted[row['sentence_id']]['text']:
                raise ValueError('Selected matrix waveform or text changed')
            systems[row['sentence_id']].append(detail)
    added = []
    for identifier, job in wanted.items():
        group = systems[identifier]
        if len(group) != 3 or {s['system'] for s in group} != {'indicf5_stock', 'pocket_hindi_parent', 'pocket_nepali_v1c'}:
            raise ValueError('Need the actual three matched TTS systems for every review example')
        words = [[w['word'] for w in s['words']] for s in group]
        if any(w != words[0] for w in words[1:]):
            raise ValueError('Matrix word sequences differ')
        focus = [i for i, w in enumerate(words[0]) if w == job['focus_word']]
        if len(focus) != 1:
            raise ValueError('Fixed matrix example requires one unambiguous focus occurrence')
        if identifier + '.json' in files:
            raise ValueError('Case identity collision with base review')
        order = {'pocket_nepali_v1c': 0, 'indicf5_stock': 1, 'pocket_hindi_parent': 2}
        group.sort(key=lambda s: order[s['system']])
        case = {'id': identifier, 'text': job['text'], 'case_kind': 'matched_tts_word',
                'focus_word': job['focus_word'], 'initial_word_index': focus[0], 'systems': group}
        files[identifier + '.json'] = annotate(case)
        added.append({'id': identifier, 'text': job['focus_word'] + ' · IndicF5/Pocket · ' + job['speaker_id'],
                      'case_kind': 'matched_tts_word', 'systems': len(group), 'words': len(words[0])})
    index = added + index
    files['index.json'] = index
    inventory = []
    for name, value in files.items():
        body = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
        inventory.append({'path': name, 'sha256': hashlib.sha256(body).hexdigest(), 'bytes': len(body)})
    files['manifest.json'] = {'files': inventory, 'base_manifest_sha256': sha(args.base / 'manifest.json'),
                              'application_items_sha256': sha(args.application / 'items.jsonl'),
                              'comparisons_sha256': sha(args.summary / 'comparisons.json'),
                              'source_sha256': sha(Path(__file__)), 'new_matched_cases': len(added),
                              'selection_policy': 'First carrier in frozen job order; first two distinct reference groups for every matrix focus word. No score-based selection.',
                              'actual_hindi_accuracy': None, 'actual_prosody_accuracy': None}
    write_group(args.out, files, 'Review actual word-crop schwa components, observed human score support and matched Pocket style; fixed examples span every original focus word without score selection. No invented Hindi/prosody labels or extra WAV copies.',
                [str(args.base.resolve()), str(args.application.resolve()), str(args.summary.resolve()),
                 '.venv-indicf5/bin/python -m scripts.build_transfer_review --base review/human_word_data_v6 --application reports/real_word_transfer_v8 --summary reports/real_word_transfer_summary_v8 --out NEW'])
    print(json.dumps({'total_cases': len(index), 'new_matched_cases': len(added), 'annotated_word_occurrences': sum(bool(w.get('schwa_transfer')) for case in files.values() if isinstance(case, dict) and 'systems' in case for s in case['systems'] for w in s['words'])}))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Verified transfer review build failed; no fabricated examples or scores')
        raise
