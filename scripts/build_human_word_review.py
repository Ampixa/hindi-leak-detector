"""Expose actual human word contexts alongside the existing TTS comparisons."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from hindi_detector.artifacts import write_group


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--boundary-audit', type=Path)
    args = parser.parse_args()
    bank = json.loads((args.bank / 'word_bank.json').read_text())
    selection = json.loads((args.bank / 'selection.json').read_text())
    clips = {r['clip_id']: r for r in selection['clips']}
    by_audio = {r['audio']: r for r in clips.values()}
    manifest = json.loads((args.evidence / 'manifest.json').read_text())
    if manifest['g2p_bank_sha256'] != sha(args.bank / 'word_bank.json') or manifest['failures']:
        raise ValueError('Need complete evidence matching this exact human bank')
    boundary_lookup = {}
    if args.boundary_audit is not None:
        report = json.loads((args.boundary_audit / 'result/report.json').read_text())
        if report['bank_sha256'] != sha(args.bank / 'word_bank.json') or report['evidence_manifest_sha256'] != sha(args.evidence / 'manifest.json') or report['failed_clips']:
            raise ValueError('Boundary audit does not match the complete human evidence')
        for line in (args.boundary_audit / 'items.jsonl').read_text().splitlines():
            row = json.loads(line)
            if row['status'] != 'ok_unreviewed_boundary_audit' or row['sha256'] != clips[row['clip_id']]['sha256']:
                raise ValueError('Boundary audit source identity/status mismatch')
            for word in row['words']:
                key = (row['clip_id'], word['word_index'])
                if key in boundary_lookup:
                    raise ValueError('Duplicate audited word')
                boundary_lookup[key] = word
    details = {}
    for row in manifest['items']:
        path = (args.evidence / row['detail']).resolve(strict=True)
        if not path.is_relative_to(args.evidence.resolve()) or sha(path) != row['detail_sha256']:
            raise ValueError('Human word evidence checksum/path mismatch')
        clip = clips[row['clip_id']]
        detail = json.loads(path.read_text())
        if detail['audio_sha256'] != clip['sha256'] or detail['system'] != 'human_nepali_unlabeled':
            raise ValueError('Human audio identity/system mismatch')
        details[row['clip_id']] = detail
    files, index = {}, []
    colors = ['#007d78', '#315eae', '#b65332', '#775293', '#76601c', '#316b8b', '#9b4b69', '#426336']
    selected = 0
    for family in bank:
        word = family['word']
        identifier = 'human-word-' + hashlib.sha256(word.encode()).hexdigest()[:16]
        systems = []
        for ordinal, example in enumerate(family['human_examples']):
            detail = deepcopy(details[by_audio[example['audio']]['clip_id']])
            occurrences = [(i, w) for i, w in enumerate(detail['words']) if w['word'] == word]
            if not occurrences:
                raise ValueError('Selected human word missing from aligned transcript: ' + word)
            # Fixed first occurrence; never choose based on acoustic score.
            chosen_index, chosen = occurrences[0]
            if args.boundary_audit is not None:
                audited = boundary_lookup[(detail['clip_id'], chosen_index)]
                if (audited['word'], audited['start'], audited['end']) != (chosen['word'], chosen['start'], chosen['end']):
                    raise ValueError('Boundary audit word/timing changed')
                chosen['boundary_audit'] = audited['boundary_audit']
            detail['words'] = [chosen]
            times = detail['tracks']['times']
            take = [i for i, t in enumerate(times) if chosen['start'] - .04 <= t <= chosen['end'] + .04]
            if any(len(v) != len(times) for v in detail['tracks'].values()):
                raise ValueError('Track shapes disagree')
            detail['tracks'] = {k: [v[i] for i in take] for k, v in detail['tracks'].items()}
            detail.update(display_label=f"Human {ordinal + 1} · {example['district']}",
                          display_color=colors[ordinal % len(colors)],
                          reported_region={'state': example['state'], 'district': example['district']},
                          all_selected_word_occurrences=len(occurrences))
            systems.append(detail)
        files[identifier + '.json'] = {'id': identifier, 'text': word, 'case_kind': 'human_word',
                                     'g2p_priors': {k: family[k] for k in ('nepali_g2p', 'hindi_g2p')},
                                     'judgments': family['separate_judgments'], 'systems': systems}
        index.append({'id': identifier, 'text': word + ' · real human contexts', 'case_kind': 'human_word',
                      'systems': len(systems), 'words': 1})
        selected += len(systems)
    base_manifest = json.loads((args.base / 'manifest.json').read_text())
    for member in base_manifest['files']:
        path = (args.base / member['path']).resolve(strict=True)
        if not path.is_relative_to(args.base.resolve()) or sha(path) != member['sha256']:
            raise ValueError('Existing TTS comparison identity mismatch')
    base_index = json.loads((args.base / 'index.json').read_text())
    for row in base_index:
        name = row['id'] + '.json'
        if name in files:
            raise ValueError('Duplicate review case')
        files[name] = json.loads((args.base / name).read_text())
    index.extend(base_index)
    files['index.json'] = index
    inventory = []
    for name, value in files.items():
        body = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
        inventory.append({'path': name, 'sha256': hashlib.sha256(body).hexdigest(), 'bytes': len(body)})
    files['manifest.json'] = {'files': inventory, 'human_word_families': len(bank),
                              'human_recording_word_pairs': selected,
                              'unique_human_recordings': len(details),
                              'boundary_audit_used': str(args.boundary_audit) if args.boundary_audit else None,
                              'boundary_audit_items_sha256': sha(args.boundary_audit / 'items.jsonl') if args.boundary_audit else None,
                              'different_contexts_are_not_matched_sentences': True,
                              'actual_human_error_labels': None}
    write_group(args.out, files,
                'Listen to real regional human word realizations and alternative G2P priors. First occurrence is fixed. Different contexts cannot provide matched intonation judgments; all error/style/native labels remain unset. Only word-window pitch tracks are duplicated, never WAVs.',
                ['Source SHA ' + sha(Path(__file__)), 'Bank SHA ' + sha(args.bank / 'word_bank.json'),
                 'Evidence SHA ' + sha(args.evidence / 'manifest.json'), 'Base manifest SHA ' + sha(args.base / 'manifest.json'),
                 'python -m scripts.build_human_word_review --bank data/human_g2p_bank_v5 --evidence reports/human_g2p_evidence_v5 --base review/wordlab_full_data --out NEW'])
    print(json.dumps(files['manifest.json'] | {'files': len(inventory), 'total_cases': len(index)}))


if __name__ == '__main__':
    main()
