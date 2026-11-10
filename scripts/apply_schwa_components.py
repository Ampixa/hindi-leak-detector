"""Apply fixed synthetic-request heads to real TTS/human words with abstentions."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import traceback

import soundfile as sf

from hindi_detector.artifacts import write_group
from hindi_detector.interventions import LABEL_SOURCE
from hindi_detector.real_word_transfer import score_real_word
from hindi_detector.review_server import read_items
from hindi_detector.word_evidence import WordEvidenceEngine

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)
SOURCES = (
    ('matrix_indicf5', 'data/wordmatrix_v3', 'reports/wordlab_matrix_f5_v3', 'focus'),
    ('matrix_pocket', 'data/pocket_wordmatrix_v4', 'reports/wordlab_pocket_matrix_v4', 'focus'),
    ('regional_humans', 'data/human_g2p_bank_v5', 'reports/human_g2p_evidence_v5', 'selected_words'),
    ('original_benchmark', 'data/indicf5_audio8_v1', 'reports/wordlab_test_v3', 'all_candidate_words'),
    ('native_challenge', 'data/pocket_native_challenge_v4', 'reports/wordlab_pocket_native_v4', 'all_candidate_words'),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-clips-per-source', type=int)
    args = parser.parse_args()
    if args.max_clips_per_source is not None and args.max_clips_per_source < 1:
        raise ValueError('Positive explicit source clip limit required')
    if not (args.out / 'README.md').is_file() or (args.out / 'items.jsonl').exists():
        raise ValueError('Need a documented fresh real-word application group')
    models_path = ROOT / 'reports/intervention_heads_v7/selection/models.json'
    models = json.loads(models_path.read_text())
    if set(models) != {'ctc_site', 'pooled_phone'} or any(m['label_source'] != LABEL_SOURCE for m in models.values()):
        raise ValueError('Expected unchanged explicit intervention heads')
    candidates_path = ROOT / 'data/schwa_candidates_v1/candidates.json'
    candidates = json.loads(candidates_path.read_text()); by_id = {r['candidate_id']: r for r in candidates}
    by_word = defaultdict(list)
    for row in candidates:
        by_word[row['word']].append(row)
    engine = WordEvidenceEngine(ROOT)
    if any(m['phone_encoder_revision'] != engine.phone_inventory['revision'] for m in models.values()):
        raise ValueError('Component/phone checkpoint mismatch')
    with (ROOT / '.sources/frontend/data/frontend/ipa_map.tsv').open() as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        if reader.fieldnames != ['phone', 'ipa_candidate', 'status', 'notes']:
            raise ValueError('IPA candidate map changed')
        ipa = {r['phone']: r['ipa_candidate'] for r in reader}
    source_files = [Path(__file__), ROOT / 'hindi_detector/real_word_transfer.py', ROOT / 'hindi_detector/interventions.py',
                    ROOT / 'hindi_detector/schwa_probe.py', ROOT / 'hindi_detector/ctc.py', ROOT / 'hindi_detector/word_evidence.py']
    (args.out / 'source_snapshot.json').write_text(json.dumps([{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()} for p in source_files], indent=2) + '\n')
    plans = []; inputs = []; identities = set()
    for source, pack_name, evidence_name, target_policy in SOURCES:
        pack, evidence = ROOT / pack_name, ROOT / evidence_name
        selected = json.loads((pack / 'selection.json').read_text())
        jobs = {r['id']: r for r in selected['jobs']}
        clips, _ = read_items(pack)
        manifest_path = evidence / 'manifest.json'; manifest = json.loads(manifest_path.read_text())
        if manifest['failures']:
            raise ValueError('Source word evidence has failed clips: ' + source)
        if manifest['phone_encoder_revision'] != engine.phone_inventory['revision']:
            raise ValueError('Original word-evidence probe differs from fixed component')
        inputs.append({'source': source, 'pack_selection_sha256': sha(pack / 'selection.json'),
                       'evidence_manifest_sha256': sha(manifest_path), 'pack': pack_name, 'evidence': evidence_name,
                       'target_policy': target_policy})
        items = manifest['items'][:args.max_clips_per_source] if args.max_clips_per_source else manifest['items']
        for item in items:
            if item['clip_id'] in identities:
                raise ValueError('Duplicate clip identity across application packs')
            identities.add(item['clip_id'])
            row = clips[item['clip_id']]; job = jobs[row['sentence_id']]
            for key in ('system', 'speaker_id', 'sentence_id', 'text', 'split'):
                if item[key] != row[key]:
                    raise ValueError('Word/source clip identity differs: ' + key)
            if item['audio_sha256'] != row['sha256'] or job['text'] != row['text']:
                raise ValueError('Waveform/text source identity mismatch')
            detail_path = (evidence / item['detail']).resolve(strict=True)
            if not detail_path.is_relative_to(evidence.resolve()) or sha(detail_path) != item['detail_sha256']:
                raise ValueError('Cached alignment/detail checksum mismatch')
            detail = json.loads(detail_path.read_text())
            if detail['audio_sha256'] != row['sha256']:
                raise ValueError('Cached word detail has another waveform parent')
            if target_policy == 'focus':
                target = by_id[job['focus_candidate_id']]
                if target['word'] != job['focus_word']:
                    raise ValueError('Frozen matrix focus candidate and word disagree')
                wanted = {target['word']: [target]}
            elif target_policy == 'selected_words':
                wanted = {word: by_word[word] for word in job['target_words']}
            else:
                wanted = by_word
            targets = [(i, word, candidate) for i, word in enumerate(detail['words'])
                       if word['word'] in wanted for candidate in wanted[word['word']]]
            if target_policy != 'all_candidate_words' and not set(wanted) <= {w['word'] for _, w, _ in targets}:
                raise ValueError('Planned focus word is missing from cached aligned transcript')
            plans.append({'source': source, 'pack': pack, 'item': item, 'row': row, 'job': job,
                          'targets': targets, 'target_policy': target_policy})
    (args.out / 'plan.json').write_text(json.dumps({'models_sha256': sha(models_path), 'candidates_sha256': sha(candidates_path),
          'sources': inputs, 'clips': len(plans), 'max_clips_per_source': args.max_clips_per_source,
          'candidate_word_occurrences': sum(len(p['targets']) for p in plans)}, indent=2) + '\n')
    records = []; failures = 0

    def status(phase):
        path = args.out / 'status.json'; temporary = path.with_suffix('.partial')
        temporary.write_text(json.dumps({'status': phase, 'processed_clips': len(records), 'planned_clips': len(plans),
                                        'failed_clips': failures, 'updated_utc': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')
        temporary.replace(path)

    status('applying_fixed_components_to_real_words')
    for plan in plans:
        row, item, job = plan['row'], plan['item'], plan['job']
        record = {k: row[k] for k in ('clip_id', 'system', 'speaker_id', 'sentence_id', 'text', 'split')}
        record.update(source=plan['source'], audio_sha256=row['sha256'], detail_sha256=item['detail_sha256'],
                      target_policy=plan['target_policy'], prompt_id=job.get('prompt_id'), actual_hindi_error_label=None)
        try:
            path = (plan['pack'] / row['audio']).resolve(strict=True)
            if sha(path) != row['sha256']:
                raise ValueError('Actual application waveform changed')
            wave, rate = sf.read(path, dtype='float32', always_2d=True)
            if wave.shape[1] != 1:
                raise ValueError('Expected explicitly prepared mono source audio')
            words = []
            for index, word, candidate in plan['targets']:
                score = score_real_word(engine, models, wave[:, 0], rate, word, candidate, ipa)
                words.append({'word': word['word'], 'word_index': index, 'candidate_id': candidate['candidate_id'],
                              'start': word['start'], 'end': word['end'], 'alignment_score': word['alignment_score'],
                              'prosody': word['prosody'], 'component': score,
                              'actual_realization_gold': None, 'actual_hindi_error_label': None})
            record.update(status='ok_unreviewed_real_word_application', words=words)
        except Exception:
            failures += 1
            record.update(status='failed_explicitly_excluded', error=traceback.format_exc(), words=[])
            LOG.exception('Real-word application failed for %s; no imputed scores', row['clip_id'])
        with (args.out / 'items.jsonl').open('a') as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        records.append(record)
        if len(records) % 10 == 0:
            status('applying_fixed_components_to_real_words')
            LOG.info('Real-word application %s/%s (%s failures)', len(records), len(plans), failures)
    report = {'status': 'real_word_components_applied_not_validated_hindi_accuracy', 'clips': len(records),
              'failed_clips': failures, 'candidate_word_occurrences': sum(len(r['words']) for r in records),
              'sources': dict(Counter(r['source'] for r in records)), 'systems': dict(Counter(r['system'] for r in records)),
              'models_sha256': sha(models_path), 'actual_hindi_accuracy': None, 'native_acceptability_accuracy': None,
              'preferred_style_is_correctness_gold': False,
              'limits': ['Frozen component classes are synthetic phoneme requests, not reviewed real-word errors.',
                         'Each crop is independently encoded as in component training; short/impossible/unsupported crops remain unavailable.',
                         'MMS word timestamps remain automatic; seven fixed windows measure classifier sensitivity but do not prove boundaries.',
                         'Source/checkpoint ordering is descriptive evidence, not actual Hindi-leakage accuracy.',
                         'The regional human bank supplies observations, never labels inferred from geography.']}
    write_group(args.out / 'result', {'report.json': report},
                'Apply frozen schwa request components to actual IndicF5, Pocket, Audio8 and human speech while preserving uncertainty and absent real-error labels.',
                ['Parent plan.json, items.jsonl and source_snapshot.json', 'Frozen models SHA ' + sha(models_path),
                 '.venv-indicf5/bin/python -m scripts.apply_schwa_components --out NEW'])
    status('real_word_application_complete_with_explicit_coverage')
    print(json.dumps(report))
    if failures:
        raise RuntimeError('Real-word application contains failed clips; inspect before summarizing or fitting')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Actual-speech application incomplete; no synthetic success substituted for real accuracy')
        raise
