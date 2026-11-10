"""Third-encoder discovery audit of every existing Swar candidate word."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import traceback

import soundfile as sf
import torch

from hindi_detector.artifacts import write_group
from hindi_detector.aspirate_context import AspirateContextTable
from hindi_detector.corrected_synthesis import sha
from hindi_detector.independent_phone import IndependentPhone
from hindi_detector.intrinsic_vowels import measure
from hindi_detector.phone_lattice import VOWELS
from hindi_detector.pronunciation_sites import windows
from hindi_detector.teacher_reference_cache import verified_paths
from scripts.extract_corrected_training import read_rows
from scripts.score_swar_sites import load_item

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / 'reports/additional_swar_phones_v65'
    release = ROOT / 'reports/task_vowel_release_v31'
    if out.exists():
        raise FileExistsError(out)
    cases = {row['id']: row for row in json.loads((release / 'actual_swar/cases.json').read_text())}
    priorities = json.loads((release / 'priority_flags.json').read_text())
    speech = ROOT / 'data/swar_speech_v18'
    assets = {(row['id'], row['kind']): row for row in json.loads((speech / 'inventory.json').read_text())['assets']}
    teachers = ROOT / 'data/swar_teacher_v21'
    teacher_features = ROOT / 'reports/swar_teacher_features_v21'
    jobs = json.loads((teachers / 'selection.json').read_text())['jobs']
    receipts = {row['id']: row for row in read_rows(teacher_features / 'items.jsonl')}
    content_path = ROOT / 'reports/swar_teacher_content_v21/items.jsonl'
    content = {row['id']: row for row in read_rows(content_path)}
    needed_jobs = [job for job in jobs if job['text_id'] in {p['id'] for p in priorities}]
    verified_paths(teachers, needed_jobs)
    write_group(out, {'protocol.json': {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'priority_sha256': sha(release / 'priority_flags.json'),
        'case_sha256': sha(release / 'actual_swar/cases.json'), 'selected_model_sha256': sha(release / 'model.pt'),
        'teacher_content_receipts_sha256': sha(content_path),
        'selection': 'Every existing21 candidate words; two known user examples are development checks, all others discovery.',
        'independent_validation': False, 'no_fit_or_threshold_tuning': True,
        'comparison': 'Unconditioned230-phone independent Allosaurus encoder, original recorded Nepali speech, Swar query and both exact24-layer teacher references.',
        'support_rule': 'Deletion only: alternate favored at all3 query bounds, Native favored at all3 human/reference bounds; query alternative and reference Native mean fit above-3; each reference focal Native emission above-1.5; query lexical content pass; both teachers plus recorded comparison required.',
        'quality_limit': 'The existing one-unit diphthong projection does not establish an offglide; report quality evidence without admission.',
        'claim': 'Convergent acoustic evidence for a Hindi-compatible vowel omission relative to these recordings; no causal checkpoint claim, acceptability gold or measured real-error accuracy.',
        'production_predictions_changed': False},
        'source_snapshot.json': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()}
            for p in (Path(__file__), ROOT / 'hindi_detector/independent_phone.py', ROOT / 'hindi_detector/aspirate_context.py')]},
        'Third-encoder audit of actual21 candidate word occurrences. items.json full acoustic evidence '
        'for exact original human/Swar/teacher WAVs; exclusions.json explicit unsupported paths/sites; '
        'report.json all outcomes and supported new words, with actual Hindi-error labels unset. '
        'runtime.json pinned independent model/vocabulary. No audio/frame/model copies, no fit or '
        'production-score changes. Current consumer demonstrate new real-word cases beyond the two '
        'development examples; owner Ampixa/user, review2026-11-10. Retain compact evidence during '
        'this task; not on HF. Source/audio/MMS rights inherited; independent code GPL-3.0, checkpoint '
        'terms absent from release, internal research/no redistribution. Candidate discovery is '
        'selected on existing flags and is not an unbiased accuracy evaluation.',
        ['.venv-indicf5/bin/python -m scripts.audit_additional_swar_phones'])

    def status(phase, **extra):
        (out / 'status.json').write_text(json.dumps({'phase': phase, 'pid': os.getpid(),
            'updated_utc': datetime.now(timezone.utc).isoformat(), **extra}, indent=2) + '\n')

    try:
        status('loading_independent_encoder')
        torch.set_num_threads(4)
        runtime = IndependentPhone(ROOT)
        table = AspirateContextTable(ROOT / 'data/panphon_context_v58')
        ipa = json.loads((ROOT / 'data/swar_phone_plan_v18/ipa_map.json').read_text())
        (out / 'runtime.json').write_text(json.dumps(runtime.metadata, ensure_ascii=False, indent=2) + '\n')
        vowel_ids = frozenset(runtime.vocabulary[token] for token in VOWELS if token in runtime.vocabulary)
        cache = {}

        def encoded(path, digest):
            if path not in cache:
                if sha(path) != digest:
                    raise ValueError('Original acoustic audit waveform hash differs: ' + str(path))
                samples, rate = sf.read(path, dtype='float32')
                if samples.ndim != 1:
                    raise ValueError('Expected mono source')
                values = runtime.encode(samples, rate)
                cache[path] = values, runtime.decode(values)
            return cache[path]

        def evidence(path, digest, word, variant, contrast):
            values, decoded = encoded(path, digest)
            item = {'detail': {'words': [word], 'phone_frame_seconds': runtime.frame_seconds}, 'phone_log_probs': values}
            vector, detail = measure({'any_vowel': vowel_ids}, variant, contrast, windows(item, 0))
            local = [phone for phone in decoded if word['start'] - .06 <= phone['start'] <= word['end'] + .06]
            return {'audio': str(path.relative_to(ROOT)), 'audio_sha256': digest,
                    'word_start': word['start'], 'word_end': word['end'], 'features': vector.tolist(),
                    'evidence': detail, 'free_phones': local}

        records, exclusions = [], []
        for number, priority in enumerate(priorities):
            case = cases[priority['id']]
            ordinal = priority['word_ordinal']
            word = case['analysis']['swar']['words'][ordinal]
            if word['word'] != priority['word']:
                raise ValueError('Frozen candidate lexical identity changed')
            for declared in priority['sites']:
                key = declared['kind'], declared['index'], declared['native_phones']
                matches = [site for site in word['sites'] if (site['site']['kind'], site['site']['index'], site['native_phones']) == key]
                if len(matches) != 1:
                    raise ValueError('Frozen candidate site identity ambiguous')
                site = matches[0]
                record = {'id': case['id'], 'word': word['word'], 'word_ordinal': ordinal, 'site': site['site'],
                          'native_phones': site['native_phones'], 'text': case['text'], 'user_pick': case['user_pick'],
                          'known_development': priority['user_confirmed_development_regression'],
                          'original_teacher_supported': priority['replicated_teacher_reference_support'],
                          'query_content_decision': priority['query_content_decision'], 'readings': {},
                          'actual_hindi_error_label': True if priority['user_confirmed_development_regression'] else None}
                try:
                    units, projections = [], []
                    for phone in site['native_phones']:
                        unit, _, projection = table.representation(ipa[phone], runtime.vocabulary)
                        units.append(unit)
                        projections.append(projection)
                    variant = {'phones': site['native_phones'], 'units': units, 'projections': projections}
                    contrast = dict(site['site'])
                    if contrast['kind'] == 'vowel_quality':
                        contrast['alternative_unit'] = table.representation(contrast['alternative'], runtime.vocabulary)[0]
                    for source in ('swar', 'real'):
                        reading = case['analysis'][source]
                        asset = assets[case['id'], source]
                        if reading['audio_sha256'] != asset['sha256'] or reading['words'][ordinal]['word'] != word['word']:
                            raise ValueError('Actual comparison source/word identity mismatch')
                        record['readings'][source] = evidence(speech / asset['path'], asset['sha256'], reading['words'][ordinal], variant, contrast)
                    for job in [job for job in needed_jobs if job['text_id'] == case['id']]:
                        if job['text'] != case['text']:
                            record.setdefault('reference_exclusions', []).append({'id': job['id'], 'reason': 'Original numeric teacher transcript differs from explicitly expanded case'})
                            continue
                        receipt = receipts[job['id']]
                        screen = content[job['id']]
                        if screen['sha256'] != receipt['sha256'] or screen['text'] != job['text']:
                            raise ValueError('Teacher content receipt belongs to different audio/text')
                        item = load_item(teacher_features, receipt)
                        target = item['detail']['words'][ordinal]
                        if target['word'] != word['word']:
                            raise ValueError('Original teacher word identity differs')
                        record['readings'][job['id']] = evidence(teachers / 'teacher' / receipt['audio'], receipt['sha256'], target, variant, contrast)
                        record['readings'][job['id']]['content_decision'] = screen['content_quality']['decision']
                    query = record['readings']['swar']['features']
                    refs = [value['features'] for name, value in record['readings'].items() if name != 'swar']
                    record['third_encoder_support'] = bool(contrast['kind'] == 'vowel_deletion' and
                        priority['query_content_decision'] == 'pass' and len(refs) == 3 and query[1] > 0 and query[6] > -3 and
                        all(ref[2] < 0 and ref[5] > -3 and ref[8] > -1.5 for ref in refs) and
                        all(reading['content_decision'] == 'pass' for name, reading in record['readings'].items() if name not in {'real', 'swar'}))
                    record['status'] = 'measured_convergent_deletion_support' if record['third_encoder_support'] else 'measured_without_full_support'
                except ValueError as error:
                    logging.warning('Explicit actual third-encoder site limitation %s/%s: %s', case['id'], word['word'], error)
                    record.update(status='unsupported_site', third_encoder_support=False, reason=str(error))
                    exclusions.append({'id': case['id'], 'word': word['word'], 'site': site['site'], 'reason': str(error)})
                records.append(record)
                (out / 'items.json').write_text(json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
            status('checking_actual_candidate_words', word_occurrences=number + 1, planned=len(priorities))
        new = [row for row in records if row['third_encoder_support'] and not row['known_development'] and row['word'] not in {'हैन', 'किनकि'}]
        report = {'status': 'complete_actual_third_encoder_discovery', 'candidate_word_occurrences': len(priorities),
            'measured_sites': sum(row['status'] != 'unsupported_site' for row in records), 'unsupported_sites': len(exclusions),
            'new_supported_occurrences': len(new), 'new_supported_word_families': len({row['word'] for row in new}),
            'new_supported_cases': [{key: row[key] for key in ('id', 'word', 'word_ordinal', 'site', 'text', 'user_pick')} for row in new],
            'actual_hindi_error_accuracy': None, 'independent_test': False, 'production_predictions_changed': False,
            'claim': 'Independent encoder corroboration of a localized Hindi-compatible vowel omission versus original recorded speech and two24-layer references; candidates remain unadjudicated.',
            'items_sha256': sha(out / 'items.json')}
        (out / 'exclusions.json').write_text(json.dumps(exclusions, ensure_ascii=False, indent=2) + '\n')
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        status(report['status'], **{k: report[k] for k in ('new_supported_occurrences', 'new_supported_word_families')})
        print(json.dumps(report, ensure_ascii=False), flush=True)
    except Exception:
        status('failed_explicitly', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Actual third-encoder discovery failed explicitly; no invented findings or prediction changes')
        raise
