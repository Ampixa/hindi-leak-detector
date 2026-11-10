"""Persistent frozen-release scoring of fresh same-text outputs outside Swar."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import time

import torch

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.dual_g2p import DualG2P
from hindi_detector.dual_paired_inference import analyze_pair
from hindi_detector.dual_runtime import PhonePair
from hindi_detector.intrinsic_inference import assign_decisions
from hindi_detector.task_vowel_detector import decisions, load, score
from hindi_detector.word_evidence import WordEvidenceEngine, normalized_words
from scripts.calibrate_asr_gate import audio
from scripts.detect_nepali_leakage import reference_qualified
from scripts.extract_corrected_training import read_rows

ROOT = Path(__file__).resolve().parents[1]
NATIVE = 'pocket_teacher_250000_24l'
SYSTEMS = (NATIVE, 'indicf5_stock')


def consensus(query, references, frontend, engine, pair, networks, state):
    groups = defaultdict(list)
    per_reference_words = []
    for voice, reference in references:
        words, sites, x = analyze_pair(query[0], reference[0], query[1], reference[1],
            frontend.word, frontend.ipa, engine.vocab, pair.vocabularies[1])
        if len(x):
            probabilities = score(networks, state, x)
            assign_decisions(words, sites, probabilities, decisions(state, x, probabilities))
        else:
            assign_decisions(words, [], [], [])
        per_reference_words.append(words)
        for site in sites:
            key = (site['word_ordinal'], site['site']['kind'], site['site']['index'], tuple(site['native_phones']))
            groups[key].append({'reference_voice': voice, 'qualified': reference_qualified(site),
                'decision': site['decision'], 'controlled_change_score': site['controlled_vowel_change_score'],
                'site': site['site'], 'native_phones': site['native_phones']})
    words = [{k: w[k] for k in ('word', 'start', 'end', 'phone_probe_sequence')} |
             {'sites': [], 'realized_error_label': None} for w in query[0]['detail']['words']]
    for key, values in groups.items():
        valid = [s for s in values if s['qualified']]
        labels = {s['decision'] for s in valid}
        decision = next(iter(labels)) if len(labels) == 1 else 'uncertain'
        if not valid:
            decision = 'unsupported_reference'
        words[key[0]]['sites'].append({'site': values[0]['site'], 'native_phones': values[0]['native_phones'],
            'decision': decision, 'qualified_references': len(valid), 'available_references': len(values), 'per_reference': values})
    for ordinal, word in enumerate(words):
        labels = {s['decision'] for s in word['sites']}
        word['decision'] = ('strong_vowel_change' if 'strong_vowel_change' in labels else
            'uncertain' if labels & {'uncertain', 'unsupported_reference'} else 'no_strong_vowel_change' if labels else 'untested_site')
        if word['decision'] == 'strong_vowel_change' and any(not r[ordinal]['sites'] and not r[ordinal]['unavailable_reasons'] for r in per_reference_words):
            word['decision'] = 'uncertain'
            word['native_variant_ambiguity'] = True
        word['unavailable_reasons'] = sorted({reason for r in per_reference_words for reason in r[ordinal]['unavailable_reasons']})
    return words


def main():
    plan = ROOT / 'data/cross_system_sentences_v80'
    content = ROOT / 'reports/cross_system_content_v80'
    out = ROOT / 'reports/cross_system_results_v80'
    if out.exists():
        raise FileExistsError(out)
    selection = json.loads((plan / 'selection.json').read_text())
    jobs = {j['id']: j for j in selection['jobs']}
    lookup = {(j['text_id'], j['voice_ordinal']): j for j in jobs.values()}
    packs = {NATIVE: plan / 'teacher', 'indicf5_stock': plan / 'indicf5'}
    active = json.loads((ROOT / 'active_vowel_model.json').read_text())
    model = ROOT / active['model_directory'] / 'model.pt'
    if sha(model) != json.loads((plan / 'protocol.json').read_text())['frozen_detector_sha256']:
        raise ValueError('Frozen detector changed after sentence selection')
    networks, state = load(model)
    for name, digest in state['code_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Release dependency changed: ' + name)
    write_group(out, {'inputs.json': {'selection_sha256': sha(plan / 'selection.json'), 'model_sha256': sha(model),
        'script_sha256': sha(Path(__file__)), 'same_model_as_live': True,
        'labels': None, 'query_sources': list(SYSTEMS), 'reference_conditioning_excludes_query': True}},
        'Fresh v80 cross-system frozen-detector outputs. cases/*.json each complete same-text matched-voice comparison with words/sites/qualifications, full-word playback intervals, raw audio hashes and lexical-content decisions. '
        'index.json ready sentence index, status.json producer progress, report.json complete per-system denominators/focal outcomes, generation_failures.json explicit missing audio. No raw frame/model/audio copies. '
        'Each new realization has null human error label. Known-word contexts are regressions/discovery;10 new focal spellings are outside prior fit/dev and Swar. No morphological or unseen-engine independence assertion. '
        'Owner Ampixa/user, review2026-11-10; active consumer cross-system listening. NOT on HF, inherited source/MMS rights. Retain compact results while audio is reviewed; no training or live-model changes.',
        ['.venv-indicf5/bin/python -m scripts.score_cross_system_sentences'])
    (out / 'cases').mkdir()
    frontend = DualG2P(ROOT)
    engine = WordEvidenceEngine(ROOT, device='mps', pitch_method='dio', temporal_bins=20, retain_frame_evidence=True)
    pair = PhonePair(ROOT, state, device='mps', primary_engine=engine)
    index, all_readings = [], []
    pending = list(selection['contexts'])
    deadline = time.monotonic() + 2100
    while pending:
        screened = {(r['system'], r['job_id']): r for r in read_rows(content / 'items.jsonl')}
        produced = {s: {r['id']: r for r in read_rows(p / 'items.jsonl')} for s, p in packs.items()}
        fresh = False
        for context in pending[:]:
            current = [lookup[context['id'], v] for v in range(4)]
            if any((s, j['id']) not in screened for s in SYSTEMS for j in current):
                continue
            frames, errors = {}, {}
            for system in SYSTEMS:
                for job in current:
                    row = produced[system][job['id']]
                    check = screened[system, job['id']]
                    if row['status'] != 'ok_unreviewed' or check['content']['decision'] == 'reject':
                        continue
                    if row['sha256'] != check['sha256']:
                        raise ValueError('ASR receipt refers to another waveform')
                    try:
                        samples = audio(packs[system], row)
                        detail, arrays = engine.analyze(samples, context['text'])
                        other = pair.encode_one(samples, 1)
                        frames[system, job['voice_ordinal']] = (
                            {'detail': detail, 'phone_log_probs': arrays['phone_log_probs']},
                            {'detail': {**detail, 'phone_frame_seconds': len(samples) / 16000 / len(other)}, 'phone_log_probs': other})
                    except ValueError as error:
                        logging.warning('Retained measurement abstention %s/%s: %s', system, job['id'], error)
                        errors[system, job['voice_ordinal']] = str(error)
            readings = []
            for system in SYSTEMS:
                for job in current:
                    voice = job['voice_ordinal']
                    row, check = produced[system][job['id']], screened[system, job['id']]
                    reading = {'system': system, 'job_id': job['id'], 'text_id': context['id'], 'voice_ordinal': voice,
                        'speaker_id': job['speaker_id'], 'content': check.get('content'), 'words': [], 'realized_error_label': None,
                        'model_sha256': active['model_sha256'], 'reference_voice_ordinals_planned': job['native_comparison_voice_ordinals']}
                    reading['expected_word_positions'] = len(normalized_words(context['text'], engine.romanizer))
                    if row['status'] != 'ok_unreviewed':
                        reading.update(status='unscored_generation_failure', error=row['error'])
                    else:
                        reading.update(audio={'path': str((packs[system] / row['audio']).relative_to(ROOT)), 'sha256': row['sha256'], 'bytes': row['bytes']}, seconds=row['seconds'])
                        if check['content']['decision'] == 'reject':
                            reading['status'] = 'unscored_content_mismatch'
                        elif (system, voice) not in frames:
                            reading.update(status='unscored_measurement_failure', error=errors[system, voice])
                        else:
                            refs = [v for v in job['native_comparison_voice_ordinals'] if (NATIVE, v) in frames and
                                screened[NATIVE, lookup[context['id'], v]['id']]['content']['decision'] == 'pass']
                            if any(produced[NATIVE][lookup[context['id'], v]['id']]['sha256'] == row['sha256'] for v in refs):
                                raise ValueError('Same waveform supplied as both query and reference')
                            reading['reference_voice_ordinals_available'] = refs
                            if not refs:
                                reading['status'] = 'unscored_reference_quality'
                            else:
                                reading.update(status='scored_vowel_evidence', words=consensus(frames[system, voice],
                                    [(v, frames[NATIVE, v]) for v in refs], frontend, engine, pair, networks, state))
                    reading['target_words'] = [w for w in reading['words'] if w['word'] == context['target_word']]
                    reading['flagged_words'] = [w['word'] for w in reading['words'] if w['decision'] == 'strong_vowel_change']
                    reading['tested_words'] = sum(any(s['qualified_references'] for s in w['sites']) for w in reading['words'])
                    readings.append(reading)
            payload = {'context': context, 'readings': readings, 'human_error_labels': None}
            path = out / 'cases' / (context['id'] + '.json')
            path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False) + '\n')
            index.append({'id': context['id'], 'text': context['text'], 'target_word': context['target_word'], 'category': context['category'],
                'case_sha256': sha(path), 'readings': len(readings), 'flagged_readings': sum(bool(r['flagged_words']) for r in readings)})
            all_readings.extend(readings); pending.remove(context); fresh = True
            (out / 'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2) + '\n')
        temporary = out / 'status.partial'
        temporary.write_text(json.dumps({'phase': 'complete_frozen_cross_system_scoring' if not pending else 'scoring_ready_fresh_contexts',
            'pid': os.getpid(), 'updated_utc': datetime.now(timezone.utc).isoformat(), 'contexts_done': len(index), 'contexts_expected': len(selection['contexts'])}, indent=2) + '\n')
        temporary.replace(out / 'status.json')
        if (content / 'failure.json').exists():
            raise RuntimeError('Content source failed explicitly')
        if time.monotonic() > deadline:
            raise TimeoutError('Fresh comparison did not finish within35minutes; partial results retained')
        if not fresh and pending:
            time.sleep(3)
    summary = {}
    for system in SYSTEMS:
        group = [r for r in all_readings if r['system'] == system]
        summary[system] = {'requested': len(group), 'statuses': dict(Counter(r['status'] for r in group)),
            'content_decisions': dict(Counter(r['content']['decision'] if r['content'] else 'generation_failure' for r in group)),
            'readings_with_word_flags': sum(bool(r['flagged_words']) for r in group),
            'readings_with_focal_flags': sum(any(w['decision'] == 'strong_vowel_change' for w in r['target_words']) for r in group),
            'focal_flags_on_content_pass_queries': sum(r['content'] is not None and r['content']['decision'] == 'pass' and any(w['decision'] == 'strong_vowel_change' for w in r['target_words']) for r in group),
            'tested_word_positions': sum(r['tested_words'] for r in group), 'word_positions_with_alignment': sum(len(r['words']) for r in group),
            'word_positions_all_requested': sum(r['expected_word_positions'] for r in group), 'accuracy': None}
    report = {'status': 'complete_frozen_cross_system_discovery', 'systems': summary, 'contexts': len(index),
        'model_sha256': sha(model), 'index_sha256': sha(out / 'index.json'), 'actual_hindi_error_accuracy': None,
        'new_human_confirmations': 0, 'production_changed': False, 'training': False,
        'limits': ['Flags are candidates, never automatic human-confirmed errors.', 'Native teacher queries are also unlabeled realizations, not guaranteed correct controls.',
            'These source model families appeared in earlier work; fresh text does not establish unseen-engine accuracy.',
            'The speech detector has no admitted prosody head; intonation remains diagnostic.']}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (out / 'generation_failures.json').write_text(json.dumps([r for r in all_readings if r['status'] == 'unscored_generation_failure'], ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Frozen cross-system scoring failed explicitly; partial outcomes retained, no model change')
        raise
