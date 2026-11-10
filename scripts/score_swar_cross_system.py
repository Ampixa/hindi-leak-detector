"""Frozen v31 detector on Swar v84 renders, with the same v80 Pocket references and consensus as score_cross_system_sentences."""
from __future__ import annotations

from collections import Counter
import json
import logging
from pathlib import Path

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.dual_g2p import DualG2P
from hindi_detector.dual_runtime import PhonePair
from hindi_detector.task_vowel_detector import load
from hindi_detector.word_evidence import WordEvidenceEngine, normalized_words
from scripts.calibrate_asr_gate import audio
from scripts.extract_corrected_training import read_rows
from scripts.score_cross_system_sentences import NATIVE, consensus

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'data/cross_system_sentences_v80'
PACKS = {NATIVE: PLAN / 'teacher', 'swar_podvoices_ema': ROOT / 'data/cross_system_swar_v84'}
CONTENT = {NATIVE: ROOT / 'reports/cross_system_content_v80', 'swar_podvoices_ema': ROOT / 'reports/cross_system_swar_content_v84'}
OUT = ROOT / 'reports/cross_system_swar_v84'
SWAR = 'swar_podvoices_ema'


def main():
    selection = json.loads((PLAN / 'selection.json').read_text())
    lookup = {(j['text_id'], j['voice_ordinal']): j for j in selection['jobs']}
    active = json.loads((ROOT / 'active_vowel_model.json').read_text())
    model = ROOT / active['model_directory'] / 'model.pt'
    if sha(model) != json.loads((PLAN / 'protocol.json').read_text())['frozen_detector_sha256']:
        raise ValueError('Frozen detector changed after sentence selection')
    networks, state = load(model)
    for name, digest in state['code_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Release dependency changed: ' + name)
    screened = {(r['system'], r['job_id']): r for s, p in CONTENT.items() for r in read_rows(p / 'items.jsonl') if r['system'] == s}
    produced = {s: {r['id']: r for r in read_rows(p / 'items.jsonl')} for s, p in PACKS.items()}
    write_group(OUT, {'inputs.json': {'selection_sha256': sha(PLAN / 'selection.json'), 'model_sha256': sha(model), 'script_sha256': sha(Path(__file__)),
            'swar_items_sha256': sha(PACKS[SWAR] / 'items.jsonl'), 'swar_content_sha256': sha(CONTENT[SWAR] / 'items.jsonl'), 'labels': None}},
        'Frozen v31 detector outputs for Swar podvoices_ema on the 26 v80 texts x 4 prompts; references are the same two other-voice v80 Pocket renders. '
        'cases/*.json per-text readings with words/sites/per-reference scores; report.json denominators comparable to reports/cross_system_results_v80/report.json. '
        'Flags are candidates, not confirmed errors. Created by Claude (claude-opus-5-5) Claude Code session. Status active; review 2026-11-17. NOT on HF.',
        ['.venv-indicf5/bin/python -m scripts.score_swar_cross_system'])
    (OUT / 'cases').mkdir()
    frontend = DualG2P(ROOT)
    engine = WordEvidenceEngine(ROOT, device='mps', pitch_method='dio', temporal_bins=20, retain_frame_evidence=True)
    pair = PhonePair(ROOT, state, device='mps', primary_engine=engine)

    def frames(system, job):
        row = produced[system][job['id']]
        if row['sha256'] != screened[system, job['id']]['sha256']:
            raise ValueError('ASR receipt refers to another waveform')
        samples = audio(PACKS[system], row)
        detail, arrays = engine.analyze(samples, job['text'])
        other = pair.encode_one(samples, 1)
        return ({'detail': detail, 'phone_log_probs': arrays['phone_log_probs']},
                {'detail': {**detail, 'phone_frame_seconds': len(samples) / 16000 / len(other)}, 'phone_log_probs': other})

    readings = []
    for context in selection['contexts']:
        current = [lookup[context['id'], v] for v in range(4)]
        native = {}
        for job in current:
            if screened[NATIVE, job['id']]['content']['decision'] == 'pass':
                try:
                    native[job['voice_ordinal']] = frames(NATIVE, job)
                except ValueError as error:
                    logging.warning('Reference abstention %s: %s', job['id'], error)
        case = []
        for job in current:
            row, check = produced[SWAR][job['id']], screened[SWAR, job['id']]
            reading = {'system': SWAR, 'job_id': job['id'], 'text_id': context['id'], 'voice_ordinal': job['voice_ordinal'], 'content': check.get('content'),
                'words': [], 'realized_error_label': None, 'model_sha256': active['model_sha256'],
                'model_text': row.get('generation_detail', {}).get('model_text'),
                'expected_word_positions': len(normalized_words(context['text'], engine.romanizer))}
            if row['status'] != 'ok_unreviewed':
                reading.update(status='unscored_generation_failure', error=row['error'])
            elif check['content']['decision'] == 'reject':
                reading.update(status='unscored_content_mismatch')
            else:
                reading['audio'] = {'path': str((PACKS[SWAR] / row['audio']).relative_to(ROOT)), 'sha256': row['sha256']}
                refs = [v for v in job['native_comparison_voice_ordinals'] if v in native]
                reading['reference_voice_ordinals_available'] = refs
                if not refs:
                    reading['status'] = 'unscored_reference_quality'
                else:
                    try:
                        query = frames(SWAR, job)
                        reading.update(status='scored_vowel_evidence', words=consensus(query, [(v, native[v]) for v in refs], frontend, engine, pair, networks, state))
                    except ValueError as error:
                        logging.warning('Measurement abstention %s: %s', job['id'], error)
                        reading.update(status='unscored_measurement_failure', error=str(error))
            reading['target_words'] = [w for w in reading['words'] if w['word'] == context['target_word']]
            reading['flagged_words'] = [w['word'] for w in reading['words'] if w['decision'] == 'strong_vowel_change']
            reading['tested_words'] = sum(any(s['qualified_references'] for s in w['sites']) for w in reading['words'])
            case.append(reading)
        (OUT / 'cases' / (context['id'] + '.json')).write_text(json.dumps({'context': context, 'readings': case, 'human_error_labels': None}, ensure_ascii=False, allow_nan=False) + '\n')
        readings.extend(case)
        logging.info('%s flagged %s', context['id'], [r['flagged_words'] for r in case])
    summary = {'requested': len(readings), 'statuses': dict(Counter(r['status'] for r in readings)),
        'content_decisions': dict(Counter(r['content']['decision'] if r['content'] else 'generation_failure' for r in readings)),
        'readings_with_word_flags': sum(bool(r['flagged_words']) for r in readings),
        'readings_with_focal_flags': sum(any(w['decision'] == 'strong_vowel_change' for w in r['target_words']) for r in readings),
        'focal_flags_on_content_pass_queries': sum(r['content'] is not None and r['content']['decision'] == 'pass' and any(w['decision'] == 'strong_vowel_change' for w in r['target_words']) for r in readings),
        'tested_word_positions': sum(r['tested_words'] for r in readings), 'word_positions_with_alignment': sum(len(r['words']) for r in readings),
        'word_positions_all_requested': sum(r['expected_word_positions'] for r in readings), 'accuracy': None}
    report = {'status': 'complete_frozen_swar_discovery', 'systems': {SWAR: summary}, 'model_sha256': sha(model), 'actual_hindi_error_accuracy': None,
        'limits': ['Flags are candidates, not confirmed errors.', 'Swar text passes through its own frontend; sampler matched to v80 IndicF5, silence removal off.']}
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Swar scoring failed explicitly; no model change')
        raise
