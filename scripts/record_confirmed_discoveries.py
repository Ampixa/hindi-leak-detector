"""Record the user's three explicit listening confirmations; no model changes."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {('real_yt_yt04_032', 14): 'हुनलाई', ('real_yt_yt07_053', 14): 'चलाउनको',
           ('real_yt_yt12_089', 7): 'नगरौ'}


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    group = ROOT / 'reports/listener_confirmations_v79'
    if group.exists():
        raise FileExistsError(group)
    active = json.loads((ROOT / 'active_vowel_model.json').read_text())
    directory = ROOT / active['model_directory']
    if sha(directory / 'model.pt') != active['model_sha256']:
        raise ValueError('Active model bytes differ')
    report_path = directory / 'review_report.json'
    report = json.loads(report_path.read_text())
    examples_path = ROOT / 'reports/more_listening_v77/examples.json'
    examples = json.loads(examples_path.read_text())
    if {(e['id'], e['word_ordinal']): e['word'] for e in examples} != TARGETS:
        raise ValueError('Actual listening examples differ from explicit user confirmation targets')
    if any(e['status'] != 'unconfirmed_candidate' or e['human_error_label'] is not None for e in examples):
        raise ValueError('Expected previous unconfirmed state differs')
    cases = {c['id']: c for c in json.loads((directory / 'actual_swar/cases.json').read_text())}
    for e in examples:
        for kind, interval in e['readings'].items():
            word = cases[e['id']]['analysis'][kind]['words'][e['word_ordinal']]
            if word['word'] != e['word'] or any(word[k] != interval[k] for k in ('start', 'end')):
                raise ValueError('Confirmed word/recording interval identity differs')
    old_checks = report['new_word_checks']
    pidh = [e for e in old_checks['cases'] if e['word'] == 'पिढको']
    if len(pidh) != 1 or pidh[0]['status'] != 'listener_similar':
        raise ValueError('Prior listener-similar check changed unexpectedly')
    now = datetime.now(timezone.utc).isoformat()
    feedback = {'recorded_utc': now, 'source': 'user conversation',
        'quote': 'they are conformed for hunalai , chalaunako, nagarau',
        'confirmations': [{'case_id': e['id'], 'word_ordinal': e['word_ordinal'], 'word': e['word'],
            'reading': 'swar', 'audio_sha256': e['readings']['swar']['audio_sha256'],
            'status': 'user_confirmed_pronunciation_anomaly', 'human_error_label': 1,
            'phonetic_boundary_manually_verified': False} for e in examples],
        'scope': 'User confirms these three previously flagged occurrences. One listener, detector-selected development discoveries; no independent precision/recall estimate, exact phone-boundary verification or general Hindi causality claim.',
        'total_confirmed_occurrences': 7, 'used_for_training': False, 'independent_test': False,
        'pidh_status': 'listener_similar; error label remains null', 'prediction_changes': False}
    write_group(group, {'feedback.json': feedback, 'previous_word_checks.json': old_checks,
        'previous_examples.json': examples, 'inputs.json': {'active_model_sha256': active['model_sha256'],
            'review_before_sha256': sha(report_path), 'examples_before_sha256': sha(examples_path),
            'source_script_sha256': sha(Path(__file__)), 'disk_free_before': shutil.disk_usage(ROOT).free}},
        'Record explicit user confirmations of हुनलाई, चलाउनको and नगरौ. feedback.json exact quote and waveform/word identities; previous_word_checks.json and previous_examples.json compact metadata rollback; inputs.json source/checkpoint hashes; applied.json verified after hashes and unchanged model. '
        'No audio, model, threshold, training or held-out-test changes. These are listener-confirmed discovery occurrences, not independent accuracy. Owner Ampixa/user; review2026-11-10, current consumers listening review and announcement draft. NOT on HF; original speech rights apply. '
        'Retain compact annotation provenance while confirmed examples are used; rollback snapshots disposable24h after review closes. Known copies Mac and k2.',
        ['.venv-indicf5/bin/python -m scripts.record_confirmed_discoveries'])
    for e in examples:
        e.update(status='user_confirmed', human_error_label=1, confirmation_source='reports/listener_confirmations_v79/feedback.json',
                 confirmation_recorded_utc=now, used_for_training=False, independent_test=False)
        identity = (e['id'], e['word_ordinal'])
        if any((c['id'], c['word_ordinal']) == identity for c in old_checks['cases']):
            raise ValueError('Confirmation target already present in live cards')
        old_checks['cases'].append({k: e[k] for k in ('id', 'word_ordinal', 'word', 'readings')} |
            {'status': 'user_confirmed', 'status_label': 'Confirmed by you',
             'explanation': 'You confirmed this previously flagged pronunciation anomaly in the Swar recording. The phone interval remains model-estimated.',
             'confirmation_source': 'reports/listener_confirmations_v79/feedback.json'})
    old_checks['summary'] = 'Seven occurrences now confirmed by you: हैन, किनकि, नेकपा, तलको, हुनलाई, चलाउनको and नगरौ. पिढको remains listener-similar; आइसके remains alignment-sensitive. These detector-selected discoveries are not an accuracy estimate.'
    old_checks['listener_confirmations_v79'] = {'path': 'reports/listener_confirmations_v79/feedback.json', 'sha256': sha(group / 'feedback.json')}
    old_checks['total_confirmed_occurrences'] = 7
    save(examples_path, examples)
    save(report_path, report)
    provenance_path = directory / 'review_provenance.json'
    provenance = json.loads(provenance_path.read_text())
    for member in provenance['files']:
        if member['path'] == 'review_report.json':
            member.update(bytes=report_path.stat().st_size, sha256=sha(report_path))
    provenance.setdefault('amendments', []).append({'recorded_utc': now, 'source': 'reports/listener_confirmations_v79/feedback.json',
        'sha256': sha(group / 'feedback.json'), 'scope': 'Three explicit user confirmations; display/annotation metadata only; no predictions changed.'})
    save(provenance_path, provenance)
    for readme in (directory / 'README.md', examples_path.parent / 'README.md'):
        with readme.open('a') as stream:
            stream.write('\n' + now + ': Codex GPT-6 /root recorded explicit user confirmations for हुनलाई, चलाउनको, नगरौ. Current examples are annotations, superseding creation-time unconfirmed wording. See reports/listener_confirmations_v79/feedback.json for quote and exact scope. No model/prediction/test changes. Current examples.json SHA ' + sha(examples_path) + '; live review_report.json SHA ' + sha(report_path) + '. Owner Ampixa/user; review2026-11-10.\n')
    if sha(directory / 'model.pt') != active['model_sha256']:
        raise AssertionError('Model unexpectedly changed during annotation update')
    save(group / 'applied.json', {'updated_utc': now, 'review_after_sha256': sha(report_path),
        'examples_after_sha256': sha(examples_path), 'active_model_sha256_unchanged': active['model_sha256'],
        'total_confirmed_occurrences': 7, 'prediction_changes': False, 'disk_free_after': shutil.disk_usage(ROOT).free})
    print(json.dumps(json.loads((group / 'applied.json').read_text())), flush=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Listener confirmation update failed explicitly; inspect recorded rollback before retrying')
        raise
