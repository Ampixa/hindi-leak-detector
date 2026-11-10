"""Count potential coverage without changing production predictions or mappings."""
from __future__ import annotations

from collections import Counter
import json
import logging
from pathlib import Path

from hindi_detector.artifacts import write_group
from hindi_detector.consonant_context import NonSyllabicTable, compile_context_word
from hindi_detector.corrected_synthesis import sha
from hindi_detector.dual_g2p import DualG2P
from hindi_detector.pronunciation_sites import compile_word

ROOT = Path(__file__).resolve().parents[1]


def summary(compiled):
    return {'variants': [{'variant_id': variant['variant_id'], 'phones': variant['phones'],
        'unknown_context_count': variant.get('unknown_context_count', 0),
        'contrasts': [{k: v for k, v in contrast.items() if k != 'alternative_unit'} for contrast in variant['contrasts']],
        'representations': variant['projections']} for variant in compiled['variants']],
        'hindi_representations': compiled.get('hindi_projections'),
        'normalization': compiled.get('input_normalization'),
        'has_contrast': any(variant['contrasts'] for variant in compiled['variants'])}


def main():
    out = ROOT / 'reports/consonant_coverage_v58'
    if out.exists():
        raise FileExistsError(out)
    selected = ROOT / 'reports/task_vowel_release_v31'
    cases = json.loads((selected / 'actual_swar/cases.json').read_text())
    native = [word for case in cases for word in case['analysis'].get('real', {}).get('words', [])]
    counts = Counter(word['word'] for word in native)
    frontend = DualG2P(ROOT)
    table = NonSyllabicTable(ROOT / 'data/panphon_context_v58')
    vocabularies = [json.loads((ROOT / directory / 'vocab.json').read_text())
                    for directory in ('models/phoneme_probe', 'models/phoneme_probe_lv60_v20')]
    inventories = []
    for vocabulary in vocabularies:
        _, pool = table.pool(vocabulary)
        inventories.append({'pool': pool, 'token_classification': {t: table.classify(t) for t, i in vocabulary.items() if i > 3}})
    results = []
    for word, occurrences in sorted(counts.items()):
        raw = frontend.word(word)
        record = {'word': word, 'occurrences': occurrences, 'source_G2P': raw}
        for name in ('strict', 'unknown_context'):
            probes = []
            for vocabulary in vocabularies:
                try:
                    compiled = compile_word(raw, frontend.ipa, vocabulary) if name == 'strict' else compile_context_word(raw, frontend.ipa, vocabulary, table)
                    probes.append({'status': 'compiled', **summary(compiled)})
                except ValueError as error:
                    probes.append({'status': 'explicit_unsupported', 'reason': str(error), 'has_contrast': False})
            record[name] = {'probes': probes, 'both_compiled': all(p['status'] == 'compiled' for p in probes),
                            'both_have_contrast': all(p['has_contrast'] for p in probes)}
        results.append(record)
    report = {'status': 'complete_potential_context_coverage_audit', 'actual_current_tested_word_positions': sum(bool(w['sites']) for w in native),
        'actual_total_word_positions': len(native), 'unique_words': len(counts),
        'potential_positions_with_contrast': {name: sum(row['occurrences'] for row in results if row[name]['both_have_contrast'])
                                            for name in ('strict', 'unknown_context')},
        'potential_new_positions': sum(row['occurrences'] for row in results if row['unknown_context']['both_have_contrast'] and not row['strict']['both_have_contrast']),
        'new_unique_words': [row['word'] for row in results if row['unknown_context']['both_have_contrast'] and not row['strict']['both_have_contrast']],
        'strict_primary_failure_counts_by_occurrence': dict(Counter({reason: sum(row['occurrences'] for row in results
            if row['strict']['probes'][0].get('reason') == reason) for reason in {row['strict']['probes'][0].get('reason') for row in results}
            if reason is not None})),
        'counts_are_acoustic_measurements': False, 'production_changed': False, 'actual_hindi_error_accuracy': None,
        'limits': ['Compilation is a possible hypothesis, not measured qualification or correct pronunciation.',
                   'The broad context class never asserts identity of an unsupported consonant.',
                   'Vowel checks and G2P lexical alignment stay distinct from the pooled consonant acoustic emissions.',
                   'No promotion before controlled-change and Native-reference false-flag checks.']}
    write_group(out, {'report.json': report, 'words.json': results, 'acoustic_classes.json': inventories,
        'inputs.json': {'actual_cases_sha256': sha(selected / 'actual_swar/cases.json'),
            'model_sha256': sha(selected / 'model.pt'), 'feature_resource_sha256': sha(ROOT / 'data/panphon_context_v58/inventory.json')},
        'source_snapshot.json': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()}
                               for p in (Path(__file__), ROOT / 'hindi_detector/consonant_context.py')]},
        'Research-only compilation coverage for all actual90-case words. report.json distinguishes '
        'delivered measured coverage from merely possible contrast positions; words.json records '
        'actual G2P inputs, every compiled contrast and explicit failure/normalization. '
        'acoustic_classes.json audits every observed probe token and pure nonsyllabic pool. '
        'Pinned PanPhon data used as a class-membership source only, licence retained in '
        'data/panphon_context_v58. Consonant identity remains unknown, never approximated '
        'as another consonant. No production source, model, thresholds or predictions changed. '
        'Current consumer feasibility decision and controlled waveform audit, review2026-11-10. '
        'Retain compact source/data pins while current goal active; no new audio/weights/HF upload.',
        ['.venv-indicf5/bin/python -m scripts.audit_consonant_coverage'])
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Unknown-consonant coverage feasibility failed explicitly')
        raise
