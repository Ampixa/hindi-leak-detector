"""Frozen release against separate teacher references, including human variation."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import traceback

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.dual_g2p import DualG2P
from hindi_detector.dual_paired_inference import analyze_pair
from hindi_detector.intrinsic_inference import assign_decisions
from hindi_detector.task_vowel_detector import decisions, load, score
from scripts.detect_nepali_leakage import reference_qualified
from scripts.extract_corrected_training import read_rows
from scripts.score_swar_sites import load_item

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)


def main():
    out = ROOT / 'reports/release_teacher_consensus_v36'
    if out.exists():
        raise FileExistsError(out)
    model = ROOT / 'reports/task_vowel_release_v31/model.pt'
    primary = ROOT / 'reports/swar_corpus_features_v18'
    secondary = ROOT / 'reports/swar_phone_crosscheck_v20'
    reference = ROOT / 'reports/swar_teacher_features_v21'
    second_reference = ROOT / 'reports/swar_teacher_secondary_v36'
    content = ROOT / 'reports/swar_teacher_content_v21'
    paths = [directory / 'items.jsonl' for directory in (primary, secondary, reference, second_reference, content)]
    write_group(out, {'protocol.json': {'model_sha256': sha(model),
        'inputs': [{'path': str(path.relative_to(ROOT)), 'sha256': sha(path)} for path in paths],
        'selection': 'All86 available same-text cases; two original cached teacher conditioning groups; all3 recording sources',
        'reference_policy': 'Global ASR content pass plus both-probe site qualification; ignore unqualified; conflict abstains. Report reference-count support.',
        'interpretation': 'Real human disagreement with teacher is observed pronunciation variation, not automatically an error or false-positive label.',
        'frozen_model': True, 'training_or_threshold_updates': False},
        'source_snapshot.json': [{'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'source_utf8': path.read_text()}
            for path in (Path(__file__), ROOT / 'scripts/detect_nepali_leakage.py')]},
        'Actual recording versus separately voiced teacher reference audit. items.jsonl: every '
        'paired site decision/feature114 and qualification; words.json: conservative reference '
        'consensus and per-word untested/uncertain states for all3 sources; report.json: source/voice '
        'counts, development regressions and flags. status.json live owned PID/phase. No raw frame, '
        'waveform or model copies. Human flags are variation/candidate disagreements, never invented '
        'native-error FPR. Current consumer deployment reference sensitivity; review2026-11-10, '
        'retain compact decisions while release active, redundant comparisons expire24h after '
        'consumer closes. NOT on HF; inherited source/model/MMS research restrictions apply.',
        ['.venv-indicf5/bin/python -m scripts.audit_release_teacher_consensus'])

    def status(phase, **values):
        path = out / 'status.json'
        temp = path.with_suffix('.partial')
        temp.write_text(json.dumps({'phase': phase, 'pid': os.getpid(),
            'updated_utc': datetime.now(timezone.utc).isoformat(), **values}, indent=2) + '\n')
        temp.replace(path)

    try:
        networks, state = load(model)
        for path, digest in state['code_sha256'].items():
            if sha(ROOT / path) != digest:
                raise ValueError('Frozen model dependency changed: ' + path)
        for directory in (secondary, second_reference):
            runtime = json.loads((directory / 'runtime.json').read_text())
            if runtime['phone'] != state['secondary_phone_inventory']:
                raise ValueError('Secondary acoustic runtime differs')
        pr = json.loads((primary / 'runtime.json').read_text())
        rr = json.loads((reference / 'runtime.json').read_text())
        sr = json.loads((secondary / 'runtime.json').read_text())
        if pr['phone'] != rr['primary_phone'] or pr['phone'] != state['phone_inventory'] or pr['alignment'] != rr['alignment']:
            raise ValueError('Primary teacher/query acoustic alignment runtime differs')
        synthesis = json.loads((ROOT / 'data/swar_teacher_v21/teacher/runtime.json').read_text())
        if synthesis['loaded_layers'] != 24 or synthesis['weights_sha256'] != '2b4636e3aed0e4f2d035630e504be95fc58f494b690a851a91bfe0dd8299516f':
            raise ValueError('Wrong teacher source')
        references = defaultdict(list)
        for row in read_rows(reference / 'items.jsonl'):
            references[row['text_id']].append(row)
        r2 = {row['id']: row for row in read_rows(second_reference / 'items.jsonl')}
        q1 = {(row['id'], row['kind']): row for row in read_rows(primary / 'items.jsonl')}
        q2 = {(row['id'], row['kind']): row for row in read_rows(secondary / 'items.jsonl')}
        cq = {row['id']: row['content_quality']['decision'] for row in read_rows(content / 'items.jsonl')}
        if set(r2) != {row['id'] for rows in references.values() for row in rows}:
            raise ValueError('Incomplete measured teacher secondary frames')
        metadata = {row['id']: row for row in json.loads((ROOT / 'data/swar_speech_v18/selection.json').read_text())}
        g2p = DualG2P(ROOT)
        all_words = []
        comparisons = 0
        for number, (key, refs) in enumerate(sorted(references.items())):
            if len(refs) != 2:
                raise ValueError('Need exactly two original conditioning references per text')
            queries = {source: (load_item(primary, q1[key, source]), load_item(secondary, q2[key, source]))
                       for source in ('real', 'swar', 'a450')}
            grouped = defaultdict(list)
            per_reference_words = defaultdict(list)
            for ref_row in refs:
                r = load_item(reference, ref_row)
                s = load_item(second_reference, r2[ref_row['id']])
                for source, (q, qs) in queries.items():
                    words, sites, x = analyze_pair(q, r, qs, s, g2p.word, g2p.ipa, pr['vocabulary'], sr['vocabulary'])
                    probabilities = score(networks, state, x) if len(x) else []
                    ds = decisions(state, x, probabilities) if len(x) else []
                    assign_decisions(words, sites, probabilities, ds)
                    per_reference_words[source].append(words)
                    for site in sites:
                        identity = (source, site['word_ordinal'], site['site']['kind'], site['site']['index'], tuple(site['native_phones']))
                        compact = {'id': key, 'source': source, 'word': words[site['word_ordinal']]['word'],
                            'word_ordinal': site['word_ordinal'], 'site': site['site'], 'native_phones': site['native_phones'],
                            'reference_id': ref_row['id'], 'reference_speaker': ref_row['speaker_id'],
                            'reference_content_decision': cq[ref_row['id']],
                            'reference_qualified': bool(cq[ref_row['id']] == 'pass' and reference_qualified(site)),
                            'score': site['controlled_vowel_change_score'], 'decision': site['decision'], 'features': site['features']}
                        grouped[identity].append(compact)
                        with (out / 'items.jsonl').open('a') as handle:
                            handle.write(json.dumps(compact, ensure_ascii=False, allow_nan=False) + '\n')
                        comparisons += 1
            for source, (q, _) in queries.items():
                for ordinal, word in enumerate(q['detail']['words']):
                    sites = []
                    for identity, rows in grouped.items():
                        if identity[:2] != (source, ordinal):
                            continue
                        valid = [row for row in rows if row['reference_qualified']]
                        values = {row['decision'] for row in valid}
                        decision = next(iter(values)) if len(values) == 1 else 'uncertain'
                        if not valid:
                            decision = 'unsupported_reference'
                        sites.append({'site': rows[0]['site'], 'native_phones': rows[0]['native_phones'],
                            'decision': decision, 'qualified_references': len(valid), 'available_references': len(rows),
                            'reference_decisions': [{k: row[k] for k in ('reference_id', 'reference_qualified', 'decision', 'score')} for row in rows]})
                    values = {site['decision'] for site in sites}
                    decision = 'strong_vowel_change' if 'strong_vowel_change' in values else 'uncertain' if values & {'uncertain', 'unsupported_reference'} else 'no_strong_vowel_change' if values else 'untested_site'
                    variant_ambiguous = any(not ref_words[ordinal]['sites'] and not ref_words[ordinal]['unavailable_reasons'] for ref_words in per_reference_words[source])
                    if decision == 'strong_vowel_change' and variant_ambiguous:
                        decision = 'uncertain'
                    all_words.append({'id': key, 'source': source, 'voice': metadata[key]['voice'],
                        'word_ordinal': ordinal, 'word': word['word'], 'start': word['start'], 'end': word['end'],
                        'decision': decision, 'sites': sites, 'native_variant_ambiguity': variant_ambiguous,
                        'actual_hindi_error_label': None})
            status('measuring_teacher_reference_consensus', cases=number + 1, planned=len(references), comparisons=comparisons)
            if (number + 1) % 10 == 0:
                LOG.info('Measured teacher consensus %s/%s', number + 1, len(references))
        counts = Counter(row['source'] + '/' + row['decision'] for row in all_words)
        flags = [row for row in all_words if row['decision'] == 'strong_vowel_change']
        regressions = []
        for label in json.loads((ROOT / 'data/swar_phone_plan_v18/regressions.json').read_text()):
            matches = [row for row in all_words if all(row[k] == label[k] for k in ('id', 'source', 'word'))]
            regressions.append({**label, 'flagged': any(row['decision'] == 'strong_vowel_change' for row in matches)})
        report = {'status': 'complete_release_teacher_consensus', 'model_sha256': sha(model),
            'cases': len(references), 'voice_groups': len({metadata[key]['voice'] for key in references}),
            'references': len(r2), 'site_comparisons': comparisons, 'word_decisions': dict(counts),
            'flagged_words': flags, 'regressions': regressions,
            'reference_content_decisions': dict(Counter(cq.values())),
            'actual_hindi_error_accuracy': None, 'native_error_false_positive_rate': None,
            'limits': ['Human flags quantify disagreement with generated references, not unacceptable Native pronunciation.',
                'Teacher/probe/G2P mistakes and natural reductions can cause disagreement; global ASR pass is not pronunciation gold.',
                'All six picks/two known errors were previously inspected; this is reference sensitivity, not independent accuracy.',
                'Both phone probes share training-family and MMS/G2P biases.']}
        (out / 'words.json').write_text(json.dumps(all_words, ensure_ascii=False) + '\n')
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        status(report['status'], cases=len(references), comparisons=comparisons)
        print(json.dumps({'cases': len(references), 'word_decisions': dict(counts), 'regressions': regressions}, ensure_ascii=False))
    except Exception:
        status('failed_explicitly', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Frozen release teacher-consensus audit failed explicitly')
        raise
