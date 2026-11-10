"""Replay the fixed human-word comparison plan with the frozen release heads."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.dual_g2p import DualG2P
from hindi_detector.dual_paired_inference import analyze_pair
from hindi_detector.intrinsic_inference import assign_decisions
from hindi_detector.task_vowel_detector import decisions, load, score
from scripts.extract_corrected_training import read_rows
from scripts.score_swar_sites import load_item

ROOT = Path(__file__).resolve().parents[1]


def one_word(item, ordinal):
    # Keep the original full-utterance phone frames/time basis. Only the word
    # identity list is narrowed; no splice, re-encoding or boundary change.
    return {**item, 'detail': {**item['detail'], 'words': [item['detail']['words'][ordinal]]}}


def main():
    out = ROOT / 'reports/release_native_variation_v37'
    if out.exists():
        raise FileExistsError(out)
    model = ROOT / 'reports/task_vowel_release_v31/model.pt'
    old = ROOT / 'reports/native_word_variation_v22/comparisons.json'
    source = json.loads(old.read_text())
    pairs = sorted({(row['query_case'], row['query_ordinal'], row['reference_case'], row['reference_ordinal']) for row in source})
    metadata = {row['id']: row for row in json.loads((ROOT / 'data/swar_speech_v18/selection.json').read_text())}
    primary = ROOT / 'reports/swar_corpus_features_v18'
    secondary = ROOT / 'reports/swar_phone_crosscheck_v20'
    pr = json.loads((primary / 'runtime.json').read_text())
    sr = json.loads((secondary / 'runtime.json').read_text())
    networks, state = load(model)
    for path, digest in state['code_sha256'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('Frozen dependency changed: ' + path)
    if pr['phone'] != state['phone_inventory'] or sr['phone'] != state['secondary_phone_inventory']:
        raise ValueError('Acoustic checkpoint identity differs')
    receipts = [{row['id']: row for row in read_rows(directory / 'items.jsonl') if row['kind'] == 'real'}
                for directory in (primary, secondary)]
    cache = {}

    def item(index, key, ordinal):
        identity = (index, key)
        if identity not in cache:
            cache[identity] = load_item((primary, secondary)[index], receipts[index][key])
        return one_word(cache[identity], ordinal)

    g2p = DualG2P(ROOT)
    records = []
    for qkey, qo, rkey, ro in pairs:
        if metadata[qkey]['voice'] == metadata[rkey]['voice']:
            raise ValueError('Fixed variation pair reused query voice group')
        words, sites, x = analyze_pair(item(0, qkey, qo), item(0, rkey, ro),
            item(1, qkey, qo), item(1, rkey, ro), g2p.word, g2p.ipa, pr['vocabulary'], sr['vocabulary'])
        probabilities = score(networks, state, x) if len(x) else []
        ds = decisions(state, x, probabilities) if len(x) else []
        assign_decisions(words, sites, probabilities, ds)
        records.append({'query_case': qkey, 'query_ordinal': qo, 'query_voice': metadata[qkey]['voice'],
            'reference_case': rkey, 'reference_ordinal': ro, 'reference_voice': metadata[rkey]['voice'],
            'word': words[0]['word'], 'decision': words[0]['decision'],
            'unavailable_reasons': words[0]['unavailable_reasons'],
            'sites': [{k: site[k] for k in ('site', 'native_phones', 'features', 'controlled_vowel_change_score', 'decision')}
                      for site in sites], 'actual_hindi_error_label': None})
    by_word = {}
    for word in sorted({row['word'] for row in records}):
        rows = [row for row in records if row['word'] == word]
        by_word[word] = {'comparison_pairs': len(rows), 'decisions': dict(Counter(row['decision'] for row in rows)),
            'query_voice_groups': len({row['query_voice'] for row in rows}),
            'reference_voice_groups': len({row['reference_voice'] for row in rows})}
    report = {'status': 'complete_release_native_variation', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'model_sha256': sha(model), 'fixed_pair_plan_sha256': sha(old), 'word_pairs': len(pairs),
        'word_families': len(by_word), 'decisions': dict(Counter(row['decision'] for row in records)),
        'site_comparisons': sum(len(row['sites']) for row in records),
        'site_flags': sum(site['decision'] == 'strong_vowel_change' for row in records for site in row['sites']),
        'by_word': by_word, 'native_error_false_positive_rate': None, 'actual_hindi_error_accuracy': None,
        'limits': ['Same word spoken in different contexts by different source voice groups. Legitimate pronunciation/prosody variation is possible.',
            'The metadata-only pair plan was fixed for the earlier model and is replayed without score-based pair selection.',
            'These paired disagreement counts are not a human-error false-positive rate.',
            'No new training, threshold tuning, source-label gold or voice embedding feature.']}
    write_group(out, {'report.json': report, 'comparisons.json': records,
        'source_snapshot.json': {'sha256': sha(Path(__file__)), 'source_utf8': Path(__file__).read_text()}},
        'Frozen release on the existing fixed102 human-word comparison pairs across18 families. '
        'Original full-frame/time coordinates and hashes checked; different source voice groups; '
        'observed disagreement is variation evidence, not native-error gold. No waveform/model/frame '
        'copies, no fitting or threshold update. Consumer native-variation-aware detector interpretation; '
        'review2026-11-10; retain compact counts/decisions while release is active. NOT on HF; inherited '
        'source/probe/MMS research usage restrictions apply.',
        ['.venv-indicf5/bin/python -m scripts.audit_release_native_variation', str(old)])
    print(json.dumps({k: report[k] for k in ('word_pairs', 'word_families', 'decisions', 'site_comparisons', 'site_flags')}))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception('Frozen Native-word variation audit failed explicitly')
        raise
