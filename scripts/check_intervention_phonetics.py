"""Check renderer interventions with an independent phone probe, not accent gold."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from hindi_detector.artifacts import write_group
from hindi_detector.schwa_probe import compare_site, site_paths
from hindi_detector.word_evidence import WordEvidenceEngine

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    pack = ROOT / 'data/schwa_interventions_pilot_v7'
    out = ROOT / 'reports/intervention_acoustic_check_v7'
    if out.exists():
        raise FileExistsError('Do not overwrite an intervention acoustic check')
    receipts = json.loads((pack / 'result/receipts.json').read_text())
    candidates = {r['candidate_id']: r for r in json.loads((ROOT / 'data/schwa_candidates_v1/candidates.json').read_text())}
    with (ROOT / '.sources/frontend/data/frontend/ipa_map.tsv').open() as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        if reader.fieldnames != ['phone', 'ipa_candidate', 'status', 'notes']:
            raise ValueError('Pinned IPA candidate-map schema changed')
        ipa = {r['phone']: r['ipa_candidate'] for r in reader}
    engine = WordEvidenceEngine(ROOT)
    rows = []; groups = defaultdict(dict)
    for receipt in receipts:
        path = (pack / receipt['audio']).resolve(strict=True)
        if not path.is_relative_to(pack.resolve()) or sha(path) != receipt['sha256']:
            raise ValueError('Intervention waveform checksum/path mismatch')
        wave, rate = sf.read(path, dtype='float32', always_2d=True)
        if wave.shape[1] != 1 or not np.isfinite(wave).all():
            raise ValueError('Need finite mono intervention audio')
        divisor = int(np.gcd(rate, 16000))
        samples = resample_poly(wave[:, 0], 16000 // divisor, rate // divisor).astype('float32')
        with engine.torch.inference_mode():
            logits = engine.phone(**engine.extractor(samples, sampling_rate=16000, return_tensors='pt')).logits[0]
            log_probs = logits.log_softmax(-1).cpu().numpy()
        paths = site_paths(candidates[receipt['candidate_id']], ipa, engine.vocab)
        if len(paths) != 1:
            raise ValueError('Pilot expected exactly one candidate site')
        score = compare_site(log_probs, paths[0])
        predicted = log_probs.argmax(axis=1)
        greedy = [engine.tokens[int(token)] for i, token in enumerate(predicted)
                  if token not in engine.special and (i == 0 or token != predicted[i - 1])]
        durations = np.asarray(receipt['actual_duration_output'], dtype=float)
        if durations.shape != (1, len(receipt['native_encoded_input_ids'])) or not np.isfinite(durations).all() or np.any(durations < 0):
            raise ValueError('Renderer duration shape/range differs from inspected pilot')
        rounded_steps = int(np.ceil(durations).sum())
        record = {'word': receipt['word'], 'speaker': receipt['speaker'], 'condition': receipt['condition'],
                  'audio': receipt['audio'], 'audio_sha256': receipt['sha256'],
                  'probe_greedy_sequence': greedy, 'site_evidence': score,
                  'ceil_duration_steps': rounded_steps,
                  'inferred_samples_per_step': receipt['audio_frames'] // rounded_steps if rounded_steps and receipt['audio_frames'] % rounded_steps == 0 else None,
                  'probe_revision': engine.phone_inventory['revision'],
                  'actual_hindi_leakage_label': None, 'human_pronunciation_gold': False}
        rows.append(record)
        key = (receipt['candidate_id'], receipt['speaker'])
        if receipt['condition'] in groups[key]:
            raise ValueError('Duplicate intervention condition')
        groups[key][receipt['condition']] = score['marginal_alternative_minus_lexicon']
        LOG.info('Independent phone check %s/%s/%s: %.3f', receipt['word'], receipt['speaker'], receipt['condition'], score['marginal_alternative_minus_lexicon'])
    pairs = []
    for key, values in sorted(groups.items()):
        if set(values) != {'lexicon_path', 'site_count_alternative'}:
            raise ValueError('Unpaired intervention audio')
        difference = values['site_count_alternative'] - values['lexicon_path']
        pairs.append({'candidate_id': key[0], 'speaker': key[1], 'lexicon_request_score': values['lexicon_path'],
                      'alternate_request_score': values['site_count_alternative'],
                      'alternate_minus_lexicon_request_score': difference,
                      'probe_orders_requests_as_expected': difference > 0})
    report = {'status': 'independent_probe_intervention_check_not_human_leakage_accuracy',
              'clips': len(rows), 'paired_interventions': len(pairs),
              'pairs_ordered_as_requested': sum(r['probe_orders_requests_as_expected'] for r in pairs),
              'observed_duration_step_sizes': sorted({r['inferred_samples_per_step'] for r in rows if r['inferred_samples_per_step'] is not None}),
              'duration_outputs_all_have_integral_sample_step': all(r['inferred_samples_per_step'] is not None for r in rows),
              'actual_hindi_accuracy': None, 'native_acceptability_accuracy': None,
              'limits': ['Independent probe agreement is a renderer/input-adherence sanity check, not verified realization labels.',
                         'The candidate site alternative is not automatically a Hindi error.',
                         'Whole-word likelihood can be biased by vowel quality, phone count and probe coverage.',
                         'Ceiled-duration/sample correspondence is measured; no manually verified acoustic phone boundaries exist.']}
    sources = [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()}
               for p in (Path(__file__), ROOT / 'hindi_detector/schwa_probe.py', ROOT / 'hindi_detector/ctc.py', ROOT / 'hindi_detector/word_evidence.py')]
    write_group(out, {'report.json': report, 'clip_evidence.json': rows, 'pairs.json': pairs, 'source_snapshot.json': sources},
                'Test whether same-voice one-site requests produce independently distinguishable acoustic evidence before larger intervention training.',
                ['Pilot receipts SHA ' + sha(pack / 'result/receipts.json'), 'Pinned phoneme probe inventory',
                 '.venv-indicf5/bin/python -m scripts.check_intervention_phonetics'])
    print(json.dumps(report))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Independent intervention check failed; no gold labels or substitute evidence')
        raise
