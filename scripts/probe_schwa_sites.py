"""Measure timed schwa-site hypotheses on actual native and synthesized words."""
from __future__ import annotations

import argparse
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
from hindi_detector.review_server import read_items
from hindi_detector.schwa_probe import compare_site, site_paths
from hindi_detector.word_evidence import WordEvidenceEngine

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pack', type=Path, action='append', required=True)
    parser.add_argument('--evidence', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Need a new immutable schwa-site probe')
    registry = {}
    for pack in args.pack:
        items, _ = read_items(pack)
        for key, row in items.items():
            if key in registry:
                raise ValueError('Duplicate probe source clip')
            registry[key] = (pack, row)
    candidates = json.loads((ROOT / 'data/schwa_candidates_v1/candidates.json').read_text())
    lookup = defaultdict(list)
    for candidate in candidates:
        lookup[candidate['word']].append(candidate)
    ipa_path = ROOT / '.sources/frontend/data/frontend/ipa_map.tsv'
    with ipa_path.open() as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        if reader.fieldnames != ['phone', 'ipa_candidate', 'status', 'notes']:
            raise ValueError('IPA map schema changed')
        ipa = {row['phone']: row['ipa_candidate'] for row in reader}
    source_files = [Path(__file__), ROOT / 'hindi_detector/schwa_probe.py', ROOT / 'hindi_detector/ctc.py']
    snapshots = [{'path': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'source_utf8': p.read_text()} for p in source_files]
    engine = WordEvidenceEngine(ROOT)
    results, inputs, seen = [], [], set()
    for directory in args.evidence:
        manifest_path = directory / 'manifest.json'; manifest = json.loads(manifest_path.read_text())
        inputs.append({'path': str(manifest_path.resolve()), 'sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest()})
        for item in manifest['items']:
            if item['status'] != 'ok_unreviewed_word_evidence':
                continue
            path = (directory / item['detail']).resolve(strict=True)
            if not path.is_relative_to(directory.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != item['detail_sha256']:
                raise ValueError('Word timestamps/details failed identity check')
            detail = json.loads(path.read_text())
            targets = [(i, w) for i, w in enumerate(detail['words']) if w['word'] in lookup]
            if not targets:
                continue
            if item['clip_id'] in seen:
                raise ValueError('Repeated word evidence clip')
            seen.add(item['clip_id'])
            pack, row = registry[item['clip_id']]; audio = pack / row['audio']
            if hashlib.sha256(audio.read_bytes()).hexdigest() != item['audio_sha256'] or item['audio_sha256'] != row['sha256']:
                raise ValueError('Schwa probe audio/evidence mismatch')
            wave, sr = sf.read(audio, dtype='float32', always_2d=True)
            if wave.shape[1] != 1 or not np.isfinite(wave).all() or not .2 <= len(wave) / sr <= 30:
                raise ValueError('Need finite mono speech within the verified engine duration')
            divisor = int(np.gcd(sr, 16000))
            samples = resample_poly(wave[:, 0], 16000 // divisor, sr // divisor).astype('float32') if sr != 16000 else wave[:, 0]
            with engine.torch.inference_mode():
                output = engine.phone(**engine.extractor(samples, sampling_rate=16000, return_tensors='pt')).logits[0]
                log_probs = output.log_softmax(-1).cpu().numpy()
            seconds = len(samples) / 16000
            for index, word in targets:
                start, end = max(0., word['start'] - .03), min(seconds, word['end'] + .03)
                cropped = log_probs[int(start / seconds * len(log_probs)):max(1, int(np.ceil(end / seconds * len(log_probs))))]
                for candidate in lookup[word['word']]:
                    record = {k: item[k] for k in ['clip_id', 'system', 'speaker_id', 'sentence_id', 'split', 'audio_sha256']}
                    record.update(word=word['word'], word_index=index, candidate_id=candidate['candidate_id'],
                                  start=word['start'], end=word['end'], boundary_context_seconds=.03)
                    try:
                        paths = site_paths(candidate, ipa, engine.vocab)
                        record.update(status='ok_unreviewed_counterfactual', sites=[compare_site(cropped, p) for p in paths])
                    except ValueError as error:
                        # Unknown IPA or impossible CTC paths are counted exclusions, never native evidence.
                        record.update(status='explicitly_unsupported_or_impossible', error=str(error))
                    results.append(record)
            LOG.info('Probed actual word sites in %s (%s candidate words)', row['clip_id'], len(targets))
    report = {'status': 'temporal_schwa_site_hypotheses_measured_not_validated_detector', 'clips_with_candidate_words': len(seen),
              'candidate_word_occurrences': len(results), 'supported': sum(r['status'] == 'ok_unreviewed_counterfactual' for r in results),
              'excluded': sum(r['status'] != 'ok_unreviewed_counterfactual' for r in results),
              'inputs': inputs, 'ipa_map_sha256': hashlib.sha256(ipa_path.read_bytes()).hexdigest(),
              'phone_revision': engine.phone_inventory['revision'], 'validated_schwa_accuracy': None,
              'limits': ['Lexicon paths are unreviewed hypotheses, including known questionable entries; no approved gold is fabricated.',
                         'Each site changes its schwa count while other project phones stay fixed; non-schwa IPA is never collapsed to fit the probe.',
                         'Three central-vowel qualities are nuisance alternatives, equally marginalized within each path family.',
                         'CTC likelihoods can favor short paths and the phone model may be wrong on Nepali. Native references are needed to audit this bias.',
                         'Unsupported vocabulary/impossible paths are counted exclusions, never low-risk predictions.',
                         'The old whole-clip proxy is not used as evidence for a word or schwa site.']}
    write_group(args.out, {'report.json': report, 'word_sites.json': results, 'source_snapshot.json': snapshots},
                'Inspect exact temporal CTC hypotheses at candidate vowel sites and audit them against multi-speaker native readings.',
                [str(ipa_path.resolve()), 'data/schwa_candidates_v1/candidates.json',
                 'python -m scripts.probe_schwa_sites --pack PACK --evidence WORD_EVIDENCE --out NEW'])
    print(json.dumps(report))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Temporal schwa-site probe failed; no fabricated acoustic evidence')
        raise
