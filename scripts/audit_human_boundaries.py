"""Audit fixed boundary perturbations against unchanged cached human evidence."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import traceback

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from hindi_detector.artifacts import write_group
from hindi_detector.boundary_audit import WINDOWS, audit_boundaries
from hindi_detector.word_evidence import WordEvidenceEngine

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-clips', type=int)
    args = parser.parse_args()
    if args.max_clips is not None and args.max_clips < 1:
        raise ValueError('Clip limit must be positive when supplied')
    if not (args.out / 'README.md').is_file() or (args.out / 'items.jsonl').exists():
        raise ValueError('Need a documented fresh boundary audit')
    bank_path = args.bank / 'word_bank.json'
    bank = json.loads(bank_path.read_text())
    selected = json.loads((args.bank / 'selection.json').read_text())
    clips = {row['clip_id']: row for row in selected['clips']}
    allowed = {row['word']: {e['audio'] for e in row['human_examples']} for row in bank}
    manifest_path = args.evidence / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['g2p_bank_sha256'] != sha(bank_path) or manifest['failures']:
        raise ValueError('Need complete unchanged human G2P evidence')
    if {r['clip_id'] for r in manifest['items']} != set(clips):
        raise ValueError('Human clips and evidence do not agree')
    items = manifest['items'][:args.max_clips] if args.max_clips is not None else manifest['items']
    files = [Path(__file__), ROOT / 'hindi_detector/boundary_audit.py', ROOT / 'hindi_detector/g2p_evidence.py',
             ROOT / 'hindi_detector/ctc.py', ROOT / 'hindi_detector/word_evidence.py']
    sources = [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()} for p in files]
    (args.out / 'source_snapshot.json').write_text(json.dumps(sources, indent=2) + '\n')

    def status(phase: str, processed: int, failures: int) -> None:
        path = args.out / 'status.json'; temporary = path.with_suffix('.partial')
        temporary.write_text(json.dumps({'status': phase, 'processed_clips': processed,
                                        'planned_clips': len(items), 'failed_clips': failures,
                                        'updated_utc': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')
        temporary.replace(path)

    status('loading_verified_phone_probe', 0, 0)
    engine = WordEvidenceEngine(ROOT, g2p_word_bank=bank)
    if engine.vocab != manifest['phone_vocabulary'] or engine.phone_inventory['revision'] != manifest['phone_encoder_revision']:
        raise ValueError('Boundary audit phone probe differs from original feature extraction')
    results = []; all_words = []; differences = []
    for item in items:
        row = clips[item['clip_id']]
        result = {k: row[k] for k in ('clip_id', 'speaker_id', 'system', 'sha256')}
        try:
            if row['system'] != 'human_nepali_unlabeled' or row['sha256'] != item['audio_sha256']:
                raise ValueError('Human source identity/unknown-label role changed')
            detail_path = (args.evidence / item['detail']).resolve(strict=True)
            if not detail_path.is_relative_to(args.evidence.resolve()) or sha(detail_path) != item['detail_sha256']:
                raise ValueError('Cached word timing/evidence checksum mismatch')
            detail = json.loads(detail_path.read_text())
            path = Path(row['audio']).resolve(strict=True)
            if not path.is_relative_to((ROOT.parent / 'indicvoices_r_ne/audio').resolve()) or sha(path) != row['sha256']:
                raise ValueError('Human source waveform changed or left declared corpus')
            wave, rate = sf.read(path, dtype='float32', always_2d=True)
            if wave.shape[1] != 1 or not np.isfinite(wave).all():
                raise ValueError('Need finite, explicitly staged mono speech')
            divisor = int(np.gcd(rate, 16000))
            samples = resample_poly(wave[:, 0], 16000 // divisor, rate // divisor).astype('float32') if rate != 16000 else wave[:, 0]
            seconds = len(samples) / 16000
            with engine.torch.inference_mode():
                logits = engine.phone(**engine.extractor(samples, sampling_rate=16000, return_tensors='pt')).logits[0]
                log_probs = logits.log_softmax(-1).cpu().numpy()
            words = []
            for index, word in enumerate(detail['words']):
                if word['word'] not in allowed or row['audio'] not in allowed[word['word']]:
                    continue
                audit = audit_boundaries(log_probs, engine.g2p_hypotheses[word['word']],
                                         start=word['start'], end=word['end'], seconds=seconds)
                cached = word['g2p_path_evidence']['hindi_minus_nepali_log_likelihood']
                repeated = audit['windows'][0]['hindi_minus_nepali_log_likelihood'] if audit['windows'] else None
                if (cached is None) != (repeated is None):
                    raise ValueError('Repeated baseline and cached evidence support disagree')
                difference = abs(cached - repeated) if cached is not None else None
                if difference is not None and difference > 1e-4:
                    raise ValueError('Repeated phone baseline failed numeric reproduction tolerance')
                record = {'clip_id': row['clip_id'], 'speaker_id': row['speaker_id'], 'word': word['word'],
                          'word_index': index, 'start': word['start'], 'end': word['end'],
                          'alignment_score': word['alignment_score'],
                          'shorter_than_80ms_warning': word['end'] - word['start'] < .08,
                          'baseline_cached_difference': difference, 'boundary_audit': audit,
                          'native_acceptability_label': None, 'hindi_leakage_label': None}
                words.append(record)
            # Commit a clip only after all planned-word baseline checks pass.
            differences.extend(w['baseline_cached_difference'] for w in words if w['baseline_cached_difference'] is not None)
            all_words.extend(words)
            result.update(status='ok_unreviewed_boundary_audit', words=words)
        except Exception:
            result.update(status='failed_explicitly_excluded', error=traceback.format_exc())
            LOG.exception('Boundary audit failed for %s; no replacement score', row['clip_id'])
        with (args.out / 'items.jsonl').open('a') as handle:
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
        results.append(result)
        failed = sum(r['status'] != 'ok_unreviewed_boundary_audit' for r in results)
        status('auditing_fixed_word_windows', len(results), failed)
        LOG.info('Boundary audit %s/%s (%s failures)', len(results), len(items), failed)
    grouped = defaultdict(list)
    for word in all_words:
        grouped[word['word']].append(word)
    word_summary = []
    for word, records in sorted(grouped.items()):
        word_summary.append({'word': word, 'occurrences': len(records),
                             'source_groups': len({r['speaker_id'] for r in records}),
                             'statuses': dict(Counter(r['boundary_audit']['status'] for r in records)),
                             'source_groups_with_direction_changes': len({r['speaker_id'] for r in records if r['boundary_audit']['direction_changes']}),
                             'short_span_warnings': sum(r['shorter_than_80ms_warning'] for r in records)})
    failures = sum(r['status'] != 'ok_unreviewed_boundary_audit' for r in results)
    report = {'status': 'fixed_boundary_sensitivity_measured_not_hindi_accuracy', 'clips': len(results),
              'selected_corpus_clips': len(clips), 'explicit_clip_limit': args.max_clips,
              'failed_clips': failures, 'planned_word_occurrences': len(all_words),
              'statuses': dict(Counter(r['boundary_audit']['status'] for r in all_words)),
              'baseline_comparisons_reproduced': len(differences),
              'maximum_baseline_absolute_difference': max(differences) if differences else None,
              'shorter_than_80ms_warnings': sum(r['shorter_than_80ms_warning'] for r in all_words),
              'fixed_window_recipes': WINDOWS, 'hindi_leakage_accuracy': None,
              'native_acceptability_accuracy': None, 'preferred_style_accuracy': None,
              'bank_sha256': sha(bank_path), 'evidence_manifest_sha256': sha(manifest_path),
              'limits': ['Changing sign across fixed windows shows fragile path evidence, not which pronunciation is correct.',
                         'Consistent sign does not prove the alignment, G2P or probe is correct.',
                         'Wider windows can contain neighboring speech; all fixed windows stay visible and none is selected by score.',
                         'Unsupported phones and impossible paths remain explicit, never native support or zero risk.',
                         'All human error/style labels remain unknown; geography is not a feature or label.']}
    write_group(args.out / 'result', {'report.json': report, 'word_summary.json': word_summary},
                'Quantify sensitivity of human dual-G2P comparisons to fixed word-window changes before calibration.',
                ['Parent items.jsonl and source_snapshot.json', str(bank_path.resolve()), str(manifest_path.resolve()),
                 'python -m scripts.audit_human_boundaries --bank data/human_g2p_bank_v5 --evidence reports/human_g2p_evidence_v5 --out NEW'])
    status('boundary_audit_complete_with_explicit_coverage' if not failures else 'boundary_audit_complete_with_failed_clips', len(results), failures)
    print(json.dumps(report))
    if failures:
        raise RuntimeError('Some boundary audit clips failed; inspect explicit exclusions')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Human boundary audit incomplete; no accent accuracy inferred')
        raise
