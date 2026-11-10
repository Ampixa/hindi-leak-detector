"""Extract phone/site evidence for the frozen renderer intervention experiment."""
from __future__ import annotations

from datetime import datetime, timezone
import csv
import hashlib
import json
import logging
from pathlib import Path
import traceback

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from hindi_detector.interventions import intervention_features, require_disjoint, require_partition
from hindi_detector.schwa_probe import compare_site, site_paths
from hindi_detector.word_evidence import WordEvidenceEngine

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/intervention_features_v7'
LOG = logging.getLogger(__name__)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if not (OUT / 'README.md').is_file() or (OUT / 'items.jsonl').exists():
        raise ValueError('Need a documented fresh intervention feature group')
    pack = ROOT / 'data/schwa_interventions_v7'
    plan = json.loads((pack / 'plan.json').read_text())
    report = json.loads((pack / 'render/result/report.json').read_text())
    rows = json.loads((pack / 'render/result/receipts.json').read_text())
    if report['plan_sha256'] != sha(pack / 'plan.json') or report['completed'] != len(plan['jobs']) or {r['id'] for r in rows} != {r['id'] for r in plan['jobs']}:
        raise ValueError('Frozen plan/render coverage mismatch')
    planned = {r['id']: r for r in plan['jobs']}
    for row in rows:
        if any(row[k] != value for k, value in planned[row['id']].items()):
            raise ValueError('Renderer receipt changed planned request metadata')
    partitions = {s: [r for r in rows if r['partition'] == s] for s in ('train', 'development', 'reserved_evaluation')}
    for name, records in partitions.items():
        require_partition(records, name)
    require_disjoint(partitions['train'], partitions['development'])
    require_disjoint(partitions['train'] + partitions['development'], partitions['reserved_evaluation'])
    candidates = {r['candidate_id']: r for r in plan['candidates']}
    with (ROOT / '.sources/frontend/data/frontend/ipa_map.tsv').open() as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        if reader.fieldnames != ['phone', 'ipa_candidate', 'status', 'notes']:
            raise ValueError('Explicit IPA map changed')
        ipa = {r['phone']: r['ipa_candidate'] for r in reader}
    files = [Path(__file__), ROOT / 'hindi_detector/interventions.py', ROOT / 'hindi_detector/schwa_probe.py',
             ROOT / 'hindi_detector/ctc.py', ROOT / 'hindi_detector/word_evidence.py']
    (OUT / 'source_snapshot.json').write_text(json.dumps([{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()} for p in files], indent=2) + '\n')
    engine = WordEvidenceEngine(ROOT)
    results = []; dimensions = {}
    for row in rows:
        metadata = {k: row[k] for k in ('id', 'word', 'candidate_id', 'site_kind', 'word_split', 'voice_split', 'partition', 'speaker', 'speaker_id', 'length_scale', 'condition', 'intervention_class_id', 'label_source', 'actual_hindi_leakage_label', 'human_pronunciation_gold', 'sha256')}
        try:
            path = (pack / 'render' / row['audio']).resolve(strict=True)
            if not path.is_relative_to((pack / 'render').resolve()) or sha(path) != row['sha256']:
                raise ValueError('Control audio checksum/path mismatch')
            wave, rate = sf.read(path, dtype='float32', always_2d=True)
            if wave.shape[1] != 1 or not np.isfinite(wave).all():
                raise ValueError('Need finite mono control audio')
            divisor = int(np.gcd(rate, 16000))
            samples = resample_poly(wave[:, 0], 16000 // divisor, rate // divisor).astype('float32')
            with engine.torch.inference_mode():
                logits = engine.phone(**engine.extractor(samples, sampling_rate=16000, return_tensors='pt')).logits[0]
                log_probs = logits.log_softmax(-1).cpu().numpy()
            candidate = candidates[row['candidate_id']]
            paths = site_paths(candidate, ipa, engine.vocab)
            if len(paths) != 1:
                raise ValueError('Expected one-site intervention')
            evidence = compare_site(log_probs, paths[0])
            vectors = intervention_features(log_probs, evidence, candidate, engine.vocab, engine.special)
            current_dimensions = {k: len(v) for k, v in vectors.items()}
            if dimensions and dimensions != current_dimensions:
                raise ValueError('Intervention feature dimensions changed')
            dimensions = current_dimensions
            destination = OUT / (row['id'] + '.npz')
            if destination.exists():
                raise FileExistsError('Refuse to replace a retained feature cache')
            np.savez_compressed(destination, **vectors,
                                raw_ctc_alternative_minus_lexicon=np.array(evidence['marginal_alternative_minus_lexicon']),
                                parent_audio_sha256=np.array(row['sha256']))
            metadata.update(status='ok_intervention_evidence_not_human_gold', features=destination.name,
                            feature_bytes=destination.stat().st_size, feature_sha256=sha(destination))
        except Exception:
            metadata.update(status='failed_explicitly_excluded', error=traceback.format_exc())
            LOG.exception('Intervention extraction failed for %s; no imputed phonetic evidence', row['id'])
        results.append(metadata)
        with (OUT / 'items.jsonl').open('a') as handle:
            handle.write(json.dumps(metadata, ensure_ascii=False, allow_nan=False) + '\n')
        if len(results) % 25 == 0:
            status = {'status': 'extracting_intervention_phone_evidence', 'processed': len(results), 'planned': len(rows),
                      'failures': sum(r['status'] != 'ok_intervention_evidence_not_human_gold' for r in results),
                      'updated_utc': datetime.now(timezone.utc).isoformat()}
            temporary = OUT / 'status.partial'; temporary.write_text(json.dumps(status, indent=2) + '\n'); temporary.replace(OUT / 'status.json')
            LOG.info('Intervention evidence %s/%s (%s failures)', len(results), len(rows), status['failures'])
    manifest = {'creator': 'Codex GPT-6 /root', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'status': 'intervention_phone_evidence_complete_not_human_error_gold', 'items': results,
                'successes': sum(r['status'] == 'ok_intervention_evidence_not_human_gold' for r in results),
                'failures': sum(r['status'] != 'ok_intervention_evidence_not_human_gold' for r in results),
                'feature_dimensions': dimensions, 'phone_encoder_revision': engine.phone_inventory['revision'],
                'plan_sha256': sha(pack / 'plan.json'), 'render_receipts_sha256': sha(pack / 'render/result/receipts.json'),
                'label_source': 'requested_single_schwa_count_intervention_not_human_error_gold',
                'excluded_inputs': ['source voice ID', 'requested condition/class', 'realized waveform duration', 'decoder duration/timing outputs', 'pitch', 'hidden speech embeddings']}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'status.json').write_text(json.dumps({k: manifest[k] for k in ('status', 'successes', 'failures')}, indent=2) + '\n')
    print(json.dumps({k: manifest[k] for k in ('status', 'successes', 'failures', 'feature_dimensions')}))
    if manifest['failures']:
        raise RuntimeError('Failed intervention extraction requires inspection before any fitting')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Intervention feature extraction incomplete; do not fit partial biased data')
        raise
