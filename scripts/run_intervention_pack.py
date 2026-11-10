"""Render the frozen same-voice intervention plan with explicit request labels."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path

from hindi_detector.artifacts import write_group
from hindi_detector.kala_render import KalaControlRenderer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/schwa_interventions_v7/render'
LOG = logging.getLogger(__name__)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if not (OUT / 'README.md').is_file() or (OUT / 'items.jsonl').exists():
        raise ValueError('Need a documented fresh render directory; inspect partial jobs before retry')
    plan_path = OUT.parent / 'plan.json'
    plan = json.loads(plan_path.read_text())
    files = [Path(__file__), ROOT / 'hindi_detector/kala_render.py']
    (OUT / 'source_snapshot.json').write_text(json.dumps([{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'source_utf8': p.read_text()} for p in files], indent=2) + '\n')
    renderer = KalaControlRenderer(ROOT)
    rows = []; repeat_verified = set()

    def status(phase: str) -> None:
        path = OUT / 'status.json'; temporary = path.with_suffix('.partial')
        temporary.write_text(json.dumps({'status': phase, 'completed': len(rows), 'planned': len(plan['jobs']),
                                        'updated_utc': datetime.now(timezone.utc).isoformat(), 'plan_sha256': sha(plan_path)}, indent=2) + '\n')
        temporary.replace(path)

    status('rendering_same_voice_schwa_interventions')
    for job in plan['jobs']:
        if renderer.speakers[job['speaker']] != job['speaker_id']:
            raise ValueError('Frozen speaker ID differs from actual model config')
        payload, record = renderer.render(job['requested_phones'], job['speaker'], job['length_scale'])
        key = (job['speaker'], job['length_scale'])
        if key not in repeat_verified:
            repeated, repeated_record = renderer.render(job['requested_phones'], job['speaker'], job['length_scale'])
            if payload != repeated or record != repeated_record:
                raise ValueError('Same-input zero-noise rendering is not deterministic')
            repeat_verified.add(key)
        filename = job['id'] + '.wav'; path = OUT / filename
        if path.exists():
            raise FileExistsError('Refuse to replace retained control audio')
        path.write_bytes(payload)
        row = {**job, **record, 'audio': filename, 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(),
               'status': 'ok_unreviewed_requested_phone_intervention', 'human_pronunciation_gold': False}
        with (OUT / 'items.jsonl').open('a') as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        rows.append(row)
        if len(rows) % 10 == 0:
            status('rendering_same_voice_schwa_interventions')
            LOG.info('Rendered controls %s/%s', len(rows), len(plan['jobs']))
    report = {'status': 'same_voice_interventions_rendered_not_human_error_labels', 'completed': len(rows),
              'plan_sha256': sha(plan_path), 'audio_bytes': sum(r['bytes'] for r in rows),
              'speaker_rate_repeat_checks': len(repeat_verified), 'renderer_duration_step_samples': 256,
              'actual_hindi_accuracy': None, 'actual_pronunciation_gold': False,
              'input_label_meaning': 'Known requested one-site vowel-count change, not independently verified acoustic realization or human acceptability.'}
    write_group(OUT / 'result', {'report.json': report, 'receipts.json': rows},
                'Same-voice one-site controls with lexical and voice partitions frozen before rendering. All audio identities, input IDs and duration correspondence retained.',
                ['Parent plan.json, source_snapshot.json and items.jsonl', '.venv-kala-controls/bin/python -m scripts.run_intervention_pack'])
    status('same_voice_interventions_complete_unreviewed')
    print(json.dumps(report))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Intervention rendering stopped; inspect retained partial outputs before any retry')
        raise
