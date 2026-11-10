"""Generate independent Pocket anchors, extract them, then fit frozen reference heads."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

from hindi_detector.artifacts import write_group

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'reports/pocket_reference_pipeline_v10'
LOG = logging.getLogger(__name__)
OWNS_STATE = False


def save(phase: str) -> None:
    path = STATE / 'status.json'; temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps({'phase': phase, 'pid': os.getpid(), 'updated_utc': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')
    temporary.replace(path)


def command(phase: str, arguments: list[str]) -> None:
    save(phase)
    with (STATE / (phase + '.log')).open('wb') as log:
        subprocess.run(arguments, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                       env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2'})


def main() -> None:
    global OWNS_STATE
    if sys.version_info[:2] != (3, 12) or STATE.exists():
        raise ValueError('Need pinned word runtime and fresh reference pipeline')
    source = Path(__file__)
    write_group(STATE, {'source_snapshot.json': {'path': str(source.relative_to(ROOT)), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'source_utf8': source.read_text()}},
                'Generate63 separate-voice Pocket anchors, extract real acoustic evidence and fit planned source heads. status.json/PID track progress; each phase log holds traceback/progress. Logs disposable within24h of reviewed completion. Preserve actual failure state, never auto-restart.',
                ['data/native_reference_plan_v10 (frozen generation plan)', 'reports/native_reference_plan_v10/plan.json (frozen statistical plan)', '.venv-indicf5/bin/python -m scripts.run_reference_pipeline'])
    OWNS_STATE = True
    pack = ROOT / 'data/native_reference_pocket_v10'
    pack.mkdir()
    (pack / 'README.md').write_text(f'''# Independent Pocket comparison anchors
Created Codex GPT-6 /root on k2, UTC {datetime.now(timezone.utc).isoformat()}.
Owner Ampixa/user; origin {pack}; status active; last review2026-11-07 next review2026-11-08.
Purpose/current consumer: same-text references in separate conditioning groups for
the focal source model, with query/reference groups held out together. NOT Hindi gold.
Inventory: plan.json source/checkpoint/settings/job identities; selection.json final
clip registry; pocket/run.json progress/model/runtime; pocket/items.jsonl per-WAV
size/SHA/timing; pocket/source_snapshot.json exact generator/upstream sources;
pocket/pocket_nepali_v1c.yaml config; pocket/pocket_nepali_v1c_<job_id>.wav63 expected
mono24kPCM16 generated readings. README.md this provenance. No new weights/native
audio copies; NOT on HF; underlying model/data licences apply. Size pending completion.
Reproduce with recorded frozen plan and scripts.run_pocket_matrix --pack
data/native_reference_pocket_v10 --source-pack data/native_reference_plan_v10
--systems pocket_nepali_v1c using pinned Pocket runtime. Keep WAVs while current
training/listening consumes them; compact results/configs/hashes after supersession.
Logs live in reports/pocket_reference_pipeline_v10, expire24h after reviewed completion.
''')
    command('generating_separate_pocket_anchors', ['/Users/k2/madhyam/pocket-tts/.venv/bin/python', '-m', 'scripts.run_pocket_matrix', '--pack', 'data/native_reference_pocket_v10', '--source-pack', 'data/native_reference_plan_v10', '--systems', 'pocket_nepali_v1c'])
    run = json.loads((pack / 'pocket/run.json').read_text())
    if run['status'] != 'complete_unreviewed_audio' or run['completed'] != 63:
        raise RuntimeError('Reference generation did not complete all63 planned anchors')
    out = ROOT / 'reports/pocket_reference_evidence_v10'; out.mkdir()
    (out / 'README.md').write_text(f'''# Independent Pocket reference word evidence
Created Codex GPT-6 /root on k2, UTC {datetime.now(timezone.utc).isoformat()}.
Owner Ampixa/user; origin {out}; status active; last review2026-11-07 next review2026-11-08.
Consumer matched reference source-model training and word/prosody inspection.
Inventory: <clip_id>.json actual word/prosody/G2P evidence; <clip_id>.npz phonetic
and prosody arrays; items.jsonl success/failure and SHA/size; manifest.json final
feature/checkpoint inventory; source_snapshot.json exact extraction source.
No duplicate WAV/model, NOT on HF; source licences apply. Size pending completion.
Reproduce scripts.extract_word_evidence --pack data/native_reference_pocket_v10
--g2p-bank data/human_g2p_bank_v5/word_bank.json --out NEW using pinned word runtime.
Keep while source-model evaluation/listening consumes features; compact results
after supersession; logs in parent pipeline expire24h after reviewed completion.
''')
    command('extracting_reference_words', [sys.executable, '-m', 'scripts.extract_word_evidence', '--pack', 'data/native_reference_pocket_v10', '--g2p-bank', 'data/human_g2p_bank_v5/word_bank.json', '--out', 'reports/pocket_reference_evidence_v10'])
    manifest = json.loads((out / 'manifest.json').read_text())
    if manifest['failures'] or manifest['successes'] != 63:
        raise RuntimeError('Reference extraction has missing/failed clips')
    while True:
        continuation = json.loads((ROOT / 'reports/human_matched_v9/continuation.json').read_text())
        if continuation['phase'] == 'exact_context_word_evidence_complete_unreviewed':
            break
        if continuation['phase'].startswith('failed'):
            raise RuntimeError('Primary exact-context word extraction failed')
        observed = subprocess.run(['ps', '-p', str(continuation['pid']), '-o', 'command='], capture_output=True, text=True)
        if observed.returncode != 0 or 'scripts.continue_human_matched' not in observed.stdout:
            raise RuntimeError('Primary extraction controller is not live and no success is recorded')
        save('verified_wait_for_primary_word_evidence'); time.sleep(15)
    command('fitting_pocket_reference_heads', [sys.executable, '-m', 'scripts.train_matched_reference', '--plan', 'reports/native_reference_plan_v10/plan.json', '--reference-kind', 'pocket', '--reference-evidence', 'reports/pocket_reference_evidence_v10', '--reference-pack', 'data/native_reference_pocket_v10', '--out', 'reports/pocket_reference_heads_v10'])
    save('reference_heads_complete_source_accuracy_only')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Independent-reference pipeline stopped; inspect explicit failure before retry')
        if OWNS_STATE:
            save('failed_inspect_logs_before_retry')
        raise
