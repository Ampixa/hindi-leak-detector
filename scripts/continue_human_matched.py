"""Extract exact-context evidence; wait only on the verified owned F5 worker."""
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

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'reports/human_matched_v9'
LOG = logging.getLogger(__name__)
OWNS_STATE = False


def save(phase: str, **fields) -> None:
    path = STATE / 'continuation.json'; temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps({'phase': phase, 'pid': os.getpid(), 'updated_utc': datetime.now(timezone.utc).isoformat(), **fields}, indent=2) + '\n')
    temporary.replace(path)


def extract(pack_name: str, output_name: str) -> None:
    pack = ROOT / pack_name; out = ROOT / output_name
    if out.exists():
        raise FileExistsError('Inspect existing exact-context word evidence before continuing')
    out.mkdir()
    out.joinpath('README.md').write_text(f'''# Exact-context word evidence

Created by Codex GPT-6 /root on k2, UTC {datetime.now(timezone.utc).isoformat()}.
Owner Ampixa/user. Origin {out}. Status active. Last review2026-11-07;
next review2026-11-08. Consumer actual human/F5/Pocket pronunciation and prosody
comparison. No actual Hindi/error/native labels. NOT on HF; source licences apply.

Inventory: <clip_id>.json word times, G2P evidence, pitch/energy tracks and source
identity; <clip_id>.npz compact phone/prosody feature arrays; items.jsonl per-clip
success/failure and SHA/size; manifest.json exact inventory/checkpoint identities;
source_snapshot.json exact extractor/dependencies. No extra WAV or model copies.
Size pending completion. Logs are in reports/human_matched_v9. Data parent {pack}.
Reproduce with pinned word runtime: python -m scripts.extract_word_evidence
--pack {pack_name} --g2p-bank data/human_g2p_bank_v5/word_bank.json --out NEW.
Retain while matched analysis consumes features; keep compact conclusions/configs
when superseded, expire logs within24h after reviewed completion. Do not delete
active partial outputs or impute failed observations.
''')
    arguments = [sys.executable, '-m', 'scripts.extract_word_evidence', '--pack', pack_name,
                 '--g2p-bank', 'data/human_g2p_bank_v5/word_bank.json', '--out', output_name]
    with (STATE / (out.name + '.log')).open('wb') as log:
        subprocess.run(arguments, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                       env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2'})
    manifest = json.loads((out / 'manifest.json').read_text())
    planned = json.loads((ROOT / 'data/human_matched_v9/report.json').read_text())['jobs_per_tts']
    if manifest['failures'] or manifest['successes'] != planned:
        raise RuntimeError('Exact-context extraction has missing/failed clips; inspect explicit failures')


def main() -> None:
    global OWNS_STATE
    if sys.version_info[:2] != (3, 12) or not (STATE / 'README.md').is_file() or (STATE / 'continuation.json').exists():
        raise ValueError('Use pinned word runtime and a fresh documented continuation')
    OWNS_STATE = True
    source = Path(__file__)
    (STATE / 'continuation_source.json').write_text(json.dumps({'path': str(source.relative_to(ROOT)), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'source_utf8': source.read_text()}, indent=2) + '\n')
    pocket_run = json.loads((ROOT / 'data/human_matched_pocket_v9/pocket/run.json').read_text())
    planned = json.loads((ROOT / 'data/human_matched_v9/report.json').read_text())['jobs_per_tts']
    if pocket_run['status'] != 'complete_unreviewed_audio' or pocket_run['completed'] != planned:
        raise ValueError('Pocket synthesis must be demonstrably complete before extraction')
    save('extracting_exact_context_pocket_words', expected_clips=planned)
    extract('data/human_matched_pocket_v9', 'reports/human_matched_pocket_evidence_v9')
    switch = json.loads((STATE / 'f5_device_switch.json').read_text())
    pid = switch['new_mps_pid']
    while True:
        run = json.loads((ROOT / 'data/human_matched_f5_v9/indicf5/run.json').read_text())
        if run['status'] == 'complete_unreviewed_audio' and run['completed'] == planned:
            break
        observed = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True)
        if observed.returncode != 0 or 'scripts.run_indicf5_comparison --pack data/human_matched_f5_v9 --device mps' not in observed.stdout:
            raise RuntimeError('Owned F5 worker is no longer live and synthesis is not complete; inspect actual receipts, no automatic restart')
        save('verified_wait_for_owned_f5_generation', verified_live_f5_pid=pid, expected_clips=planned)
        time.sleep(15)
    save('extracting_exact_context_f5_words', expected_clips=planned)
    extract('data/human_matched_f5_v9', 'reports/human_matched_f5_evidence_v9')
    save('exact_context_word_evidence_complete_unreviewed', clips_per_tts=planned,
         actual_hindi_accuracy=None, actual_prosody_accuracy=None)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Matched human-context continuation failed; no fabricated labels or silent retries')
        if OWNS_STATE:
            save('failed_inspect_actual_logs_before_retry')
        raise
