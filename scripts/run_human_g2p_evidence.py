"""Pinned-runtime continuation for human word evidence and its neutral summary."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/human_g2p_evidence_v5'
LOG = logging.getLogger(__name__)


def save(value):
    value.update(updated_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid())
    p = OUT / 'job.json'; temporary = p.with_name(p.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n'); temporary.replace(p)


def main():
    if not (OUT / 'README.md').is_file() or (OUT / 'manifest.json').exists() or (OUT / 'items.jsonl').exists():
        raise ValueError('Need documented fresh evidence; inspect partial outputs before resuming')
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Use the pinned Python3.12 word runtime for this continuation')
    phases = [
        ('extracting_real_human_word_evidence', 'extract.log', ['-m', 'scripts.extract_word_evidence', '--pack', 'data/human_g2p_bank_v5', '--g2p-bank', 'data/human_g2p_bank_v5/word_bank.json', '--out', 'reports/human_g2p_evidence_v5']),
        ('summarizing_real_human_g2p_paths', 'summary.log', ['-m', 'scripts.summarize_human_g2p', '--bank', 'data/human_g2p_bank_v5', '--evidence', 'reports/human_g2p_evidence_v5', '--out', 'reports/human_g2p_summary_v5'])]
    base = {'creator': 'Codex GPT-6 /root', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'controller_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'controller_source_utf8': Path(__file__).read_text(), 'python': sys.version,
            'phases': phases, 'source_bank_sha256': hashlib.sha256((ROOT / 'data/human_g2p_bank_v5/word_bank.json').read_bytes()).hexdigest()}
    for phase, filename, arguments in phases:
        save(base | {'status': phase})
        with (OUT / filename).open('w') as log:
            subprocess.run([sys.executable, *arguments], cwd=ROOT,
                           env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2'},
                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
    save(base | {'status': 'evidence_and_summary_complete_unlabeled'})


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Real-human G2P evidence failed; inspect partial outputs before retry')
        if OUT.is_dir():
            save({'status': 'failed_inspect_logs_before_retry'})
        raise
