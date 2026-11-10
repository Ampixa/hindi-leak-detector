"""Remove only cancelled job inputs after pinned remote and replacement checks."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/home/ampixa/hindi-detector-wordmatrix-v3')
LOG = logging.getLogger(__name__)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    verification = json.load(sys.stdin)
    receipt = ROOT / 'cancelled_input_cleanup.json'
    if receipt.exists():
        raise FileExistsError('Inspect existing cleanup receipt before another run')
    cancellation = json.loads((ROOT / 'pocket_switch.json').read_text())
    if cancellation['status'] != 'stopped_waiting_queue_for_user_requested_pocket_switch':
        raise ValueError('Audio8 queue is not recorded as cancelled')
    queue_pid = int((ROOT / 'queue.pid').read_text())
    alive = subprocess.run(['ps', '-p', str(queue_pid), '-o', 'pid='], capture_output=True, text=True)
    if alive.returncode != 1 or alive.stdout.strip() or alive.stderr.strip():
        raise RuntimeError('Cancelled queue PID still exists or process check failed')
    if list((ROOT / 'outputs').glob('*.wav')):
        raise ValueError('Unexpected generated Audio8 outputs need separate consumer review')
    restoration = json.loads((ROOT / 'models/restoration.json').read_text())
    hf = verification['hf_metadata']
    if hf['sha'] != restoration['source_revision'] or hf['id'] != restoration['source_repo']:
        raise ValueError('Restorable HF source identity mismatch')
    siblings = {r['rfilename']: r for r in hf['siblings']}
    model_root, reference_root = ROOT / 'models/audio8', ROOT / 'references'
    expected_models = set()
    for row in restoration['files']:
        path = model_root / row['name']
        if path.parent != model_root or sha(path) != row['sha256'] or path.stat().st_size != row['bytes']:
            raise ValueError('Restored model input changed')
        remote = siblings[row['name']]
        if remote['size'] != row['bytes']:
            raise ValueError('Pinned HF file size changed')
        if 'lfs' in remote:
            if remote['lfs']['sha256'] != row['sha256']:
                raise ValueError('Pinned remote model checksum mismatch')
        else:
            body = path.read_bytes()
            blob = hashlib.sha1(b'blob ' + str(len(body)).encode() + b'\0' + body).hexdigest()
            if blob != remote['blobId']:
                raise ValueError('Pinned remote Git blob checksum mismatch')
        expected_models.add(path)
    model_members = {p for p in model_root.rglob('*') if p.is_file()}
    if any(p not in expected_models and not p.is_relative_to(model_root / '.cache/huggingface') for p in model_members):
        raise ValueError('Uninventoried model directory member')
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    replacements = {r['source_path']: r['sha256'] for r in verification['references']}
    expected_references = set()
    for row in manifest['references']:
        path = reference_root / row['name']
        if path.parent != reference_root or replacements.get(row['source_path']) != row['sha256'] or sha(path) != row['sha256']:
            raise ValueError('Original k2 reference replacement or staged copy mismatch')
        expected_references.add(path)
    if {p for p in reference_root.rglob('*') if p.is_file()} != expected_references:
        raise ValueError('Uninventoried reference member')
    for directory in (model_root, reference_root):
        if directory.is_symlink() or any(p.is_symlink() for p in directory.rglob('*')):
            raise ValueError('Refuse symlinked cleanup inputs')
        # These inaccessible virtual mounts are outside both explicit /home input
        # directories. Exclude only those mounts, keeping target-file checks strict.
        check = subprocess.run(['/usr/bin/lsof', '-e', '/run/docker/netns/default',
                                '-e', '/sys/kernel/debug/tracing', '+D', str(directory)],
                               capture_output=True, text=True)
        if check.returncode != 1 or check.stdout.strip() or check.stderr.strip():
            raise RuntimeError('Open files or unsuccessful lsof check; defer cleanup')
    members = sorted(model_members | expected_references)
    rows = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p),
             'reason': 'Unused task-owned Audio8 input after user switched to Pocket; verified pinned HF restore or original k2 reference exists.'} for p in members]
    record = {'creator': 'Codex GPT-6 /root', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'verified_deletion_planned', 'owner': 'Ampixa/user',
              'source_sha256': sha(Path(__file__)), 'files': rows, 'deleted': [],
              'free_bytes_before': shutil.disk_usage(ROOT).free,
              'lsof_excluded_unrelated_virtual_mounts': ['/run/docker/netns/default', '/sys/kernel/debug/tracing'],
              'hf_revision': hf['sha'], 'reference_replacements': verification['references'],
              'no_unique_audio_deleted': True, 'shared_checkpoints_untouched': True}

    def save() -> None:
        temporary = receipt.with_suffix('.partial')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(receipt)

    save()
    for path in members:
        path.unlink()
        record['deleted'].append(str(path))
        save()
    for directory in (model_root, reference_root):
        for child in sorted((p for p in directory.rglob('*') if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            child.rmdir()
        directory.rmdir()
    record.update(status='verified_unused_inputs_deleted', bytes_removed=sum(r['bytes'] for r in rows),
                  free_bytes_after=shutil.disk_usage(ROOT).free,
                  historical_state='Task-owned restored model/reference copies no longer exist locally; all 14 model files are restorable from the verified HF pin, and original reference bytes remain on k2. No Audio8 matrix output ever existed.')
    save()
    with (ROOT / 'README.md').open('a') as handle:
        handle.write('\nCancelled-input cleanup: cancelled_input_cleanup.json inventories every deleted input/cache path, size and SHA, replacement checks and measured free space. models/audio8 and references were removed after cancellation and no-open-file checks. No generated outputs existed. Compact records remain active reference until superseded; next review 2026-11-14. No new HF upload.\n')
    with (ROOT / 'models/README.md').open('a') as handle:
        handle.write('\n2026-11-07: all restored model bytes and task-owned HF download metadata deleted because Pocket replaced this job. restoration.json and ../cancelled_input_cleanup.json retain pinned file identities and verified remote availability. Restore with restore_audio8_runtime.py and the existing pinned runtime; credentials via stdin only. No shared R3 weights changed.\n')
    print(json.dumps({k: record[k] for k in ('status', 'bytes_removed', 'free_bytes_before', 'free_bytes_after')}))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Cancelled-input cleanup stopped; inspect the receipt before retry')
        raise
