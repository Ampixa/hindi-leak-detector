"""Acquire public pinned Whisper Hindi ASR; no gated model access workflow."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
REPO = 'openai/whisper-large-v3-turbo'
REVISION = '41f01f3fe87f28c78e2fbf8b568835947dd65ed9'
OUT = ROOT / 'models/hindi_asr_whisper_turbo'
LOG = logging.getLogger(__name__)


def main() -> None:
    if OUT.exists():
        raise FileExistsError('Inspect an existing ASR acquisition before retrying')
    info = HfApi(token=False).model_info(REPO, revision=REVISION, files_metadata=True)
    if info.sha != REVISION or info.gated or info.card_data.get('license') != 'mit':
        raise ValueError('Expected inspected public ASR revision and actual MIT licence metadata')
    OUT.mkdir()
    OUT.joinpath('README.provenance.md').write_text(f'''# Hindi ASR acquisition
Created Codex GPT-6 /root on k2 UTC {datetime.now(timezone.utc).isoformat()}.
Owner Ampixa/user. Origin {OUT}. Status active. Last review2026-11-07;
next review2026-11-08. Current consumer overnight paired-TTS content quality filter.
Upstream {REPO}@{REVISION}, MIT. No local uploads to HF.
Inventory: upstream README.md/config/generation_config/preprocessor/tokenizer JSON,
merges.txt/vocab.json, model.safetensors; inventory.json final per-file size/SHA and
publisher hash; .cache/huggingface contains local download metadata/locks only.
Size pending completion. No duplicate weight copy created by this script. Retain
while ASR calibration/filtering consumes it; remove redundant downloads when
superseded, retaining pins/config/results. Reproduce scripts.acquire_hindi_asr.
Acquisition logs live under reports/overnight_v11, expire24h after reviewed completion.
Public model only; gated AI4Bharat model access was unavailable, no bypass attempted.
''')
    files = [f for f in info.siblings if '/' not in f.rfilename and f.rfilename != '.gitattributes']
    records = []
    for member in files:
        if member.rfilename != 'model.safetensors' and not member.rfilename.endswith(('.json', '.txt', '.md')):
            raise ValueError('Unexpected public ASR root file: ' + member.rfilename)
        path = Path(hf_hub_download(REPO, member.rfilename, revision=REVISION, local_dir=OUT, token=False))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lfs = member.lfs
        expected = lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)
        if path.stat().st_size != member.size or expected is not None and digest != expected:
            raise ValueError('ASR download identity differs from pinned publisher metadata')
        records.append({'path': member.rfilename, 'bytes': path.stat().st_size, 'sha256': digest, 'publisher_lfs_sha256': expected})
        LOG.info('Verified %s %s bytes', member.rfilename, path.stat().st_size)
    OUT.joinpath('inventory.json').write_text(json.dumps({'repo': REPO, 'revision': REVISION, 'files': records,
        'license': 'MIT', 'created_utc': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')
    print(json.dumps({'status': 'public_hindi_asr_acquired_not_yet_calibrated', 'files': len(records), 'bytes': sum(r['bytes'] for r in records)}))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Public Hindi ASR acquisition failed; preserve partial state for inspection')
        raise
