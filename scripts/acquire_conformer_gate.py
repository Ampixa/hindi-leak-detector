"""Acquire independently published MIT-source IndicConformer CPU exports."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'models/indicconformer_gate'
REPO = 'parismitaglobalsolutions/indicconformer-sherpa-onnx'
REVISION = '9721eb71eea141fae0982cfcdb9dd2e3d4953c4a'
FILES = ['hi/model.int8.onnx', 'ne/model.int8.onnx', 'tokens.txt', 'README.md']
LOG = logging.getLogger(__name__)


def main():
    if OUT.exists():
        raise FileExistsError('Inspect existing conversion acquisition rather than overwrite')
    info = HfApi(token=False).model_info(REPO, revision=REVISION, files_metadata=True)
    if info.sha != REVISION or info.gated or info.card_data.get('license') != 'apache-2.0':
        raise ValueError('Public conversion metadata differs from inspected source')
    OUT.mkdir()
    (OUT / 'README.provenance.md').write_text(f'''# Public IndicConformer ASR exports
Created Codex GPT-6 /root on k2 UTC {datetime.now(timezone.utc).isoformat()}.
Owner Ampixa/user; origin {OUT}; status active. Last review2026-11-08;
next review2026-11-09. Consumer overnight Hindi ASR content-filter calibration.
Upstream {REPO}@{REVISION}; conversion repository Apache-2.0, underlying
AI4Bharat IndicConformer MIT per the export author's provenance. This is a
separately published public conversion; no gated repository files accessed.
Inventory: hi/model.int8.onnx and ne/model.int8.onnx actual per-language ASR graphs;
tokens.txt shared decoder vocabulary; upstream README.md conversion provenance;
inventory.json per-file SHA/size/publisher pins; .cache/huggingface download metadata.
Size pending completion. No other model copies or HF uploads. Reproduce this CLI
and pinned source. Keep while filter calibration/bulk scoring consumes them; compact
results/pins after supersession. Logs reports/overnight_v11 expire24h after review.
''')
    metadata = {f.rfilename: f for f in info.siblings}; records = []
    for name in FILES:
        member = metadata[name]; path = Path(hf_hub_download(REPO, name, revision=REVISION, local_dir=OUT, token=False))
        digest = hashlib.sha256(path.read_bytes()).hexdigest(); lfs = member.lfs
        expected = lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)
        if path.stat().st_size != member.size or expected is not None and digest != expected:
            raise ValueError('Published conversion identity differs')
        records.append({'path': name, 'bytes': path.stat().st_size, 'sha256': digest, 'publisher_lfs_sha256': expected})
    (OUT / 'inventory.json').write_text(json.dumps({'repo': REPO, 'revision': REVISION, 'files': records,
        'conversion_license': 'Apache-2.0', 'source_model_license': 'MIT'}, indent=2) + '\n')
    print(json.dumps({'files': len(records), 'bytes': sum(r['bytes'] for r in records)}))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        LOG.exception('Public IndicConformer conversion acquisition failed')
        raise
