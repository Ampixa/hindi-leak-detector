# IndicF5 matched synthesis

Created by Codex GPT-6, session /root, 2026-11-06 UTC on k2 at
`/Volumes/T9/hindi-detector/data/indicf5_audio8_v1/indicf5`. Owner Ampixa/user.
Status active. Next review 2026-11-13. Consumer: blind review and reserved
evaluation of Hindi pronunciation leakage. No automatic leakage labels.

Inventory: `run.json` holds model pin, weight/config/driver hashes, device and
sampling settings; `items.jsonl` holds one completed output receipt with size,
duration and SHA-256 per job; `sent_*.wav` is the corresponding 24 kHz mono PCM
generated audio. Expected 24 examples, exact sentence IDs from ../selection.json.
`driver_snapshot.py` is the exact SHA-verified source used for this completed
evaluation synthesis. Restore it to scripts/run_indicf5_comparison.py to reproduce
that run; do not execute the archive in place because its root is script-relative.
The live driver was subsequently parameterized for a separate development pack.
A run is complete only when run.json says complete_unreviewed_audio and all
receipts and waveform hashes verify. `README.md` documents the group itself.

Input model and bundled vocoder are documented in ../../../models/README.md.
References are the exact native voice clips and transcripts from the paired
audio8 benchmark, checked by SHA-256. Upstream weights are strictly loaded; only
compiled state-key prefixes are removed by the adapter. Code: scripts/
run_indicf5_comparison.py. The upstream HF wrapper is not executed because its
initialization performs mutable downloads and compilation. Input sentence-list
quality remains subject to review. Generated silence is not stripped, and takes
are not ASR-reranked. These settings preserve prosody evidence.

Reproduce on k2 with the isolated .venv-indicf5 Python and --device mps; --limit
allows a first-item smoke test then exact-config resume. Private internal audio
only, no HF/public upload. Original model licence MIT; corpus voice rights and
source benchmark restrictions still apply. Delete superseded audio within 24h
of consumer completion; retain compact receipts and reviewed labels. No cdjk
audio duplicate is required for the review server.
