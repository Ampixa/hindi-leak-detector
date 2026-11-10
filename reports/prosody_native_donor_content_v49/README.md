# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/prosody_native_donor_content_v49`.
Creation and last review (UTC): 2026-11-09T02:36:53.893543+00:00. Status: active.
Next review: 2026-11-16.

Purpose and current consumer: New-text Native reference lexical-content screening. Only content pass is accepted; quarantine/reject are retained and excluded. Exact hypotheses, WER/CER, model identities and source SHA retained. Content pass is never a Hindi/accent label. Current consumer generic reference inference; compact receipts retained with active reference cache.

## File inventory

- `result.json`: 299458 bytes; SHA-256 `8bc1165ef43942e2c072c7ab5cbc6eb8533ff8c0b5f73a08fe82465b6426d145`.
- `source_snapshot.json`: 7808 bytes; SHA-256 `0d5fb22c2431a62e963949139cb3dbb700f32485b1341bd0ded61c3dd9e437ef`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- /Volumes/T9/hindi-detector/data/prosody_transfer_v49
- .venv-asr-gate/bin/python -m scripts.screen_teacher_reference --plan PLAN --out NEW

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. Generated waveforms, if present, are explicitly described above and listed in inventory_v57.json.
