# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/data/schwa_interventions_pilot_v7/result`.
Creation and last review (UTC): 2026-11-07T15:02:31.346813+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Inspect whether a pinned same-voice renderer supports reproducible schwa-count interventions and exposes timing.

## File inventory

- `report.json`: 887 bytes; SHA-256 `3fa631f18f3cfa9bc121524355e9cb329d27a5a8e5f7480a306bab53ddbf5616`.
- `receipts.json`: 11205 bytes; SHA-256 `77a558cd07eca343947f1ae1260862f0ea8205131daacf7d1ce42979fee56cc0`.
- `source_snapshot.json`: 22905 bytes; SHA-256 `0af90f79e31a1e907ccf02085ace6356b0dfb3bad3ac7cfc8e1722c4be8cae8f`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- Parent items.jsonl lists every WAV by size/SHA
- Model HF revision45b7d3052a0ad01d5741213ff041e65db0829f5a
- .venv-kala-controls/bin/python -m scripts.probe_kala_interventions

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
