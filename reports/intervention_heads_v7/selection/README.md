# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/intervention_heads_v7/selection`.
Creation and last review (UTC): 2026-11-07T15:19:37.190822+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Freeze intervention component weights/settings using only training and development features.

## File inventory

- `models.json`: 68606 bytes; SHA-256 `254aecc9ace68ac3ae7e25b8fdb313beb0454a0f4b0e219e829ddf968fa57c45`.
- `selection.json`: 1221 bytes; SHA-256 `5d5f62f05784529a5c321cda9de529783586790d89cbf46ef0e0cacf403ccca5`.
- `source_snapshot.json`: 23917 bytes; SHA-256 `61414ac474aa2e04080cbb40d5ea7663e044e5aff2e6184b1bd86cdeb8d0b547`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- /Volumes/T9/hindi-detector/reports/intervention_features_v7/manifest.json
- .venv-indicf5/bin/python -m scripts.train_intervention_heads

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
