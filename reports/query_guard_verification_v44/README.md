# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/query_guard_verification_v44`.
Creation and last review (UTC): 2026-11-09T01:33:12.749302+00:00. Status: active.
Next review: 2026-11-16.

Purpose and current consumer: Final guard/abstention execution verification. Each request result has actual exit0 JSON and exact model/input/ASR/source receipts; *.log subprocess output. report.json checks known flags, unchanged phonetic features, deliberately mismatched transcript rejection and two no-content-qualified-reference abstentions. Original v42 failure retained. Current consumer runnable CLI handoff; review2026-11-10, compact conclusions retained with release, logs expire24h after consumer closes. No waveform/model copies/HF upload.

## File inventory

- `plan.json`: 2422 bytes; SHA-256 `23b868ec303fdb8390db1cd44a543b2c3ce0c794dcceff715009e8b134130df2`.
- `repair.json`: 430 bytes; SHA-256 `485200077466bf9bd09157f4660ec18c56e567605ac54c36ff2820d6dcf8ba12`.
- `source_snapshot.json`: 5813 bytes; SHA-256 `9575eb667477323dfe0561851792c7389a2a9ea9d5929cc26becbb40e8160b3c`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-indicf5/bin/python -m scripts.verify_query_guard

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
