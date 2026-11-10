# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/quality_followup_summary_v38`.
Creation and last review (UTC): 2026-11-09T01:17:46.743722+00:00. Status: active.
Next review: 2026-11-16.

Purpose and current consumer: Compact frozen quality follow-up summary. report.json keeps earlier/new/pooled controlled counts, exact content screening, ASR-pass-both subset, word-family uncertainty and selection limits; input_inventory.json pins all original evidence. No new labels, source/model/audio copies or tuning. Current consumer morning release review; review2026-11-10. Retain compact decisions while release consumed; raw controls/logs expire24h after consumer closes. NOT on HF; inherited source/probe/MMS research restrictions apply.

## File inventory

- `report.json`: 3707 bytes; SHA-256 `22cb9f1664b5c8d123a3c47500a52576c608248a22b51b598b204ed3882b493a`.
- `input_inventory.json`: 1034 bytes; SHA-256 `9340aad9244f0fffb43a36310ac3127ee4d8451abfed34828dd5a518ceab659e`.
- `source_snapshot.json`: 6884 bytes; SHA-256 `e4862fd62458defdda2f338e0b3d680aa33a9a31fbb3d6a8465005b58c8bdf2d`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-indicf5/bin/python -m scripts.summarize_quality_followup

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
