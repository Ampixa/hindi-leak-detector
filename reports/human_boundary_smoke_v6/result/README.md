# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/human_boundary_smoke_v6/result`.
Creation and last review (UTC): 2026-11-07T14:45:55.188108+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Quantify sensitivity of human dual-G2P comparisons to fixed word-window changes before calibration.

## File inventory

- `report.json`: 1695 bytes; SHA-256 `07ba357cc3f72cffe2cfcf2f83961eb114fbb82297d77467706286c07b962347`.
- `word_summary.json`: 454 bytes; SHA-256 `04036939a5b1aadad31b1b726a1a07780c4d0bdcea1638c0405a99fdbf19f813`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- Parent items.jsonl and source_snapshot.json
- /Volumes/T9/hindi-detector/data/human_g2p_bank_v5/word_bank.json
- /Volumes/T9/hindi-detector/reports/human_g2p_evidence_v5/manifest.json
- python -m scripts.audit_human_boundaries --bank data/human_g2p_bank_v5 --evidence reports/human_g2p_evidence_v5 --out NEW

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
