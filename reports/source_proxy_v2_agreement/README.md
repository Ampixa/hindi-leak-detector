# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: cdjks-MacBook-Pro-2.local. Original directory: `/Users/cdjk/github/hindi-detector/reports/source_proxy_v2_agreement`.
Creation and last review (UTC): 2026-11-07T07:29:20.199093+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Conservative proxy agreement with explicit abstention coverage; exploratory, not validated Hindi detection.

## File inventory

- `report.json`: 2082 bytes; SHA-256 `978d365261c5594f2a30909f445a41f03f0913ed066d8489c1b73ce32e3c4585`.
- `predictions.json`: 24186 bytes; SHA-256 `5fa5e2cee27c6337df9404aa1f9205692e845f231004373e889a5b054956a0a4`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- /Users/cdjk/github/hindi-detector/reports/source_proxy_v2 predictions SHA256 79fae013416a2376c94fdcf739adece3cd719c5a9a4cf8df91768b5ba1d925bd
- probe predictions SHA256 6621d2de03877fa54d3505235fb232f914f0934957ddd18bafed78aff7b99863
- python -m scripts.report_proxy_agreement --revision reports/source_proxy_v2 --out NEW

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
