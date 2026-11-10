# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/human_g2p_summary_v5`.
Creation and last review (UTC): 2026-11-07T12:11:39.427472+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Ground competing G2P hypotheses in selected real human word recordings; do not equate Hindi-path preference with an error.

## File inventory

- `report.json`: 1166 bytes; SHA-256 `7599ad1f9f439464cf4033522fa7c1e42619758bf8cf9daec4dd3cc286ed34ab`.
- `word_summary.json`: 19056 bytes; SHA-256 `e883f3896949e0aa33d33a19cb91c9aadabae7592eaebdb85059b0740a5db75b`.
- `observations.json`: 440046 bytes; SHA-256 `e7c8861f797917e5bd282a8954199059125ec2ae129760f955a84165cf8ee25c`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- /Volumes/T9/hindi-detector/data/human_g2p_bank_v5
- /Volumes/T9/hindi-detector/reports/human_g2p_evidence_v5
- Evidence SHA 6dc0a55b1a04baafc05900d0888d59e451c7a3d663e00561502e19e6221fc792
- Bank SHA 056edc34c0a1788b057c983d1697b35f5504970fd9463e03dead3b1712087e8d
- Driver SHA 915aa67ba77df781fca95997f012c6415fb72569ca613566ead5255f0c4ba47f
- python -m scripts.summarize_human_g2p --bank data/human_g2p_bank_v5 --evidence reports/human_g2p_evidence_v5 --out NEW

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
