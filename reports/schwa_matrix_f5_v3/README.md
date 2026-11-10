# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/schwa_matrix_f5_v3`.
Creation and last review (UTC): 2026-11-07T09:13:47.604872+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Inspect exact temporal CTC hypotheses at candidate vowel sites and audit them against multi-speaker native readings.

## File inventory

- `report.json`: 1280 bytes; SHA-256 `0ee4a4fe1742cdf0f7859c8b03f0c11c6d99f288b90fbd2aaf0c6db26d455f6f`.
- `word_sites.json`: 1075864 bytes; SHA-256 `d824f743d4c7d9795dd6a6c1626134f20171dd30bd1ad679f0b3fb01f6831c6e`.
- `source_snapshot.json`: 15327 bytes; SHA-256 `0249b62f4d6442ffb4129bfd42571468fd28ead0433ac16b5866ae94f2315b92`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- /Volumes/T9/hindi-detector/.sources/frontend/data/frontend/ipa_map.tsv
- data/schwa_candidates_v1/candidates.json
- python -m scripts.probe_schwa_sites --pack PACK --evidence WORD_EVIDENCE --out NEW

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
