# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/schwa_matrix_reference_scores_v3`.
Creation and last review (UTC): 2026-11-07T09:13:48.009640+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Reference-relative vowel-site evidence with source-speaker/audio exclusion and explicit insufficient-coverage output.

## File inventory

- `report.json`: 348 bytes; SHA-256 `d811f39a8b2bffa598e56a20308e2bc8efc95b00eeed5787969db91e1aafe751`.
- `scores.json`: 163738 bytes; SHA-256 `8bc1f6ddb5a0561e5a1772faa5d2ee8a4608897c7d5c1e30a35f3fb0bdf4758c`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- /Volumes/T9/hindi-detector/reports/native_schwa_reference_v3/reference.json
- /Volumes/T9/hindi-detector/reports/schwa_matrix_f5_v3
- Reference SHA c98965c71b0c6c2eb5a9f07fd40622fb868631c07c61e7f4829f05d5f3f7f372
- Input sites SHA d824f743d4c7d9795dd6a6c1626134f20171dd30bd1ad679f0b3fb01f6831c6e
- python -m scripts.score_schwa_references --reference reports/native_schwa_reference_v3/reference.json --sites SITE_GROUP --out NEW

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
