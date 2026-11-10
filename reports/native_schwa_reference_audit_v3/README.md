# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/native_schwa_reference_audit_v3`.
Creation and last review (UTC): 2026-11-07T09:11:16.866807+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Reference-relative vowel-site evidence with source-speaker/audio exclusion and explicit insufficient-coverage output.

## File inventory

- `report.json`: 347 bytes; SHA-256 `e33f71cbebe819606fefef78a7138bd4b5e203aca64880ce0cc2d69743f58afc`.
- `scores.json`: 124613 bytes; SHA-256 `2a812b01273a3bd14dec52f3059910f83650e78b9be6ec4109eb197fc7d45939`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- /Volumes/T9/hindi-detector/reports/native_schwa_reference_v3/reference.json
- /Volumes/T9/hindi-detector/reports/schwa_sites_native_benchmark_v3
- Reference SHA c98965c71b0c6c2eb5a9f07fd40622fb868631c07c61e7f4829f05d5f3f7f372
- Input sites SHA afa55c5d1a2801610ffa0cdfd958ad3256ccb946e4802f2046b9a9e77ec705ba
- python -m scripts.score_schwa_references --reference reports/native_schwa_reference_v3/reference.json --sites SITE_GROUP --out NEW

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
