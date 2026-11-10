# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: cdjks-MacBook-Pro-2.local. Original directory: `/Users/cdjk/github/hindi-detector/data/schwa_candidates_v1`.
Creation and last review (UTC): 2026-11-06T07:35:31.982602+00:00. Status: active.
Next review: 2026-11-13.

Purpose and current consumer: Native-review queue for schwa contrasts; not training gold.

## File inventory

- `candidates.json`: 235974 bytes; SHA-256 `21b800f37eb29b0c901063ca34dad587ae54abf3b5e0c5ecee8b6bf282945975`.
- `metrics.json`: 136 bytes; SHA-256 `033d28edeb478ceee6c2eb8d8ae1a2633c08cabf5079d107bea4b68a25187c7a`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- /Users/cdjk/github/hindi-detector/hindi_detector/cli.py mine --frontend-root .sources/frontend --hindi-csv .sources/hindi/hindi/data/extra_large.csv --out data/schwa_candidates_v1
- sources.lock.json

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
