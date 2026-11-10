# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: cdjks-MacBook-Pro-2.local. Original directory: `/Users/cdjk/github/hindi-detector/reports/verification_human_v5`.
Creation and last review (UTC): 2026-11-07T14:38:32.641041+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Verify implementation, served real human word cases and current provenance/data-quality status without calling source-role scores Hindi-error accuracy.

## File inventory

- `verification.json`: 57873 bytes; SHA-256 `2b07b629962cdf68733e2f3d6418b56df6fc55bf0fd5d75378096117fba9ced0`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- 65 pure tests on cdjk and k2
- Current private API at http://100.80.123.8:8770
- Read-only Playwright desktop/mobile/audio checks
- Reports retain exact source and dataset identities; no annotation writes

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.

End-of-turn storage/liveness audit: storage_end.json (1071 bytes, SHA-256 9fbe8ce75579c044ad7c9fa3e02f3e8029d83290a09a762510d8c96ef484b27a) records actual free space, live original-audio worker and named retained model consumers. Active original-source recovery is not deleted.
