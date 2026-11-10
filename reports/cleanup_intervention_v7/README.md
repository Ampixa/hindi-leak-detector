# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/cleanup_intervention_v7`.
Creation and last review (UTC): 2026-11-07T15:31:54.929846+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Remove verified unused setup cache and completed success logs without disrupting active original-audio recovery.

## File inventory

- `cleanup.json`: 1556135 bytes; SHA-256 `34178daf893ba4c3eca45f45daafbe7e19e0011571680beca92e673bf23edfec`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- runtime/kala_controls.freeze
- reports/intervention_heads_v7
- All target lsof checks empty; no cache symlink dependency from retained environment

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.

Authoritative final cleanup.json after deletion: 1556228 bytes, SHA256 2f33bfd3c8a40f05677413b12b4738a8b493ee6849d375b8438b123c38cf0d8a. Earlier inventory checksum describes the predeletion plan. Free-space delta is observed during other workloads, not an isolated reclaimed-space measurement.
