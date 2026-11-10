# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: cdjks-MacBook-Pro-2.local. Original directory: `/Users/cdjk/github/hindi-detector/reports/checkpoint_selection_audit_v13`.
Creation and last review (UTC): 2026-11-08T05:07:03.000360+00:00. Status: active.
Next review: 2026-11-15.

Purpose and current consumer: Correct model-selection provenance and interpretation of v11/v12 results. Current consumer: user investigation of why old k2 checkpoint was used; readable correction added to private overnight report. Compact reference evidence; no model/audio copy, no remote mutation of training workloads.

## File inventory

- `report.json`: 1756 bytes; SHA-256 `a535a9a8fbe27dd1add9c8618d0e66e64c47007b6c1f1296040fae658fb467a3`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- Read actual files over SSH on k2 and khalti-gpu; recompute large-export SHA-256 streaming; inspect existing pilot generator code.

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.

local_review.json records measured local artifact sizes/hashes and free space. HTML correction deployed to existing private readonly report; training workloads untouched.
