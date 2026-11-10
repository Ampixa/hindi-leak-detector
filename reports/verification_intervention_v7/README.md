# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: cdjks-MacBook-Pro-2.local. Original directory: `/Users/cdjk/github/hindi-detector/reports/verification_intervention_v7`.
Creation and last review (UTC): 2026-11-07T15:33:52.028460+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Verify current trained-component results, exact direct inference, sensitivity-aware listening interface and limitations without claiming an accurate Hindi detector.

## File inventory

- `verification.json`: 7977 bytes; SHA-256 `f7506b53f190442113aa0b5c0fcf73ec2879cdcff1ac0bea6258c4c39191d5bb`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- reports/intervention_heads_v7
- reports/intervention_direct_latest_v7
- reports/human_boundary_audit_v6
- Current private Tailscale HTTP/Playwright checks; no label writes

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.

End storage/source/liveness audit: storage_end.json (846 bytes, SHA256 4c092e62db314d203584486d243358bc713e62a098a49d3bc727f2248aa08aea) records actual free space, retained model/runtime sizes, current source parity, and the still-active original-audio worker. Its copies and original recordings remain untouched. Cleanup log records logical removal versus observed concurrent free-space change.
