# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: cdjks-MacBook-Pro-2.local. Original directory: `/Users/cdjk/github/hindi-detector/reports/pocket_switch_v4`.
Creation and last review (UTC): 2026-11-07T11:04:20.375739+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Record user-authorized Pocket switch, concrete CPU continuation and reliability criteria; avoid promising a validation date before tests exist.

## File inventory

- `plan.json`: 1166 bytes; SHA-256 `545d07fb749983db6c188b5115c97a74b9af1eadf56e15088f72d6597b7f391c`.
- `reference_replacement_verification.json`: 3003 bytes; SHA-256 `1f7637e4cd1f3b0a3b33ece67857b38abf4520a725cc2b0224d2d45fa73bfc3c`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- scripts/run_pocket_matrix.py
- scripts/continue_pocket_pipeline.py
- scripts/validate_pocket_native.py
- User selected Pocket Nepali instead of Audio8 on 2026-11-07

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.

Cancelled GPU input cleanup: `cancelled_input_cleanup.json` (31528 bytes, SHA-256 `09a018661fcdddae7539177abca39e23f1e5a71428e85bf24fb7fd819341dc5b`) records deletion of 1,701,244,656 bytes of unused model/reference copies after cancellation, verified HF restore pin and rechecked k2 originals. No unique audio or shared checkpoints were deleted. Active compact retention evidence; next review 2026-11-14.
