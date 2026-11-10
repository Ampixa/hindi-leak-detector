# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/intervention_acoustic_check_v7`.
Creation and last review (UTC): 2026-11-07T15:05:36.593312+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Test whether same-voice one-site requests produce independently distinguishable acoustic evidence before larger intervention training.

## File inventory

- `report.json`: 750 bytes; SHA-256 `ac99e443bf65bf05d5bf8b9d7e119578e58375193d21bc75c5b3c74837e28fd4`.
- `clip_evidence.json`: 15330 bytes; SHA-256 `70329defedba521c498a7338e56278975f3f6fe3a11089f1fbce8bd77deec7fa`.
- `pairs.json`: 1104 bytes; SHA-256 `d6626e96452bae9042b8ab2ac2f14ddf49fd85df4a12989c4b1239469abd6ee7`.
- `source_snapshot.json`: 24653 bytes; SHA-256 `2faa5c12818787714a50a4aab76d9a990681f2f923d5da288c030a586f70c809`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- Pilot receipts SHA 77a558cd07eca343947f1ae1260862f0ea8205131daacf7d1ce42979fee56cc0
- Pinned phoneme probe inventory
- .venv-indicf5/bin/python -m scripts.check_intervention_phonetics

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
