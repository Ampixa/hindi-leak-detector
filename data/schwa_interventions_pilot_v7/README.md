# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: cdjks-MacBook-Pro-2.local. Original directory: `/Users/cdjk/github/hindi-detector/data/schwa_interventions_pilot_v7`.
Creation and last review (UTC): 2026-11-07T15:02:26.591353+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Check deterministic same-voice schwa-count control rendering before designing a supervised realization head. This is auxiliary to the user-selected Pocket/IndicF5 task, not a replacement definition of Hindi leakage.

## File inventory



`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- scripts/probe_kala_interventions.py
- models/kala_controls
- runtime/kala_controls_environment.json
- data/schwa_candidates_v1/candidates.json

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.

Inventory: schwa-<candidate>_<speaker>_<condition>.wav files are the eight expected two-word/two-voice/two-condition outputs, each enumerated by size/SHA in items.jsonl and result/receipts.json. result/ has an immutable source/report inventory. worker.log records rendering or explicit failures. The exact source ID list and observed ONNX duration output remain unmodified receipts. No human acceptability/error labels are inferred. Current consumer: acoustic verification of site interventions and same-voice training-control design; review2026-11-14. No HF upload; generated controls/internal model research only, upstream model CC-BY-SA4.0, native helper code MIT. Delete unused pilot takes/logs within24h after this validation consumer closes; retain compact receipts/source and selected example only if needed by current comparison. No shared model or source recording modified. Original artifacts on k2 T9; cdjk metadata copies only.
