# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: cdjks-MacBook-Pro-2.local. Original directory: `/Users/cdjk/github/hindi-detector/reports/human_boundary_audit_v6`.
Creation and last review (UTC): 2026-11-07T14:45:34.310245+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Measure fixed-window boundary sensitivity on all232 real human recordings, preserving unknown accent/style judgments.

## File inventory



`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- scripts/audit_human_boundaries.py
- hindi_detector/boundary_audit.py
- data/human_g2p_bank_v5
- reports/human_g2p_evidence_v5

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.

Runtime inventory: source_snapshot.json preserves exact runner, boundary recipes, G2P scorer, CTC and engine source; items.jsonl records every clip and planned-word sensitivity audit or explicit failure. status.json holds atomic counts, pid only this worker, worker.log explicit operations/tracebacks. result/ has a separate immutable inventory for report.json and word_summary.json. Input WAVs and previous feature files are read in place, not copied or changed. No posterior tensors, cropped WAVs or new model weights are saved. Current consumer: remove fragile acoustic comparisons before calibration; unknown human labels remain unknown. These tests quantify crop sensitivity, not accent accuracy. Compact metrics/source retained while this detector task consumes them, next review2026-11-14; logs expire within24h after result review. Upstream probe/alignment and source-corpus licences continue to apply; internal research only. NOT on HF.
