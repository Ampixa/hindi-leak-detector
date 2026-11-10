# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: cdjks-MacBook-Pro-2.local. Original directory: `/Users/cdjk/github/hindi-detector/data/schwa_interventions_v7/render`.
Creation and last review (UTC): 2026-11-07T15:11:32.260278+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Render960 planned same-voice one-site controls for a schwa-realization component; intervention request classes are not actual Hindi/native error labels.

## File inventory



`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- scripts/run_intervention_pack.py
- hindi_detector/kala_render.py
- Parent plan.json on k2
- models/kala_controls inventory and runtime/kala_controls.freeze

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.

Runtime inventory: cf7_<SHA-prefix>.wav files correspond one-to-one with the parent plan jobs; items.jsonl and result/receipts.json enumerate every file by size/SHA, explicit requested phone IDs, source-derived class, voice/rate/lexeme partitions and observed duration outputs. source_snapshot.json preserves runner/helper source. status.json atomic progress, pid identifies this worker, worker.log explicit operations/failures. result/ has its own immutable report/receipts inventory. Keep audio only while current realization-head training/evaluation/listening consumes it; review2026-11-14; delete unused takes within24h after consumer closes and keep compact selected metrics/configs/source. Source model CC-BY-SA4.0/native helper MIT, internal research, not uploaded to HF. No original human WAV or other project checkpoint changed. Large audio only on k2 T9; cdjk compact metadata. The overall Hindi detector goal remains unachieved until actual leakage judgments and held-out checks support it.
