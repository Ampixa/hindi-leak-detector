# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/corrected_pair_v14`.
Creation and last review (UTC): 2026-11-08T05:13:28.334533+00:00. Status: active.
Next review: 2026-11-15.

Purpose and current consumer: Corrected user-requested synthesis pair controller. *_launch.json record exact owned PIDs/commands; *.log contain explicit task progress/errors; status.json and report.json are actual measured state. No old checkpoint relabeling, no publication. Logs expire24h after reviewed completion.

## File inventory

- `request.json`: 236 bytes; SHA-256 `b254ad448d7bba7339c13aecc4078bde1d8e69ff5a90d47c0fc3fed7568831d3`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- Model paths and hashes verified in checkpoint_selection_audit_v13 and actual IndicF5 inventory.

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.

Current inventory amendment 2026-11-08T05:27:30.568368+00:00: build_launch.json/build_plan.log record plan construction; indicf5_smoke_launch.json/indicf5_smoke.log and teacher_smoke_launch.json identify exact smoke jobs; controller_launch.json/controller.log and controller_source_snapshot.json retain orchestration; *_bulk_launch.json and indicf5_bulk.log cover exact bulk handles, remote teacher_bulk.log and launch receipt remain under /home/ampixa/hindi-detector-teacher-f5-v14. asr_runtime.json pins recognizers; content_items.jsonl stores actual transcripts/decisions/SHA; smoke_report.json records actual8+8 content passes. status.json is live counts; final report.json appears only after completion. storage_review.json and source_inventory.json record artifact size/free-space review and current code hashes. current_state.json is a dated source-switch summary, not final accuracy. Review by2026-11-09; generated audio current consumers are matched comparison/quality screening and later localized training. Remaining old unscored v11 bulk is superseded and needs a usage audit before deletion; do not treat all old recordings as required. No shared model/unique data deleted during switch.

Local metadata sync at 2026-11-08T05:29:02.640082+00:00: this directory contains selected k2 reports only. Full generated audio, model runtimes and live content journals remain on k2/L40. ui_verification.json records actual desktop/mobile layout, streamed audio and source-hash checks, plus own transient snapshot cleanup. Retain compact QA for this corrected comparison; review2026-11-09.
