# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/release_teacher_consensus_v36`.
Creation and last review (UTC): 2026-11-09T00:46:50.099303+00:00. Status: active.
Next review: 2026-11-16.

Purpose and current consumer: Actual recording versus separately voiced teacher reference audit. items.jsonl: every paired site decision/feature114 and qualification; words.json: conservative reference consensus and per-word untested/uncertain states for all3 sources; report.json: source/voice counts, development regressions and flags. status.json live owned PID/phase. No raw frame, waveform or model copies. Human flags are variation/candidate disagreements, never invented native-error FPR. Current consumer deployment reference sensitivity; review2026-11-10, retain compact decisions while release active, redundant comparisons expire24h after consumer closes. NOT on HF; inherited source/model/MMS research restrictions apply.

## File inventory

- `protocol.json`: 1395 bytes; SHA-256 `10df7bc63e03f4458339807b35f84906c139a9275a9a26dc926200bd4c7102f3`.
- `source_snapshot.json`: 21320 bytes; SHA-256 `1dc6d569b39faa0ff7df2e85136fa59a0e2b82de396da869cb5777f9ca2817fc`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-indicf5/bin/python -m scripts.audit_release_teacher_consensus

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
