# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/new_text_inference_v35`.
Creation and last review (UTC): 2026-11-09T00:41:30.845693+00:00. Status: active.
Next review: 2026-11-16.

Purpose and current consumer: Unseen-text inference execution with no supplied reference; generates two verified24-layer teacher baselines on L40 and content screens them before local acoustic inference. This is an execution/qualification check, not independent accuracy. launch.json exact PID/command; run.log subprocess output; result/ final word/reference evidence. Review2026-11-10, logs disposable24h after consumer closes.

## File inventory

- `plan.json`: 1098 bytes; SHA-256 `86ccedcb14deefb1bcd687d2abbeda1863519fb416972524a73fce75f42ac0e2`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- python -m scripts.detect_nepali_leakage --query WAV --text TEXT --out NEW

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.

unknown_vowel_deletion/quality.log and *_launch.json: two cold no-reference executions; deletion_result/quality_result contain final outputs. Original run.log failed only because controller mistook fully completed smoke for partial-smoke status; explicit inspected recovery in reports/inference_ref_75036d4376e1fe995f369adc/completed_smoke_recovery. No failed WAVs or rerender. New quality text tests corrected full pipeline without recovery. Codex /root2026-11-09.
