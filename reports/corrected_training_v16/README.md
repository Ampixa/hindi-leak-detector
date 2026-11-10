# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/corrected_training_v16`.
Creation and last review (UTC): 2026-11-08T07:47:37.131566+00:00. Status: active.
Next review: 2026-11-15.

Purpose and current consumer: Current corrected training controller. training_launch.json and training.log own exact process; status.json live phase/counts; pilot/ is incremental development-only model; final/ will contain final selected weights and reserved physical/original-TTS reports. Source locks and per-wave hashes retained in feature group. v15 waiting trainer was explicitly stopped before changing the fitting objective; pilot preserved. No old TTS data or weights used. Keep selected models/compact metrics; review2026-11-09. Logs disposable24h after consumer completes.

## File inventory

- `protocol.json`: 923 bytes; SHA-256 `bd4016e996bba413d735c72154c015d5038d10c913be63769c401acf860ac507`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- User authorized continued training with teacher24layer and original AI4Bharat IndicF5.
- Fresh seed44; exact matched-focal pair objective; test arrays not loaded for pilot.

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.

## Continued training inventory amendment

Owner Ampixa/user; created by Codex (GPT-6), session /root, 2026-11-08 UTC on k2 and cdjk. Last/next review: artifact_review.json UTC /2026-11-09. NOT on HF. Existing source licences apply, including MMS noncommercial research dependency. Known duplicates: selected pilot/preview weights and compact metrics copied to cdjk; bulk audio stays on k2 and the documented teacher source mirror on L40. No extra weight backup was made.

- protocol.json freezes the matched focal/pairwise objective. training_launch.json/training.log record the original pilot/waiter; finisher_launch.json/finisher.log own the current whole-unit CPU recovery and final fit. The prior waiting PID was stopped safely before the transition; no reserved score had been opened.
- status.json is the current finisher phase/counts, not a historical training result. preview_refresh_launch.json/preview_refresh.log own the development-only refresh; pilot/ and preview/ contain model.pt, train-only normalization in the checkpoint, frozen inputs.json, source_snapshot.json, history.json and report.json. Their README and actual model SHA in report.json identify each selection. No epoch checkpoints retained.
- pilot_examples/ and preview_examples/ contain fixed-order development report/predictions/source snapshots for private word playback; no copied audio or reserved examples. Predictions include actual word intervals and source scores. Exact known-source provenance is not Hindi-error gold.
- direct_verification.json and direct inference verification subgroups document cached/direct score consistency and the long-input CPU path; these are diagnostics, not accuracy tests. Logs and redundant direct diagnostics are disposable24h after final verification closes.
- native_variation_launch.json/native_variation.log own the separate real-recorded-speaker audit in ../native_variation_audit_v16. Its feature/source journals cover all189 clips. No geography-derived error labels and no feedback into fitting.
- finalizer_launch.json/finalizer.log/finalizer_status.json own the final review assembler. final_summary.json is written only after final fitting and recorded variation finish with matching model SHA. The native reference_sensitivity_by_occurrence.json uses actual aligned start/end to distinguish repeated words.
- final/ will contain selected final weights/config/history/report and physical/original-TTS reserved predictions, written after all accepted data are ready. data_quality_addendum.json records CPU recovery attempts/results; whole matched units use the same CPU backend.
- artifact_review.json is a measured inventory/storage/source/model-pin audit, with its own provenance. It identifies the superseded old bulk as needs-review rather than claiming all historical WAVs are necessary. No deletion is implied by this report.
- *_launch.json and *.log are task-owned process handles/explicit traces. Preserve source/config/hash receipts and compact decisions; remove redundant logs/controls/features within24h after their consumer closes. Keep selected checkpoints only for current paired inference/reproduction; review2026-11-09, remove superseded pilot/preview weights once final selection/review is verified.

Inputs: exact24-layer teacher and original IndicF5 from data/teacher_indicf5_v14; measured features/physical controls in ../structured_evidence_v15. CPU recovery overlay is documented there; original failed receipts are preserved. Reproduce using exact launch command and stored source/runtime/input hashes. No Git revision existed. See root README for source-module roles and limitations.

Local cdjk copy review 2026-11-08T08:52:33.963350+00:00: local_review.json measures local free space, verifies copied preview model SHA, and records that automatic approval review blocked deletion of two completed browser snapshots. No deletion or reclaimed space. .playwright-mcp/README.md inventories retained diagnostics for cleanup review2026-11-09. Compact models/metrics retain original k2 provenance; local copies made by Codex /root for user inference/review. review_verification.json records browser/audio consistency and live detached worker commands.
