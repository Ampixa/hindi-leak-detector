# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/physical_detector_v12`.
Creation and last review (UTC): 2026-11-08T01:49:39.485824+00:00. Status: active.
Next review: 2026-11-15.

Purpose and current consumer: Physical-word follow-up candidate. Generated controls/*.wav are active training/evaluation inputs with hashes/bytes in journals; features/*.npz retains compact paired word inputs; items.jsonl records per-wave source, labels, timing. model.pt selected final checkpoint, history.json train/dev trace, report.json/predictions.json held-out internal outcomes. No human accent gold. Keep selected model/compact metrics; review waves/features within24h after consumer completes and remove if superseded.

## File inventory

- `inputs.json`: 11401 bytes; SHA-256 `e4420a08a2b9817ff77ec156fdd37a22b42bb91d49a9c848f542bc3a2aabf2c6`.
- `source_snapshot.json`: 30080 bytes; SHA-256 `43665dd3e5b93d8faca0479678cbe3602ac8c8ba990abec5665d153c292fbc59`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-indicf5/bin/python -m scripts.train_physical_detector --base reports/paired_detector_pilot_v11 --out NEW
- Fixed existing split; one query conditioning group/text and different native reference; first20 prior physical-test texts excluded. No new model weights downloaded.

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
