# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/anchored_test_v74`.
Creation and last review (UTC): 2026-11-09T08:22:31.871040+00:00. Status: active.
Next review: 2026-11-16.

Purpose and current consumer: Single locked test of local vowel omission after reviewed training and external model/threshold freeze. report.json contains predeclared admission and all attempted family denominators; predictions.json exact decisions, including abstentions; simple_baselines.json fixed empty-gap/margin rules; unfiltered_requests.json unlabeled canonical/edited request diagnostics; source_content_exclusions.json only unedited lexical-content failures; inputs.json immutable identity chain. No test-based retuning or live-model promotion. Owner Ampixa/user; review2026-11-10, active announcement/research evidence. Retain compact evidence while benchmark consumes it; raw audio remains in controls corpus, NOT on HF; inherited research rights.

## File inventory

- `report.json`: 6763 bytes; SHA-256 `94306736ef617edfe49b9a9231e949edf75fe7c19fa5616c986cac6ac17674f6`.
- `predictions.json`: 342771 bytes; SHA-256 `ff23dffdbd8641d5acbe16735c6966bf08232c1982ac3d51b18b5ad189ea7b7c`.
- `simple_baselines.json`: 4373 bytes; SHA-256 `a13ed74560c8bb5a3e5e0eefc986c0bf6cbb3f28f9e5e8a4c58b5645210d9879`.
- `unfiltered_requests.json`: 264622 bytes; SHA-256 `5f91129b73f8c8ab44efef047aae296ee4c6ea5e6439132693f29fb5a3510d12`.
- `source_content_exclusions.json`: 3 bytes; SHA-256 `37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570`.
- `inputs.json`: 638 bytes; SHA-256 `a9c082e6990a8dd4ed96b9ad11aec0e5d664fa3b400f70cf0834e1ecc9867f86`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-indicf5/bin/python -m scripts.evaluate_anchored_detector

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
