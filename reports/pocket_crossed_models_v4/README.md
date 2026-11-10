# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/pocket_crossed_models_v4`.
Creation and last review (UTC): 2026-11-07T11:09:51.992163+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Train word-focused preferred-reading comparisons across different voices with same-architecture checkpoint controls.

## File inventory

- `models.json`: 284898 bytes; SHA-256 `90d6cfec8b3559c221b93609f03ad40593cd1ffdca2679c520dff6d643cd610b`.
- `report.json`: 15157 bytes; SHA-256 `c6c53e37828ade6f21c8cb8f339f505af56843556ab4cfe3b25f44b544973758`.
- `selection.json`: 2803 bytes; SHA-256 `f885d153d0520c9eddd98c638fed83e0bbcf6b169fc9a6e711e161bab0e9bc20`.
- `test_predictions.json`: 1938662 bytes; SHA-256 `6db5d3500988f329d83cc5c681acf5f822d8d9961f5227e5b3ed0442ceb352db`.
- `source_snapshot.json`: 27025 bytes; SHA-256 `44af0d07b27b8b219103bd466c2bc4676a2e95024872e7c40a83a23d53730a69`.
- `folds.json`: 459 bytes; SHA-256 `7f33c04bcb7016d3f356dd11149039e4f56e5af658010233211e982914d3868c`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- Plan SHA 24e9999c680d94cda46d3d25a63b96b80fa30c993b83cdca0c38bba5545623fa
- Input evidence manifest hashes {"reports/wordlab_matrix_f5_v3": "ec6c8538ecd4d9974acf6d3c54e78da19b4bb78acedf9813958085a0745126e8", "reports/wordlab_pocket_matrix_v4": "d5a7dabcb8e738018a78c62aec7d4176db8241e468fe79ecc665a91bdde69425"}
- python -m scripts.train_crossed_words --f5-evidence reports/wordlab_matrix_f5_v3 --pocket-evidence reports/wordlab_pocket_matrix_v4 --plan data/pocket_wordmatrix_v4/plan.json --out NEW

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
