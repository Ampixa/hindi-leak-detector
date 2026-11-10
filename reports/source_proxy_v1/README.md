# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/source_proxy_v1`.
Creation and last review (UTC): 2026-11-06T15:53:45.757869+00:00. Status: active.
Next review: 2026-11-13.

Purpose and current consumer: User-authorized source-proxy baseline; exploratory score, not verified Hindi-leakage accuracy.

## File inventory

- `model.json`: 305830 bytes; SHA-256 `3eb508b136f35ef04820852cb5c76be3762a441eb1facea40127c5bc23a0cec1`.
- `report.json`: 3479 bytes; SHA-256 `538d3930d1791a462df69e5aad51d1edc458914cfb1af771f419d031c4f8259e`.
- `predictions.json`: 34124 bytes; SHA-256 `5c2f33fc06995a9ebdc5ec8494403da7eddb8ae32308e960a8516528fbce9532`.
- `training_inventory.json`: 24547 bytes; SHA-256 `8a2954c8380fc57e9bc64a5ec548c2f0caa32e8bc0ad12a919eab262d1f9482e`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- Driver SHA256 a1409e3cf32db53d14191db6e1953a18268c4901e1f30e70e454487e2de673ec
- python -m scripts.train_source_proxy with the three documented pack/feature inputs
- /Volumes/T9/hindi-detector/reports/development_features_v1 manifest SHA256 534d8c8c1d4cf07ed5d389418fe84f960849c8b0166dd7acadbe052ee7f34c8c
- /Volumes/T9/hindi-detector/reports/audio8_proxy_features_v1 manifest SHA256 7a4cae4b7dfa5bd2524c8a5786e5a9d96bbe24f6791857f69b46b3b449129881
- /Volumes/T9/hindi-detector/reports/comparison_features_v1 manifest SHA256 2870bb03d0d5a87d7235a89c4b0e8720c29298c186187fb0fbbeede30cb6d5cf

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
