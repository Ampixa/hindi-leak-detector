# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/source_proxy_v2`.
Creation and last review (UTC): 2026-11-07T07:24:13.735221+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: V2 native-anchored duration-robust proxy improvement; no validated accent accuracy claim.

## File inventory

- `model.json`: 209984 bytes; SHA-256 `ec6b1cb79428dc0e59bf736df3409ccb78f420c4e147777452a790f3272f62f2`.
- `report.json`: 6524 bytes; SHA-256 `242288bb8cdacd63eef7192112277bc47a8e3c8c0f2cae0606d33b3a68323fbc`.
- `candidates.json`: 12661 bytes; SHA-256 `35335f606c24234dab3bce2956165d09c18b5a18e20976552e8ebef06972542e`.
- `predictions.json`: 20810 bytes; SHA-256 `79fae013416a2376c94fdcf739adece3cd719c5a9a4cf8df91768b5ba1d925bd`.
- `probe_predictions.json`: 49801 bytes; SHA-256 `6621d2de03877fa54d3505235fb232f914f0934957ddd18bafed78aff7b99863`.
- `training_inventory.json`: 148241 bytes; SHA-256 `ba99366bef73a5dd9548a17240f5a8258ef67c912e58c3411fae50fdd773b1b9`.
- `development_inventory.json`: 49846 bytes; SHA-256 `870335483c087f70533c6ac7bc2e7bfd364e28899a75395001f0e747b731ba6d`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- Plan SHA256 5dd6f192e48b7be31de7bb0c6b04e74acc50d9f025e5750882adfbb357f11f18
- Driver SHA256 8d1ae88ef490be3f90f0d790784961fd6630f3cd8956c15b2daf8da2faa3447e
- Fitter SHA256 295860fd6820b783ee780a93aa1e42491221b1d865c9a2100f7603daffb23b2a
- /Volumes/T9/hindi-detector/reports/proxy_windows_v2 manifest SHA256 db46742c38fd45b63d17619a9e1a2bf519b3373fa419f2e5733934c9e213713b
- /Volumes/T9/hindi-detector/reports/native_probe_windows_v2 manifest SHA256 cb28877dfbe21edf69e8c7b95ea6f2ca9523c994984ba6c75811aeb33e9e91cf
- /Volumes/T9/hindi-detector/reports/comparison_features_v1 manifest SHA256 2870bb03d0d5a87d7235a89c4b0e8720c29298c186187fb0fbbeede30cb6d5cf
- python -m scripts.train_anchored_proxy with documented windows/probe/test/v1-head/plan paths

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
