# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/stem_training_v83`.
Creation and last review (UTC): 2026-11-09T14:46:21.021634+00:00. Status: active.
Next review: 2026-11-16.

Purpose and current consumer: Pretrained5ms localized vowel supervision experiment. protocol/inputs fixed fit rules and pins; status.json atomic live fold/epoch/PID; history.json scalar losses/veto strengths; cv_predictions.json heldout rows/logits/dense known-region diagnostic; report.json absolute gates and ablation; development_predictions.json calibration only; model.pt and no_audio.pt final reproducible heads; frozen_selection.json immutable candidate hashes. No fold/epoch checkpoints. Owner Ampixa/user; active consumer v83 admission and regression work. Review2026-11-10. NOT on HF; inherited research rights. Keep compact configs/results/final selected and explicit ablation heads; diagnostic arrays disposable24h after consumer closes. No new audio or encoders.

## File inventory

- `protocol.json`: 2719 bytes; SHA-256 `6dd3558e28c70624efee2cf073dcc0101dde58babcd117a754520318ac611b7d`.
- `inputs.json`: 3549 bytes; SHA-256 `94f06a1e770c1c7fb9d8af4c7f3460de16de49a31cafd7e4a6898881f5fc4c6e`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-indicf5/bin/python -m scripts.train_stem_deletion

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.

Reviewed 2026-11-09T14:51:44.902632+00:00. files.json complete file inventory with sizes/SHA. Original host k2, owner Ampixa/user. v83 weights SHA ae98b0048e0103e8f5c136b07d3bd68d842bb0efe2eb3a9ab2d3c64abe3d858a; current consumer model comparison/short-vowel improvement and regression audit, not release. Compact head/report copies Mac in same relative group; raw features remain k2 only. Model NOT on HF; unreleased research use, no new licence granted; source/teacher/XLSR/MMS restrictions still apply, release rights not verified. Nextreview2026-11-10. No files deleted. Logs/cache transient after reproduction consumer closes; keep compact failures/results and latest experimental/explicit ablation heads for current comparison.

Local Mac group is a compact partial copy. Raw feature/target caches and large OOF predictions remain on k2. files.json, if present, inventories the original k2 group, not a claim all members were copied here. Local members: README.md, files.json, frozen_selection.json, history.json, inputs.json, model.pt, no_audio.pt, protocol.json, report.json. Same current consumer/expiry as above.

local-files.json explicitly inventories actual local members and their sizes/SHA; original remote files.json may include k2-only members. No model/data on HF.
