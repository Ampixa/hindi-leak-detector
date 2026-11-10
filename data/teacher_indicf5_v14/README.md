# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/data/teacher_indicf5_v14`.
Creation and last review (UTC): 2026-11-08T05:13:31.779636+00:00. Status: active.
Next review: 2026-11-15.

Purpose and current consumer: User-requested corrected model pair. selection.json freezes768 texts/1536 reference jobs and smoke IDs; model_pair.json explicitly pins teacher24layers and AI4Bharat IndicF5, never legacy fallback. references/*.wav are derived shared <=5s prompts; reference_inventory.json pins original/crop bytes, text, boundaries and alignment. Generator subgroups retain actual per-WAV hashes/status. No pronunciation gold inferred from either model. Current consumer corrected synthesis/ASR and paired word evaluation.

## File inventory

- `model_pair.json`: 1575 bytes; SHA-256 `06b469ad88fff6ea655c3c39ba7b70082ed9bb30c607afe61141b61c16b370e6`.
- `inputs.json`: 682 bytes; SHA-256 `a6f2903e22c56eb38542ff6f6c59e22f45d56400fd57e0da227535ef397bc9f5`.
- `source_snapshot.json`: 9237 bytes; SHA-256 `0339ac2f1a2d76b8dd6716962376420a340f1e9dd6bf6eb06654e3f7f966e53d`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-indicf5/bin/python -m scripts.build_teacher_f5_plan --out NEW
- Shared prompts reproduce from original WAV SHA and measured end sample; no normalization, source transcript cropped at aligned word boundary. Source and model licences continue to apply. No HF upload; review audio by2026-11-09 and remove when superseded.

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.

path_portability_amendment.json records a pre-generation absolute-to-relative path-only correction of selection.json for L40 transfer; original builder source is preserved exactly in source_snapshot.json. Updated builder source emits relative paths directly.

This cdjk copy is a partial metadata mirror only: model_pair.json and inputs.json. Actual selection,16 references and generated WAVs remain on k2 and L40.
