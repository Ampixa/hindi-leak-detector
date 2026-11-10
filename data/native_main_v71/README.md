# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/data/native_main_v71`.
Creation and last review (UTC): 2026-11-09T07:23:16.272710+00:00. Status: active.
Next review: 2026-11-16.

Purpose and current consumer: Natural same-teacher minimal vowel requests: see inputs.json for actual word/clip counts, two voices per partition, canonical/edited readings from the same24-layer checkpoint/conditioning/seed. No splice artifacts. Train/dev/test words, conditioning groups and carrier templates are disjoint; all actual Swar vocabulary excluded. Requested phonetic changes are verified through actual frontend/Hindi outputs but realized audio remains unreviewed until acoustic audit. references/*.wav are exact six copied prompts, reference_inventory.json pins sources/SHA. teacher/ later stores the planned generated outputs. Current consumer improved vowel-head training and fresh request-class evaluation; review2026-11-09, delete redundant generated copies after consumer closes. No HF upload; source licences apply.

## File inventory

- `model_pair.json`: 1575 bytes; SHA-256 `06b469ad88fff6ea655c3c39ba7b70082ed9bb30c607afe61141b61c16b370e6`.
- `selection.json`: 3009189 bytes; SHA-256 `c9a6c521308fc6cecec1f03b3bdfbe297da59fdb081ccfca614d0ac4fa6eec91`.
- `inputs.json`: 2519 bytes; SHA-256 `c0e61d865dd8ad792ba5552b0c793e46bc5520e5f663b9d6666de770748df60e`.
- `source_snapshot.json`: 20215 bytes; SHA-256 `b1c67a0b2fc159a66fc2c43da3c04b60a6d1505d3c5021fbc8819bdf7b44d28e`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-indicf5/bin/python -m scripts.build_minimal_vowel_plan

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.

Actual main71 corpus:512 deletion lexemes,2404 requests, four conditioning voices per partition (12 prompts). Canonical requests every fold; natural edited requests only in test for unspliced requested-change diagnostics. Exact canonical/edited phone-collision groups and voice IDs do not cross folds. No quality rendering/fitting consumer here. Native_main_v70 was an unlaunched384-word sizing plan; no audio/model generated from that plan. Current consumer reviewed local-deletion training and frozen test; next review2026-11-10. Physical-label, generated-request and real-error labels remain distinct.
