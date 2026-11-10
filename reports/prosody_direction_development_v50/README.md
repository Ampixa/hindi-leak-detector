# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/prosody_direction_development_v50`.
Creation and last review (UTC): 2026-11-09T02:42:38.206525+00:00. Status: active.
Next review: 2026-11-16.

Purpose and current consumer: Development-only seven-input reference-conditioned pitch-style baseline. Contrast query to same-word NativeB and independently rendered HindiB, whereas physical positive used HindiA donor on NativeA carrier. This tests whether word-conditioned direction helps the failed global style head. model.pt selected state/training-only normalization/high threshold; model_inventory.json SHA/bytes. scores.json train/dev only, no v49 reserved arrays loaded because that experiment was already inspected. Not a real Hindi-error model and not live. Current consumer decide whether fresh independent-word synthesis is justified; review2026-11-10. Keep compact results and selected checkpoint only while active; no rawaudio copies/HF upload; inherited rights.

## File inventory

- `report.json`: 1128 bytes; SHA-256 `af3a3cac72569cff2efa030cb172aaa20d478f0f5ebfeaae8344381c383dfd76`.
- `history.json`: 49766 bytes; SHA-256 `3a4b76547c1ed709ab16b2a1da74e0012dbfcaef65af355a532ec4de6415e0c6`.
- `scores.json`: 177814 bytes; SHA-256 `62d330b53717930334cac44f860eaa6ba377ee275c7932e476614cb160184bfa`.
- `inputs.json`: 192 bytes; SHA-256 `98e2047c799ddca1b8e8f05230d7138f321f3f22219c8375fb1807e2dd201e5d`.
- `source_snapshot.json`: 9609 bytes; SHA-256 `1f4340f9a7754392be858902ee96e2085e1f4d72948e1358620a00b2bf2de182`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-indicf5/bin/python -m scripts.fit_prosody_direction

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. Generated waveforms, if present, are explicitly described above and listed in inventory_v57.json.
