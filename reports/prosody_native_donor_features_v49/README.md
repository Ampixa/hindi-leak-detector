# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/prosody_native_donor_features_v49`.
Creation and last review (UTC): 2026-11-09T02:33:34.712705+00:00. Status: active.
Next review: 2026-11-16.

Purpose and current consumer: Streaming rendered speech evidence. items.jsonl pins actual source waveform and per-clip detail/NPZ SHA or explicit failure. details/*.json preserve canonical analysis text, original generation text, word times and measured phone/prosody; features/*.npz contain primary phone_log_probs and optional secondary_phone_log_probs. runtime.json pins models/vocab/actual source runtime; no hidden voice embeddings or source-as-error labels. source audio remains in documented plan/teacher mirror. Current consumer phonetic reference/minimal-pair evaluation, review2026-11-09; remove redundant raw frames after consumer closes, retain model/metrics/input pins. No HF upload; inherited licences including shared MMS research restriction.

## File inventory

- `inputs.json`: 421 bytes; SHA-256 `9da4044b41eba5cda4f6a517ded05a5c259d9a534cdcab326ebba13fa9a9c3a3`.
- `source_snapshot.json`: 29393 bytes; SHA-256 `ced4602e6b960f24ad5a97ccdb676a3721d4a87a6e89e05ce9adc2901f3e5fab`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-indicf5/bin/python -m scripts.extract_rendered_vowels --plan PLAN --controller CONTROLLER --namespace NAME --out NEW [--secondary]

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. Generated waveforms, if present, are explicitly described above and listed in inventory_v57.json.
