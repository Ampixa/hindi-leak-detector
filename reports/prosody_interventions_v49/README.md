# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/prosody_interventions_v49`.
Creation and last review (UTC): 2026-11-09T02:37:52.001046+00:00. Status: active.
Next review: 2026-11-16.

Purpose and current consumer: Matched WORLD pitch interventions for frozen164 distinct word contexts. Same Native carrier spectral envelope/aperiodicity/voicing mask/spans in each condition; only focal F0 changes. Third-Native donors must pass isolated lexical ASR. rows.jsonl records source SHA, actual WAV/features SHA, split, eligibility and measured contour evidence. audio/*.wav actual16k float outputs; features/*.npz actual remeasured query/reference prosody and300-dimensional contextual pair. status.json/report.json explicit progress and exclusions. Carrier/donor source recordings remain in original documented packs; no spectrum/phonetic substitution. Source-style labels, not Hindi-error gold. Active consumer causal-prosody training and withheld-word evaluation; owner Ampixa/user, review2026-11-10, redundant WAVs/frames expire24h after consumer closes. No HF upload; inherited licences. Original scripts retained in source_snapshot.json.

## File inventory

- `inputs.json`: 374 bytes; SHA-256 `4d3d49f8c8894c7bf42193fa44283ccf8248fac4ed5257f1d28769d2b9e3b6c2`.
- `source_snapshot.json`: 19378 bytes; SHA-256 `54175273e285ef35f3c2b3a316bffe5b5f360495112fd5922ddc3f407ca1cbde`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-indicf5/bin/python -m scripts.build_prosody_interventions

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. Generated waveforms, if present, are explicitly described above and listed in inventory_v57.json.
