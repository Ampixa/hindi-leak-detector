# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/rhythm_interventions_v52`.
Creation and last review (UTC): 2026-11-09T02:51:30.852254+00:00. Status: active.
Next review: 2026-11-16.

Purpose and current consumer: Word-rhythm causal controls. protocol.json frozen before synthesis/score fitting. audio/*.wav Native-spectrum monotone word-duration interventions and WORLD shams, features/*.npz measured40-dimensional contextual rhythm pair and29-column measured query/reference matrices. details/*.json actual MMS word boundaries, free phones and prosody per output. rows.jsonl pins source/output SHA/bytes, intended-vs-measured duration and phone-edit-over-sham audit, explicit exclusions. runtime.json exact frozen acoustic identities, status/report actual progress. No donor pitch/spectrum imported. No human accent labels or live model changes. Consumer rhythm model admission test; review2026-11-10, raw audio/frames expire24h after consumer closes, retain compact metrics/configs/selected state. No HF upload; inherited MMS/probe/source rights.

## File inventory

- `protocol.json`: 2043 bytes; SHA-256 `0d4af87b87f6a84dcd5c0ae188bbb80679802e87a53e055b6997ccc7812596be`.
- `source_snapshot.json`: 33305 bytes; SHA-256 `892cc85769624b9e81b0d05024bdca10c194c03b14225c96ec5a853b3f9ca210`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- .venv-indicf5/bin/python -m scripts.build_rhythm_interventions

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. Generated waveforms, if present, are explicitly described above and listed in inventory_v57.json.
