# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/k2_phone_probe_v1`.
Creation and last review (UTC): 2026-11-06T07:45:23.894485+00:00. Status: active.
Next review: 2026-11-13.

Purpose and current consumer: Phone-recognizer smoke test for annotation; no accent labels.

## File inventory

- `phone_probes.json`: 1229 bytes; SHA-256 `3b0508aaea4fef5cf3b2fe0f4fadcdcf166eed6e11e9b43db886a0e2d1082af0`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- /Volumes/T9/hindi-detector/hindi_detector/cli.py probe-audio data/g00005.pocket.16k.wav data/g00005.audio8.16k.wav --model-directory models/phoneme_probe --out reports/k2_phone_probe_v1
- Pinned model revision in models/README.md

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
