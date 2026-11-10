# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: Khatra-Dev-Nepals-Mac-Studio.local. Original directory: `/Volumes/T9/hindi-detector/reports/native_schwa_reference_v3`.
Creation and last review (UTC): 2026-11-07T09:08:35.468944+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Center schwa-site evidence on actual same-word native variation instead of treating deletion as an error.

## File inventory

- `reference.json`: 29373 bytes; SHA-256 `c98965c71b0c6c2eb5a9f07fd40622fb868631c07c61e7f4829f05d5f3f7f372`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- /Volumes/T9/hindi-detector/reports/schwa_sites_native_benchmark_v3
- /Volumes/T9/hindi-detector/data/native_wordbank_v3/selection.json
- Sites SHA afa55c5d1a2801610ffa0cdfd958ad3256ccb946e4802f2046b9a9e77ec705ba
- Selection SHA 1c8037b6a4a9e63f44db6a0d768a8514e6b858cc9a828f4e1864e02068049d91
- Driver SHA 4943c3cde2fac60fae17dfc85428b448037833d7182006b1f7bbe83e1c55e633
- python -m scripts.build_schwa_reference --sites reports/schwa_sites_native_benchmark_v3 --selection data/native_wordbank_v3/selection.json --out NEW

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.
