# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: cdjks-MacBook-Pro-2.local. Original directory: `/Users/cdjk/github/hindi-detector/reports/cross_system_review_access_v80`.
Creation and last review (UTC): 2026-11-09T09:15:08.384504+00:00. Status: active.
Next review: 2026-11-16.

Purpose and current consumer: Private read-only cross-system review access. route.json records exact private bind/destination, forward.launch.json owned SSH PID/command/time, forward.log explicit forwarding errors; qa.json later records HTTP/UI/audio checks. No source audio/model copies. Owner Ampixa/user; review2026-11-10; current consumer listening. No public release or messages, NOT on HF; logs disposable24h after review closes. Stop only owned SSH PID when consumer finishes.

## File inventory

- `route.json`: 208 bytes; SHA-256 `19908156a0a0a847db039a81a1534ba35b47faed805ca04d1442f45b6cd1b274`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- ssh -N -L100.115.106.117:8782:127.0.0.1:8782 k2

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.

qa.json now records completed HTTP/range/unknown-route/browser and semantic parity checks plus end-of-session Mac disk measurement. Raw source waves stay on k2; no public exposure or messages. Compact summaries/results are copied to this Mac for current review, not raw audio or encoders.

`inventory.json` lists current compact members, sizes and SHA-256; mutable service logs deliberately have no frozen hash. This manifest does not hash itself.
