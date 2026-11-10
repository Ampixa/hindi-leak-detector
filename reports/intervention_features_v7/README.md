# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: cdjks-MacBook-Pro-2.local. Original directory: `/Users/cdjk/github/hindi-detector/reports/intervention_features_v7`.
Creation and last review (UTC): 2026-11-07T15:16:11.149383+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Frozen phone and site likelihood features for the requested one-site controls, keeping speaker/duration/decoder/condition labels out of model inputs.

## File inventory



`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- scripts/extract_intervention_features.py
- hindi_detector/interventions.py
- data/schwa_interventions_v7/plan.json and render/result/receipts.json
- Fixed phoneme probe at revision2c733782da5604684829819a5eb744c193fe9398

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.

Inventory: cf7_<id>.npz files hold only ctc_site and pooled_phone feature vectors, raw baseline CTC score for later comparison, and parent audio SHA. items.jsonl and final manifest.json enumerate every feature path/size/SHA plus exact request provenance/partitions and explicit failures. source_snapshot.json preserves exact driver/helpers. status.json progress, pid owned worker, worker.log explicit operations/tracebacks. Large feature files on k2 only; small metadata synced to cdjk. Internal research: source model CC-BY-SA4.0; phone/alignment model licences inventoried separately. NOT on HF. Current consumer intervention-head fitting/evaluation; review2026-11-14. Keep features while fitting/test reproduction consumes them; remove redundant tensors/audio/logs within24h after consumer closes, retaining compact selected weights/config/metrics/source. No actual human leakage label fabricated or input corpus modified.
