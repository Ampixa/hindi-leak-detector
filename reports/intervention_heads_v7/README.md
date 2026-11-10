# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: cdjks-MacBook-Pro-2.local. Original directory: `/Users/cdjk/github/hindi-detector/reports/intervention_heads_v7`.
Creation and last review (UTC): 2026-11-07T15:18:32.576945+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Fit explicit auxiliary schwa request-class components with setting/weight freeze before held-out feature reads. These do not replace genuine Hindi-leakage validation.

## File inventory



`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- scripts/train_intervention_heads.py
- reports/intervention_features_v7
- hindi_detector/interventions.py

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.

Inventory: selection/ retains immutable models.json, hyperparameter choices/trials and exact source snapshot before held-out NPZ reads; evaluation/ retains final request-class metrics/predictions and explicit native/Hindi/prosody accuracy gaps. Any worker.log/pid belongs only to this fitting job; raw logs expire within24h after verification. No hidden embeddings, source voice, class-condition metadata, decoder timing or measured duration are predictors. Known target-only descriptors are identical across pair conditions. Current consumer auxiliary realization-head development; review2026-11-14; retain selected compact weights/metrics/source until superseded, no full checkpoint or new HF upload. Upstream model/data licences continue to apply. Actual Hindi detector objective remains active/incomplete.
