# Controlled identity resynthesis and voice changes

Created by Codex GPT-6 /root, 2026-11-07 UTC. Owner Ampixa/user. Status active.
Origin k2 /Volumes/T9/hindi-detector/reports/voice_controls_v3; compact metadata
copy on cdjk. Consumer: testing voice/vocoder confounds in word/prosody scores.
Next review 2026-11-14. No HF/public upload. Input corpus/privacy limits apply.

Inventory: source_snapshot.json stores exact UTF-8 experiment/helper code and SHA;
generated.jsonl enumerates every converted <original_clip>_<control>.wav, size,
SHA, original metadata, mode and clipping measurement. <same_stem>.json contains
word alignment, phone estimates and normalized contours for the actual converted
WAV. report.json inventories all detail hashes/sizes and preservation/score drift.
Source-model pins and licence are under models/openvoice_v2. Two reference files
are read-only native corpus inputs; no target-speaker vectors are retained or
given to accent heads. This README inventories itself. Raw spectral tensors and
hidden speech embeddings are transient, never saved.

A matched statement/question from native, Audio8 R3 and IndicF5 is converted into
identity-resynthesized audio and two common target voices. Identity resynthesis
controls for adding a common decoder; voice changes are compared separately.
Original source WAVs remain unchanged. Automatic duration/timing/pitch/phone
preservation gates are predeclared in reports/voice_controls_v3_plan.json. A
failed gate does NOT establish voice sensitivity: conversion may have altered
pronunciation/prosody. These are diagnostic probes, not human accent labels.

Run scripts/run_voice_controls.py --out NEW in the pinned word runtime. Exact
source can be restored from the snapshot to its original project-relative paths.
Keep WAVs only while this listening/control study consumes them; delete obsolete
takes within 24h of closure, retaining compact metrics, pins and selected evidence.

Compatibility correction: old_global_model fields are excluded from interpretation. Original audio scores differed from the historical cached-feature runtime; see ../voice_runtime_compatibility_v3/report.json. The cause is unestablished. Word-branch comparisons use the same current word runtime throughout. Only four automatic preservation gates passed, including two actual voice swaps from one native original; this is insufficient to establish speaker invariance.
