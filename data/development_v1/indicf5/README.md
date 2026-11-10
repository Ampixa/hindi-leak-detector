# IndicF5 development synthesis

Created by Codex GPT-6, session /root, 2026-11-06 UTC. Owner Ampixa/user.
Origin k2 `/Volumes/T9/hindi-detector/data/development_v1/indicf5`. Status active.
Next review 2026-11-13. Current consumer: independent training/development
comparison examples for the Hindi-leakage detector, with human labels required.

Inventory: `run.json` records model, code and selection hashes and inference
settings; `items.jsonl` records one output receipt, text, voice, seed, duration
and SHA-256 per job; `dev_*.wav` contains the 32 generated 24 kHz mono outputs.
`synthesis.log` records explicit progress/failures, disposable within 24h after
the completed run is reviewed. Until run.json says complete_unreviewed_audio,
this is a partial run. No output is a positive leakage label automatically.

Inputs: ../selection.json, sixteen verified source speaker groups and distinct
native references under /Volumes/T9/indicvoices_r_ne/audio, plus the pinned
IndicF5/vocoder bundle documented in models/README.md. Corpus rights and model
licences continue to apply; private internal use, no HF/public upload. Original
native clips are not copied or changed. Newly generated audio stays on k2.

Reproduce: .venv-indicf5/bin/python -m scripts.run_indicf5_comparison
--device mps --pack data/development_v1. This uses exactly the evaluation model
and sampler settings on separate texts/speakers. Retain audio while native review
and model fitting consume it; remove superseded takes within 24h after use, while
retaining source hashes, compact receipts and irreplaceable human labels.
