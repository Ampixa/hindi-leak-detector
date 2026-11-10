# IndicF5 / audio8 matched comparison pack

Created by Codex (GPT-6), session `/root`, 2026-11-06 UTC. Owner: Ampixa/user.
Origin code/workspace cdjk `/Users/cdjk/github/hindi-detector`; audio consumer k2
`/Volumes/T9/hindi-detector/data/indicf5_audio8_v1`. Status active for the explicit
detector goal. Next review 2026-11-13. No human leakage labels exist at creation.

Metadata inventory: `source_manifest.json` is an unchanged copy of
`khalti-gpu:/home/ampixa/tts_bench_v0/manifest.json` (119 native recordings,
253 generation jobs, eight reference voices). `source_audio8_r2.jsonl` and
`source_audio8_r3.jsonl` are unchanged completed synthesis receipts for the
respective audio8 runs under that root. `selection.json` will identify selected
items, exact text, source references and hashes; `transfer_files.json` will list
every selected audio member, original path, destination and expected SHA-256.
Actual member sizes and copy verification belong in `copy_inventory.json`.
`benchmark_source.json` records the exact public NepTTS sentence-list revision and
verified Git blob hashes used for selecting the reserved evaluation subset.

Audio source author: historical benchmark generation by Codex, per source README;
original human recording authors are not inferred and remain the source corpus
speakers. Native recordings are comparison references, not automatic negative
labels for all pronunciation features. Model identity is never a leakage label.
Benchmark texts are reserved for evaluation/review, not detector training.

`audio/` contains only selected native, reference and generated audio8 clips on
k2. Newly synthesized IndicF5 output will have its own manifest, hashes and source
revision. No archive model is copied. Existing private research audio/voice rights
continue to apply; internal use only, no public site/HF upload. Other copies:
original L40 benchmark audio, k2 selected copies, small metadata on cdjk. Do not
copy audio to cdjk by default. Reproduce from locked source receipts and checksum
verification, then run the pinned IndicF5 adapter. Discard superseded generated
audio within 24h of finishing its consumer; retain compact selections, labels,
model/config hashes and conclusions. No blind source-data cleanup is authorized.
