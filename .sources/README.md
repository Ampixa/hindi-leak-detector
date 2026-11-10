# Pinned source cache

Owner: user / Ampixa. Created by Codex (GPT-6), session `/root`, on cdjk,
2026-11-06 UTC, for the Hindi-checkpoint leakage detector experiment.
Original path: `/Users/cdjk/github/hindi-detector/.sources`.

Inventory: `frontend/` contains the Python modules, phone tables, seed lexicon,
README and licence listed individually in `../sources.lock.json`; `hindi/`
contains the Hindi pronunciation lexicon and README/licence listed there.
`inventory.json` records every downloaded member, byte size, URL and SHA-256.
No models, audio or HF artifacts are stored here. NOT on HF. Exact source Git
revisions and Git blob hashes are locked; `scripts/fetch_sources.py` verifies
before using any file. Frontend code MIT; external Nepali seed data carries
upstream terms described by the frontend README. Hindi repository has an MIT
licence; underlying dictionary data rights need separate review before release.

Purpose/current consumer: candidate mining in this project. Status: active.
Expected size under 9 MB; actual byte totals in inventory. Last review:
2026-11-06; next review: 2026-11-13. Delete cache after candidate work ends;
reproduce with `python3 scripts/fetch_sources.py`. No backup required because
all inputs have pinned remote URLs. No other duplicate was created by this job.
Source inputs are unchanged upstream code/data, not newly authored gold labels.

`neptts/` adds the pinned benchmark sentence inventory, README and licence from
`Ampixa/neptts-bench@f44010141a727ab1c3ead8a36fecc2a48918ac26`. Exact files,
sizes and Git blob hashes are in the root source lock, also recorded in
`../data/indicf5_audio8_v1/benchmark_source.json`. Current consumer: selecting
reserved evaluation/review examples for the active detector goal. These benchmark
texts are excluded from detector training. Code licence is MIT; upstream corpus
and human recording rights continue to apply separately. No HF upload.
