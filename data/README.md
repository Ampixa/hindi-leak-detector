# Data inventory

Created by Codex (GPT-6), session `/root`, 2026-11-06 UTC. Owner: Ampixa/user.
Status active. Last review 2026-11-06; next review 2026-11-13.
cdjk origin: `/Users/cdjk/github/hindi-detector/data`; k2 sync:
`/Volumes/T9/hindi-detector/data`.

`schwa_candidates_v1/` has its own inventory, source inputs, hashes and sizes.
It holds 386 unreviewed Hindi/Nepali lexicon contrasts, not human labels.

On k2 only, `g00005.pocket.16k.wav` and `g00005.audio8.16k.wav` are temporary
16 kHz mono PCM conversions of existing research outputs in
`/Users/k2/madhyam/madhyam-tts/dashboard/public/audio/pocket/` with the matching
`g00005.pocket.m4a` / `g00005.audio8.m4a` names. Created with macOS `afconvert`
(`-f WAVE -d LEI16@16000 -c 1`). Current consumer: initial phoneme/prosody pipeline
smoke test. Text given to the original systems, verified from the existing page:
`सबैको सुरक्षा एनएलजीको इच्छा एनएलजी`. Audio8 is ckpt6000; Pocket is Hindi weights.
Original historical generation author unknown. Conversion author this session.
Hashes are recorded in phone/prosody output-group JSON. These files are not gold
examples of Hindi leakage, and the two systems do not establish a fine-tuning
effect. Source audio rights/consent were not re-established; keep internal.
NOT on HF. Reproduce from the original M4A files. Delete conversions within
24 hours after smoke-test verification, retain compact outputs and original paths.

Session-end status: both 16 kHz smoke-test conversions deleted after successful verification; original M4A inputs remain. See reports/cleanup.json for checksums and sizes.

`audio8_proxy_v1/` documents the 32 matched Audio8 R3 development outputs and
16 temporary reference transfers used by the source-proxy baseline. Its README
and receipts enumerate every waveform, hash, origin, licence and retention rule.
The group follows the user's source-label assumption and contains no human gold.

Human data v5/v6: human_regions_v5 contains pinned metadata recovery; human_g2p_bank_v5 selects 232 original-group examples and 32 word families with alternative G2P priors; human_original_audio_pilot_v6 verifies two original-to-staged WAV identities byte for byte. human_original_audio_v6 is the active expanded verification. Original multi-channel encoded payloads and large metadata remain on k2 T9; cdjk copies are compact JSON/source/inventory only. Do not interpret absence of local payloads as missing remote inputs. Each directory README and records inventory define current purpose, source hashes and retention. No outputs uploaded to HF; original dataset licence applies.

Auxiliary controls (v7): schwa_interventions_pilot_v7 contains eight inspected same-voice one-site request controls. schwa_interventions_v7 freezes48 lexemes/five public voice IDs/two rates and960 renders, with all current human-bank and original TTS focus words excluded. The render receipts identify every30.3MB of waveform output, requested phones, observed timing, size/SHA and split. Current consumer: component error/acoustic-adherence audit and reproducible direct inference; native/Hindi error labels remain unknown. Large audio remains only on k2. Pinned renderer CC-BY-SA4.0 and native helper MIT; no new HF upload; review2026-11-14 and delete unused outputs after consumer closure rather than retaining whole experiments indefinitely.
