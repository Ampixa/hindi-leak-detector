# Acoustic probe cache inventory

Created by Codex (GPT-6), session `/root`, 2026-11-06 UTC. Owner: Ampixa/user.
Origin host/path: k2, `/Volumes/T9/hindi-detector/models`.
Status: active. Last review 2026-11-06; next review 2026-11-13.

`phoneme_probe/` is downloaded on k2 only, not copied to cdjk. Current consumer:
the Hindi-transfer pilot's phoneme transcription / CTC contrast probe. Model:
`facebook/wav2vec2-xlsr-53-espeak-cv-ft`, HF revision
`2c733782da5604684829819a5eb744c193fe9398`. Model-card licence Apache-2.0.
Expected size approximately 1.3 GB. No new training is represented by this model.

Exact file inventory: `config.json` (architecture/blank ID),
`preprocessor_config.json` (16 kHz input normalization), `vocab.json` (392 phone
labels), `tokenizer_config.json` and `special_tokens_map.json` (token metadata),
`README.md` (upstream model card), `pytorch_model.bin` (upstream weights),
`inventory.json` (computed byte sizes and SHA-256 for all downloaded files).
`.cache/huggingface/` holds per-file download metadata, lock files and temporary
partial downloads maintained by huggingface_hub; delete together with the model
when no download process is active. `models/README.md` documents the group.

Reproduce: `python scripts/fetch_acoustic_model.py`. Original author is Meta;
downloaded by this session. Only this task-local copy is intentionally created;
other machine caches were not exhaustively hashed. No model is uploaded to HF.
Delete when the pilot ends, unless it earns a documented continuing consumer.
Acquisition output `../reports/k2_model_fetch.log` is disposable within 24h after
download verification; successful/failed run outcome is retained in compact notes.

`pocket_hindi_parent/` is the second, k2-only cache, acquired by
`scripts/run_pocket_pilot.py`. Source: `saryps-labs/pocket-tts-hindi`
@`6bff9c1e776bb90c68dd05c2467c62d1ecdf1b51`; current card says CC BY 4.0;
`NOTICE.md` records component attributions. Inventory: `model.safetensors`
(parent TTS weights), `config.yaml` (architecture), `tokenizer.model`
(SentencePiece), `README.md` and `NOTICE.md` (upstream documentation),
`inventory.json` (computed hashes and sizes), plus Hugging Face download metadata
under `.cache/`. Expected weights around 219 MB. Keep only while the pilot needs
the pinned parent. The existing Nepali fine-tune remains at its original k2 path;
no extra copy or training checkpoint backup is made.

`indicf5/` (k2 T9 only) is the IndicF5 comparison model, from
`ai4bharat/IndicF5@fb652a96e508091045890dae5bb99fe511facbcf`. Its upstream
card declares MIT. Current consumer: matched Nepali synthesis for this goal.
Created by Codex GPT-6, session /root, 2026-11-06 UTC. Expected weights
1,402,789,408 bytes; verify upstream LFS SHA-256
`ba7f3671180fb7784e24bd1dafc96e729a38ce02e7f6d3877cdef32525a1865c`.
Inventory: `model.safetensors` (generator), `config.json` (model metadata),
`model.py` and `f5_tts/api.py` (upstream API), `f5_tts/model/**/*.py` (architecture),
`f5_tts/infer/utils_infer.py` and `__init__.py` (inference),
`checkpoints/vocab.txt` (tokenization), `README.md` (upstream card),
`inventory.json` (every member, sizes and SHA-256), `.cache/huggingface/` (download
metadata). Exact selected files are enumerated before download by the pinned HF
API. Existing credentials are read into memory only and never copied here.
Reproduce with `scripts/acquire_indicf5.py`; supply the authorized credential file
path at runtime. Do not accept model gates or upload audio on the user's behalf.
Keep while the active detector goal consumes the model; next review 2026-11-13.
`indicf5/vocos/config.yaml` and `indicf5/vocos/inventory.json` pin the Vocos
architecture to `charactr/vocos-mel-24khz@0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21`.
Vocos weights are already bundled in the IndicF5 safetensors under the inspected
`vocoder._orig_mod.` prefix; the adapter strictly loads those keys without
downloading a second vocoder checkpoint.

`kala_controls/` is the auxiliary same-voice phoneme renderer, HF ampixa/real-nepali-v0.2-kala@45b7d3052a0ad01d5741213ff041e65db0829f5a, ONNX SHAab85d482a61d9d76dad79b492f5ef3985698b24735497fe9646cd7a0c4b75ffd. Only77,746,361-byte ONNX/config/speaker map/card acquired; no935MB training checkpoint. Current consumer: verify schwa-intervention acoustics, audit component errors and reproduce direct inference. This does not replace Pocket as the preferred TTS reference. Its own README/inventory record all actual files, licenceCC-BY-SA4.0, source identities and seven-day review. No HF upload.
