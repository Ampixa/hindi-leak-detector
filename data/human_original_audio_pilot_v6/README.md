# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: cdjks-MacBook-Pro-2.local. Original directory: `/Users/cdjk/github/hindi-detector/data/human_original_audio_pilot_v6`.
Creation and last review (UTC): 2026-11-07T14:29:29.825895+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Verify selected original IndicVoices-R utterance audio and staged waveform correspondence before accepting human prosody references. Pilot reads one actual Parquet row group; no full shards retained.

## File inventory



`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- scripts/verify_original_human_audio.py
- data/human_g2p_bank_v5/selection.json
- data/human_regions_v5/metadata.sqlite
- python -m scripts.verify_original_human_audio --out data/human_original_audio_pilot_v6 --max-row-groups 1; HF credential stdin only

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.

Runtime inventory: source_snapshot.json captures exact driver/metadata fingerprint/wave-comparison code. status.json holds atomic progress; failure.json, if present, records sanitized explicit failure and source frames. recordings.json lists every retained audio/<clip_id>.source payload with original container/subtype, size/SHA, original shard/row, staged SHA, and waveform comparison. row_groups.json lists read source chunks and compressed sizes. report.json records actual coverage/limitations. inventory.json inventories all retained nonlog files by path/size/SHA. pid identifies only this worker; worker.log retains explicit operations/failures, disposable within 24h of experiment closure. No credential retained. These original payloads are active detector/prosody-verification inputs, not accent labels. No full Parquet, unselected audio, or new checkpoint is saved. Licence/usage: original IndicVoices-R terms continue to apply; internal research only. Parent dataset HF pin 5f4495c91d500742a58d1be2ab07d77f73c0acf8. Our outputs NOT on HF. Next review 2026-11-14; delete original payloads if no current verification/listening consumer remains, retaining compact identities/results. No other original corpus input is removed.

Resolved/active inspection inventory: the strict mono check stopped before accepting any source recording. failure.json and worker.log preserve that explicit failure. inspection.json, if present, records the exact independent inspector source and actual original channel/format measurements; audio/original_row_<index>.source are the exact two selected payloads, enumerated by byte size and SHA in inspection.json. No downmix occurs in this diagnostic. These payloads are retained for channel/prosody verification, not for training. The historical staging recipe averages channels, whose content must be inspected first.

Final pilot inventory extension: waveform_correspondence.json captures the explicitly chosen zero-shift polyphase comparison; lag_diagnostic.json captures the bounded timing-shift diagnostic; recipe_reproduction.json captures exact native sphn0.2.1/mean-channels/peak0.95/PCM16 reproduction and runtime hashes. report.json/status.json record two byte-identical matches, never generalizing to the remaining recordings. inventory.json gives final nonlog file hashes. All inspector source is captured in its respective result JSON. Original audio remains in unchanged two-channel form. Historical pipeline revision remains unknown.
