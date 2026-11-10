# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: cdjks-MacBook-Pro-2.local. Original directory: `/Users/cdjk/github/hindi-detector/data/human_original_audio_v6`.
Creation and last review (UTC): 2026-11-07T14:34:52.906771+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Recover original utterance audio for the selected 232 human word examples and verify staged WAVs against the actual inspected native recipe. Original channels retained; no full Parquet shards saved.

## File inventory



`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- data/human_g2p_bank_v5
- data/human_regions_v5
- Pilot reproduced two original-to-staged WAVs byte for byte
- python -m scripts.verify_original_human_audio --out data/human_original_audio_v6 --max-row-groups 400; credential stdin only. Resume only after process exit/failure and source/input/checksum verification with --resume.

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.

File inventory: plan.json pins selection SHA and every selected source shard/row; source_snapshot.json captures exact driver, text fingerprint, comparison and staging helper source; audio/<clip_id>.source holds original unmodified encoded bytes, format/channels/rate and size/SHA enumerated by recordings.json. row_groups.json records projected chunk traffic/coverage, status.json atomic progress, report.json measured final coverage, inventory.json final file hashes. failure.json and failure_inventory.json, if present, expose sanitized failure and retained partial-file identities. pid identifies the owned worker; worker.log captures explicit operations/failures and expires within 24h of experiment/result review. Original audio is an active prosody/source-verification consumer; review by 2026-11-14 and delete if superseded/no consumer remains, retaining compact results. No original corpus source deleted. Licence: upstream IndicVoices-R dataset terms continue to apply; internal research. HF source ai4bharat/indicvoices_r@5f4495c91d500742a58d1be2ab07d77f73c0acf8; our outputs NOT uploaded to HF. Known duplicate: two selected raw clips were independently retained in human_original_audio_pilot_v6 for recipe verification; compare payload SHA before later deduplication. Other raw clips not copied to cdjk. Native acceptability and Hindi-error labels remain unknown. Stereo downmix is checked only via the explicitly inspected recipe, never silently used as a new accent label.

Source amendment: source_amendments.json preserves exact prior/new sources and why the fourth stereo payload changed the helper from a hard channel-correlation gate to an explicit prosody-review flag. failure_history.json preserves the inspected initial failure before resuming. The old failure_inventory.json is historical pre-resume evidence. Originals remain unchanged; this permits forensic recipe verification, not automatic prosody acceptance.
