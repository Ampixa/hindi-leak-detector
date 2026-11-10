# Artifact inventory and provenance

Created by Codex (GPT-6), session `/root`, or the user running the documented CLI.
Owner: Ampixa / user. Host: cdjks-MacBook-Pro-2.local. Original directory: `/Users/cdjk/github/hindi-detector/reports/human_review_cleanup_v5`.
Creation and last review (UTC): 2026-11-07T14:31:49.051253+00:00. Status: active.
Next review: 2026-11-14.

Purpose and current consumer: Document removal of unused browser QA diagnostics after verification.

## File inventory

- `cleanup.json`: 2454 bytes; SHA-256 `ad192a949c45e83b08384572d2482a5b453960c55e8e32a659c37aba2b68a682`.

`README.md` is this inventory; no recursive sidecar is required.

## Inputs and reproduction

- review/wordlab.html
- review/pocket-progress.html
- Private browser read-only QA

Use the same source lock and CLI command to reproduce. Code revision: working
tree in this project (initial implementation; no Git revision existed when created).
No model/data uploaded to HF. Known duplicate: source/result sync between cdjk and
k2 for this experiment only. Source licences continue to apply. Candidate outputs
are unreviewed hypotheses, not annotations. Compact metrics retained for this task;
delete after the experiment is closed or superseded. Transient diagnostic outputs
expire within 24 hours after their consumer finishes. No audio copies in this group.

Final cleanup.json checksum after deletion: SHA-256 696262c38d4081bf06e5a928b24cc653a3c64f3b97c3a4a5a5844f3e41848a80, 2508 bytes. The initial inventory hash above describes the predeletion plan; this final checksum is authoritative. Raw browser transcripts are no longer restorable and are not a source/data backup.

Final page/bytecode cleanup appended to cleanup.json: 13767 bytes, authoritative SHA-256 cb97e8ce0c64533b8499a707fecc452bbb043099b7dc058ff73af7411a2bbdbd. Source remains intact; deleted Python bytecode is regenerable by the documented tests. This final hash supersedes prior hashes in this evolving cleanup record.
