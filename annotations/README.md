# Human pronunciation and prosody labels

Created by Codex GPT-6, session /root, 2026-11-06 UTC. Owner Ampixa/user.
Origin k2 `/Volumes/T9/hindi-detector/annotations`. Status active. Next review
2026-11-13. Current consumer: actual human labels for the Hindi leakage detector
goal. No labels exist until an annotator explicitly submits a review.

Inventory: `labels.jsonl` is append-only review events from review_server.py;
each event records reviewer initials/name supplied by the person, self-reported
native Nepali competence, target variety, timestamp, clip and audio hash,
pronunciation judgment, separate prosody judgment, optional schwa spans and notes.
Multiple events may update a person's judgment; raw events are not overwritten.
`review_server.log` records requests and explicit failures, never HF credentials.
`review_server.pid` records only the task-owned server process. These two runtime
files are disposable when that service ends; keep labels and their source hashes.

`development_labels.jsonl` stores separate reviews of the native/IndicF5
development pack. These events carry their actual train/development split;
`development_review.log` and `development_review.pid` belong to its loopback
service on port 8769, with the same runtime retention policy. Labels on port 8768
remain reserved evaluation. Never merge their split assignments or infer them
from the model family. The same blind form supports both services.

This first pack is reserved evaluation: labels carry split=reserved_evaluation
and must be refused by any detector training loader. Model family is withheld
from the reviewer; no label is inferred from family/native-source identity.
Uncertain, other-error and disputed labels are preserved, not coerced into Hindi.
Human competence is self-reported, not externally certified. No public sharing,
chat message, email, HF upload or transcript disclosure is authorized.

Reproduce the service with `python -m hindi_detector.review_server --pack ...
--page review/annotate.html --labels annotations/labels.jsonl`. Only loopback is
bound; the user's desktop reaches k2 through an SSH port forward. Labels require
backup before any deletion; do not regenerate human judgments from model output.
Retain compact labels while the detector project consumes them; delete runtime
logs within 24h after service completion. Known duplicate: project source/README
on cdjk; labels remain on k2 unless an explicit verified sync is made.
