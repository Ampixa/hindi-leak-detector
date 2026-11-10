from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

from . import acoustic, candidates, ctc, prosody
from .artifacts import write_group

LOG = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Hindi-checkpoint pronunciation transfer experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    mine = sub.add_parser("mine", help="mine unreviewed Hindi/Nepali schwa contrasts")
    mine.add_argument("--frontend-root", type=Path, required=True)
    mine.add_argument("--hindi-csv", type=Path, required=True)
    mine.add_argument("--out", type=Path, required=True)
    pitch = sub.add_parser("prosody", help="measure audio; no accent prediction")
    pitch.add_argument("audio", type=Path, nargs="+")
    pitch.add_argument("--out", type=Path, required=True)
    score = sub.add_parser("score-audio", help="compare reviewed phone hypotheses for a word crop")
    score.add_argument("audio", type=Path)
    score.add_argument("--model-directory", type=Path, required=True)
    score.add_argument("--hypotheses", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    probe = sub.add_parser("probe-audio", help="unreviewed phoneme transcription for annotation")
    probe.add_argument("audio", type=Path, nargs="+")
    probe.add_argument("--model-directory", type=Path, required=True)
    probe.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.command == "mine":
            rows, counts = candidates.mine(args.frontend_root, args.hindi_csv)
            write_group(args.out, {"candidates.json": rows, "metrics.json": counts},
                        "Native-review queue for schwa contrasts; not training gold.", [" ".join(sys.argv), "sources.lock.json"])
            print(json.dumps(counts, ensure_ascii=False))
        elif args.command == "prosody":
            rows = [prosody.analyse(path) for path in args.audio]
            inputs = [f"{path.resolve()} sha256={hashlib.sha256(path.read_bytes()).hexdigest()}" for path in args.audio]
            write_group(args.out, {"features.json": rows}, "Prosody smoke check; descriptive only.", [" ".join(sys.argv), *inputs])
            print(json.dumps(rows, ensure_ascii=False, allow_nan=False))
        elif args.command == "probe-audio":
            rows = []
            for path in args.audio:
                log_probs, vocab, blank = acoustic.posterior(path, args.model_directory)
                inverse = {value: key for key, value in vocab.items()}
                best = log_probs.argmax(axis=1).tolist()
                tokens = [inverse[p] for i, p in enumerate(best) if p != blank and (i == 0 or p != best[i - 1])]
                rows.append({"audio": str(path.resolve()), "tokens": tokens,
                             "status": "unreviewed_crosslingual_phone_probe", "frames": len(best),
                             "model_directory": str(args.model_directory.resolve()),
                             "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            write_group(args.out, {"phone_probes.json": rows}, "Phone-recognizer smoke test for annotation; no accent labels.",
                        [" ".join(sys.argv), "Pinned model revision in models/README.md"])
            print(json.dumps(rows, ensure_ascii=False))
        else:
            hypotheses = json.loads(args.hypotheses.read_text())
            if hypotheses.get("review_status") != "approved" or not hypotheses.get("reviewer"):
                raise ValueError("native-reviewed hypotheses with review_status=approved and reviewer are required")
            log_probs, vocab, blank = acoustic.posterior(args.audio, args.model_directory)
            def encode(key: str) -> list[int]:
                phones = hypotheses[key]
                if not isinstance(phones, list) or not phones or any(not isinstance(p, str) for p in phones):
                    raise ValueError(f"{key} must be a nonempty list of recognizer phone tokens")
                unknown = set(phones) - vocab.keys()
                if unknown or any(p in {"<unk>", "<pad>", "<s>", "</s>"} for p in phones):
                    raise ValueError(f"unsupported/special phones in {key}: {phones}; no silent mapping allowed")
                return [vocab[p] for p in phones]
            evidence = ctc.compare(log_probs, encode("target_tokens"), encode("hindi_tokens"), blank)
            result = {**evidence.to_dict(), "hypotheses": hypotheses, "audio": str(args.audio.resolve()),
                      "model_directory": str(args.model_directory.resolve()),
                      "note": "Positive ratio supports the Hindi hypothesis; calibration and native validation are still required."}
            write_group(args.out, {"evidence.json": result}, "Reviewed pronunciation hypothesis comparison.",
                        [" ".join(sys.argv), f"audio sha256={hashlib.sha256(args.audio.read_bytes()).hexdigest()}"])
            print(json.dumps(result, ensure_ascii=False))
    except (OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError):
        LOG.exception("Experiment failed; no substitute result was produced")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
