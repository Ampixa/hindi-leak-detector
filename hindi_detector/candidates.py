"""Mine lexicon disagreements; these are review candidates, never gold labels."""
from __future__ import annotations

import csv
import importlib
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# Explicit normalization for SCHWA-ONLY candidate mining. These aliases erase
# length/affricate differences deliberately; they are NOT acoustic IPA labels.
HINDI_TO_PROJECT = {
    "a": "ax", "aa": "aa", "i": "i", "ii": "i", "u": "u", "uu": "u",
    "e": "e", "o": "o", "k": "k", "kh": "kh", "g": "g", "gh": "gh",
    "ng": "ng", "c": "ts", "ch": "tsh", "j": "dz", "jh": "dzh",
    "tt": "tx", "tth": "txh", "dd": "dx", "ddh": "dxh", "n": "n",
    "t": "t", "th": "th", "d": "d", "dh": "dh", "p": "p", "ph": "ph",
    "b": "b", "bh": "bh", "m": "m", "y": "y", "r": "r", "l": "l",
    "v": "w", "s": "s", "sh": "s", "h": "h", "z": "z", "f": "f",
}


def schwa_slots(phones: list[str]) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Represent each schwa by the gap between otherwise identical phones."""
    skeleton, counts = [], [0]
    for phone in phones:
        if phone == "ax":
            counts[-1] += 1
        else:
            skeleton.append(phone)
            counts.append(0)
    return tuple(skeleton), tuple(counts)


def mine(frontend_root: Path, hindi_csv: Path) -> tuple[list[dict], dict]:
    frontend_root = frontend_root.resolve(strict=True)
    sys.path.insert(0, str(frontend_root))
    try:
        ph = importlib.import_module("nepali_frontend.g2p.phonemizer")
    finally:
        sys.path.pop(0)
    if not Path(ph.__file__).resolve().is_relative_to(frontend_root):
        raise ValueError("a different frontend was already imported; use a fresh process")
    lexicon = ph.data.lexicon()
    variants: dict[str, set[tuple[str, ...]]] = {}
    counts = Counter()
    with hindi_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["hindi", "phon"]:
            raise ValueError(f"unexpected Hindi schema: {reader.fieldnames}")
        for row in reader:
            counts["hindi_rows"] += 1
            word = unicodedata.normalize("NFC", row["hindi"])
            if word not in lexicon:
                continue
            counts["overlap_rows"] += 1
            tokens = row["phon"].split()
            unknown = set(tokens) - HINDI_TO_PROJECT.keys()
            if unknown:
                counts["unsupported_hindi_rows"] += 1
                continue
            variants.setdefault(word, set()).add(tuple(HINDI_TO_PROJECT[p] for p in tokens))
    candidates = []
    for word, hindi_variants in sorted(variants.items()):
        target = ph.phonemize_word(word)
        phones = [p for p in target.phones if p != "."]
        skeleton, target_slots = schwa_slots(phones)
        source_variants = []
        for alternative in sorted(hindi_variants):
            other_skeleton, source_slots = schwa_slots(list(alternative))
            if other_skeleton == skeleton and source_slots != target_slots:
                source_variants.append((alternative, source_slots))
        for alternative, source_slots in source_variants:
            sites = [{"gap_index": i, "target_count": t, "hindi_count": h}
                     for i, (t, h) in enumerate(zip(target_slots, source_slots, strict=True)) if t != h]
            candidates.append({
                "candidate_id": f"schwa-{len(candidates) + 1:05d}", "language": "ne",
                "word": word, "profile": "spoken_nepali", "target_phones": phones,
                "hindi_schwa_hypothesis": list(alternative), "sites": sites,
                "target_source": target.source, "review_status": "pending",
                "note": "Normalized dictionary contrast; verify meaning, dialect, context and both pronunciations with native listeners.",
            })
    counts["candidate_pairs"] = len(candidates)
    counts["candidate_words"] = len({r["word"] for r in candidates})
    return candidates, dict(counts)
