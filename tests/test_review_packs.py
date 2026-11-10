from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from hindi_detector.review_server import read_items


class PackIsolation(unittest.TestCase):
    def prepare(self, root: Path, allow_external=True):
        pack, corpus = root / "pack", root / "corpus"
        pack.mkdir(); corpus.mkdir()
        audio = corpus / "native.wav"
        sf.write(audio, np.ones(16000) * 0.01, 16000)
        checksum = hashlib.sha256(audio.read_bytes()).hexdigest()
        selection = {"jobs": [{"id": "dev1", "text": "आज", "speaker_id": "speaker1",
                                "reference_sha256": checksum, "split": "train"}],
                     "clips": [{"clip_id": "native1", "sentence_id": "dev1", "text": "आज",
                                "audio": str(audio), "sha256": checksum, "speaker_id": "speaker1",
                                "system": "native_recording", "split": "train"}]}
        (pack / "selection.json").write_text(json.dumps(selection))
        if allow_external:
            (pack / "review_access.json").write_text(json.dumps({"schema_version": 1, "external_audio_roots": [str(corpus)]}))
        return pack, audio, checksum

    def test_declared_external_native_corpus_is_read_only_and_blinded(self):
        with tempfile.TemporaryDirectory() as directory:
            pack, _, _ = self.prepare(Path(directory))
            items, public = read_items(pack)
            self.assertEqual(items["native1"]["split"], "train")
            self.assertNotIn("system", public[0])

    def test_undeclared_external_audio_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            pack, _, _ = self.prepare(Path(directory), allow_external=False)
            with self.assertRaises(ValueError):
                read_items(pack)

    def test_generated_development_clip_keeps_its_training_split(self):
        with tempfile.TemporaryDirectory() as directory:
            pack, audio, checksum = self.prepare(Path(directory))
            output = pack / "indicf5"; output.mkdir()
            row = {"id": "dev1", "text": "आज", "speaker_id": "speaker1", "reference_sha256": checksum,
                   "audio": str(audio), "sha256": checksum, "status": "ok_unreviewed"}
            (output / "items.jsonl").write_text(json.dumps(row) + "\n")
            items, public = read_items(pack)
            generated = [r for r in items.values() if r["system"] == "indicf5_stock"]
            self.assertEqual(generated[0]["split"], "train")
            self.assertEqual(len(public), 2)

    def test_generated_receipt_must_match_selected_voice(self):
        with tempfile.TemporaryDirectory() as directory:
            pack, audio, checksum = self.prepare(Path(directory))
            output = pack / "indicf5"; output.mkdir()
            row = {"id": "dev1", "text": "आज", "speaker_id": "another_speaker", "reference_sha256": checksum,
                   "audio": str(audio), "sha256": checksum, "status": "ok_unreviewed"}
            (output / "items.jsonl").write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                read_items(pack)


if __name__ == "__main__":
    unittest.main()
