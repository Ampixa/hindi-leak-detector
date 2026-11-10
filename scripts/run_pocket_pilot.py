"""Small matched Hindi-parent/archive-finetune listening pilot on k2 CPU."""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from huggingface_hub import hf_hub_download
from pocket_tts import TTSModel

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "models/pocket_hindi_parent"
OUT = ROOT / "reports/pocket_pilot_v1"
REPO = "saryps-labs/pocket-tts-hindi"
REVISION = "6bff9c1e776bb90c68dd05c2467c62d1ecdf1b51"
FINETUNE = Path("/Users/k2/madhyam/runs/nepali_v1c/model.safetensors")
REFERENCE = Path("/Users/k2/madhyam/madhyam-tts/dashboard/public/audio/pocket/g00005.ref.m4a")
TEXTS = [
    {"id": "historical", "text": "सबैको सुरक्षा एनएलजीको इच्छा एनएलजी", "source": "existing internal g00005 paired-probe page"},
    {"id": "greeting", "text": "नमस्कार म तपाईंलाई कसरी सहयोग गर्न सक्छु", "source": "existing internal Audio8 greeting, punctuation omitted for Hindi model"},
    {"id": "schwa_draft", "text": "आज हामी नेपालको अर्थ र विकासका विषयमा कुरा गर्छौं", "source": "Codex-authored unreviewed pilot text; no gold pronunciations"},
]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"pilot already exists; inspect rather than overwrite: {OUT}")
    FINETUNE.resolve(strict=True)
    REFERENCE.resolve(strict=True)
    OUT.mkdir(parents=True)
    (OUT / "README.md").write_text(f"""# Pocket Hindi transfer listening pilot

Created by Codex (GPT-6), session /root, on {socket.gethostname()}, {datetime.now(timezone.utc).isoformat()}.
Owner: Ampixa/user. Origin: {OUT}. Status: active. Next review: 2026-11-13.
Purpose: matched inference examples to review Hindi-to-Nepali transfer, not a training run.
Current consumer: Hindi leakage lab. No native labels or accuracy claim.

Inventory: reference.wav is an explicitly converted copy of {REFERENCE};
parent.yaml and nepali_v1c.yaml are local-path inference configs; each
<system>.<text_id>.s<seed>.wav is generated mono 24k audio for the three texts and
seeds 7/29, twelve files total; manifest.json records all parameters and results.
inventory.json holds every retained member's size and SHA-256. Partial runs are
recorded by the files and traceback in the project run log; never treat as complete.

Existing fine-tune: {FINETUNE}, exported at step 1309 per its progress/train log.
Historical parent byte hash is unavailable: the current pinned Hindi parent may
not be byte-identical to the historical training initialization. This comparison
cannot by itself establish a causal fine-tuning effect. One voice is a smoke
test only, not speaker-held-out validation. Reference consent/rights are inherited
from the internal source corpus and were not established anew by this session;
keep these outputs internal. No weights or audio are uploaded to HF.

Run: PYTHONPATH=/Volumes/T9/hindi-detector /Users/k2/madhyam/pocket-tts/.venv/bin/python
/Volumes/T9/hindi-detector/scripts/run_pocket_pilot.py. Source code is in this
project; exact Pocket source revision and model hashes are in manifest.json.
Keep listening audio only while actively reviewed; delete within 24h after the
review finishes unless a dated current consumer is recorded. Keep compact
manifest/conclusions. Known duplicates: original reference source above; cdjk
may receive the twelve small listening outputs for the internal review page.
""")
    PARENT.mkdir(parents=True, exist_ok=True)
    parent_files = []
    for name in ["config.yaml", "model.safetensors", "tokenizer.model", "README.md", "NOTICE.md"]:
        path = Path(hf_hub_download(REPO, name, revision=REVISION, local_dir=PARENT, token=False))
        parent_files.append({"path": name, "size_bytes": path.stat().st_size, "sha256": sha(path)})
    (PARENT / "inventory.json").write_text(json.dumps(parent_files, indent=2) + "\n")
    config = yaml.safe_load((PARENT / "config.yaml").read_text())
    if config["flow_lm"]["transformer"]["num_layers"] != 6 or config["mimi"]["sample_rate"] != 24000:
        raise ValueError("unexpected parent structure")
    config["flow_lm"]["lookup_table"]["tokenizer_path"] = str(PARENT / "tokenizer.model")
    torch.set_num_threads(2)
    ref = OUT / "reference.wav"
    subprocess.run(["/usr/bin/afconvert", str(REFERENCE), "-f", "WAVE", "-d", "LEI16@24000", "-c", "1", str(ref)], check=True)
    source = Path("/Users/k2/madhyam/pocket-tts")
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    manifest = {"schema_version": 1, "status": "running", "texts": TEXTS, "seeds": [7, 29],
                "parent": {"repo": REPO, "revision": REVISION, "sha256": sha(PARENT / "model.safetensors")},
                "finetune": {"path": str(FINETUNE), "sha256": sha(FINETUNE), "reported_step": 1309},
                "historical_parent_hash_verified": False, "pocket_source_revision": revision,
                "reference_original": str(REFERENCE), "reference_original_sha256": sha(REFERENCE),
                "reference_wav_sha256": sha(ref), "device": "cpu", "threads": 2,
                "temperature": 0.3, "sampler_decode_steps": 1, "eos_threshold": -4.0,
                "versions": {"torch": torch.__version__, "numpy": np.__version__}, "outputs": []}
    for name, weights in [("parent", PARENT / "model.safetensors"), ("nepali_v1c", FINETUNE)]:
        local = copy.deepcopy(config)
        local["weights_path"] = str(weights)
        local["weights_path_without_voice_cloning"] = str(weights)
        cfg = OUT / f"{name}.yaml"
        cfg.write_text(yaml.safe_dump(local))
        model = TTSModel.load_model(config=cfg, temp=0.3, sampler_decode_steps=1, eos_threshold=-4.0)
        state = model.get_state_for_audio_prompt(ref)
        for text in TEXTS:
            for seed in [7, 29]:
                torch.manual_seed(seed)
                start = time.monotonic()
                # Pocket decodes on worker threads. Inference-mode tensors
                # cannot be mutated in those threads; ordinary no_grad is safe.
                with torch.no_grad():
                    wave = model.generate_audio(state, text["text"], copy_state=True).detach().cpu().numpy().reshape(-1)
                if not np.isfinite(wave).all() or len(wave) == 0:
                    raise ValueError(f"invalid generated audio: {name} {text['id']} seed {seed}")
                path = OUT / f"{name}.{text['id']}.s{seed}.wav"
                sf.write(path, wave, model.sample_rate, subtype="PCM_16")
                manifest["outputs"].append({"system": name, "text_id": text["id"], "seed": seed,
                    "path": path.name, "seconds": len(wave) / model.sample_rate,
                    "generation_wall_seconds": time.monotonic() - start, "sha256": sha(path)})
                (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
                logging.info("Generated %s", path.name)
        del model
    manifest["status"] = "complete_unreviewed_audio"
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    inventory = [{"path": p.name, "size_bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(OUT.iterdir()) if p.is_file()]
    (OUT / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        main()
    except Exception:
        logging.exception("Pocket transfer pilot failed; inspect partial outputs before rerunning")
        raise
