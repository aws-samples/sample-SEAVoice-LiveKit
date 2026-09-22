"""Synthesize benchmark audio files from manifest.json using SageMaker TTS.

Generates WAV files for each test utterance so the benchmark uses consistent
pre-recorded audio rather than live mic input.

Usage:
    uv run python scripts/bench_synth.py
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import boto3
import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "tests" / "fixtures" / "bench" / "manifest.json"
OUTPUT_DIR = REPO_ROOT / "tests" / "fixtures" / "bench"

ENDPOINT_NAME = "omnivoice-tts"
REGION = "us-west-2"


def synthesize(client, text: str) -> tuple[np.ndarray, int]:
    """Call SageMaker endpoint to generate audio."""
    payload = json.dumps({"text": text, "language": "Thai", "num_step": 16})
    response = client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=payload,
    )
    result = json.loads(response["Body"].read())
    audio_bytes = base64.b64decode(result["audio_base64"])
    audio = np.frombuffer(audio_bytes, dtype=np.int16)
    return audio, result["sample_rate"]


def main() -> int:
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    print(f"[synth] {len(entries)} utterances to synthesize")

    client = boto3.client("sagemaker-runtime", region_name=REGION)

    for entry in entries:
        idx = entry["idx"]
        bucket = entry["bucket"]
        text = entry["text"]
        filename = f"{idx:02d}_{bucket}.wav"
        out_path = OUTPUT_DIR / filename

        if out_path.exists():
            print(f"  [{idx:02d}] skip (exists): {filename}")
            entry["path"] = f"tests/fixtures/bench/{filename}"
            continue

        print(f"  [{idx:02d}] synthesizing: {text[:40]}...")
        audio, sr = synthesize(client, text)
        sf.write(str(out_path), audio, sr)
        duration_s = len(audio) / sr
        entry["path"] = f"tests/fixtures/bench/{filename}"
        entry["duration_s"] = round(duration_s, 3)
        print(f"       -> {filename} ({duration_s:.2f}s)")

    # Update manifest with paths and durations
    MANIFEST.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n[synth] Done! Manifest updated: {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
