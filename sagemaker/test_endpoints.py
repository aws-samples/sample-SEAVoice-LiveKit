"""Smoke-test the SageMaker speech endpoints (STT + TTS).

    uv run python sagemaker/test_endpoints.py

Requires the endpoints to be InService (see deploy_stt.py and deploy_tts.py).
"""

import base64
import json
import time
from pathlib import Path

import boto3

REGION = "us-west-2"
STT_ENDPOINT = "qwen3-asr"
STT_MODEL = "Qwen/Qwen3-ASR-0.6B"
TTS_ENDPOINT = "omnivoice-tts"
FIXTURE = Path(__file__).resolve().parent.parent / "tests/fixtures/bench/00_greet.wav"
_ASR_MARKER = "<asr_text>"


def test_stt(client) -> None:
    """Qwen3-ASR via the vLLM DLC's chat/completions (audio_url)."""
    if not FIXTURE.exists():
        print(f"[stt] skip — no fixture at {FIXTURE}")
        return
    audio_url = "data:audio/wav;base64," + base64.b64encode(FIXTURE.read_bytes()).decode()
    payload = json.dumps(
        {
            "model": STT_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "audio_url", "audio_url": {"url": audio_url}}],
                }
            ],
            "max_tokens": 256,
            "temperature": 0.0,
        }
    )
    t0 = time.perf_counter()
    resp = client.invoke_endpoint(
        EndpointName=STT_ENDPOINT,
        ContentType="application/json",
        Accept="application/json",
        Body=payload,
    )
    content = json.loads(resp["Body"].read())["choices"][0]["message"]["content"]
    text = content.split(_ASR_MARKER, 1)[1].strip() if _ASR_MARKER in content else content.strip()
    print(f"[stt] {(time.perf_counter() - t0) * 1000:.0f} ms -> {text!r}")


def test_tts(client) -> None:
    """OmniVoice TTS -> report latency and real-time factor."""
    payload = json.dumps({"text": "สวัสดีค่ะ ยินดีให้บริการ", "language": "Thai", "num_step": 16})
    t0 = time.perf_counter()
    resp = client.invoke_endpoint(
        EndpointName=TTS_ENDPOINT,
        ContentType="application/json",
        Accept="application/json",
        Body=payload,
    )
    elapsed = time.perf_counter() - t0
    result = json.loads(resp["Body"].read())
    dur = result.get("duration") or 0.0
    rtf = f"RTF {elapsed / dur:.2f}" if dur else ""
    print(f"[tts] {elapsed * 1000:.0f} ms -> {dur:.2f}s audio @ {result['sample_rate']}Hz {rtf}")


def main() -> None:
    client = boto3.client("sagemaker-runtime", region_name=REGION)
    test_stt(client)
    test_tts(client)


if __name__ == "__main__":
    main()
