"""LiveKit STT plugin that calls Qwen3-ASR on a SageMaker endpoint.

Targets the AWS vLLM SageMaker DLC (`vllm:server-sagemaker-cuda`) serving
Qwen3-ASR — see sagemaker/README.md. Selected via ``stt.backend: qwen_sagemaker``.
Non-streaming: each turn's audio is buffered, resampled to 16 kHz mono, wrapped as a
WAV, and sent to the endpoint's OpenAI-compatible `/invocations` (chat/completions
with an ``audio_url``). Qwen3-ASR returns ``language <Lang><asr_text><transcript>``;
we keep the part after ``<asr_text>``.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import wave
from functools import partial

import numpy as np
from livekit.agents import stt, utils
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

logger = logging.getLogger(__name__)

QWEN_ASR_SAMPLE_RATE = 16000
_ASR_TEXT_MARKER = "<asr_text>"


class QwenSageMakerSTT(stt.STT):
    def __init__(
        self,
        *,
        endpoint_name: str,
        region: str = "us-west-2",
        language: str = "en",
        model: str = "Qwen/Qwen3-ASR-0.6B",
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )
        self._endpoint_name = endpoint_name
        self._region = region
        self._language = language
        self._model = model
        self._client = None

    @property
    def model(self) -> str:
        return f"sagemaker:{self._endpoint_name}"

    @property
    def provider(self) -> str:
        return "qwen_sagemaker"

    def _ensure_client(self):
        if self._client is not None:
            return
        import boto3

        self._client = boto3.client("sagemaker-runtime", region_name=self._region)
        logger.info(f"SageMaker STT client ready (endpoint={self._endpoint_name})")

    @staticmethod
    def _pcm16_to_wav(pcm16: bytes, sample_rate: int) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm16)
        return buf.getvalue()

    def _transcribe(self, pcm16: bytes) -> str:
        """Send WAV audio to the vLLM endpoint's chat/completions, return transcript."""
        self._ensure_client()
        wav = self._pcm16_to_wav(pcm16, QWEN_ASR_SAMPLE_RATE)
        audio_url = "data:audio/wav;base64," + base64.b64encode(wav).decode()
        payload = json.dumps(
            {
                "model": self._model,
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
        response = self._client.invoke_endpoint(
            EndpointName=self._endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            Body=payload,
        )
        result = json.loads(response["Body"].read())
        content = result["choices"][0]["message"]["content"]
        # Qwen3-ASR prefixes the detected language: "language <Lang><asr_text><text>".
        if _ASR_TEXT_MARKER in content:
            content = content.split(_ASR_TEXT_MARKER, 1)[1]
        return content.strip()

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        frames = utils.merge_frames(buffer)
        audio_data = (
            np.frombuffer(frames.data, dtype=np.int16).astype(np.float32) / 32768.0
        )

        if frames.sample_rate != QWEN_ASR_SAMPLE_RATE:
            import librosa

            audio_data = librosa.resample(
                audio_data,
                orig_sr=frames.sample_rate,
                target_sr=QWEN_ASR_SAMPLE_RATE,
            )

        pcm16 = (audio_data * 32767).astype(np.int16).tobytes()
        text = await asyncio.get_event_loop().run_in_executor(
            None, partial(self._transcribe, pcm16)
        )

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    language=self._language,
                    text=text,
                    confidence=1.0,
                )
            ],
        )
