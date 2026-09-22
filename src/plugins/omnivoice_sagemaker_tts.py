"""LiveKit TTS plugin that calls OmniVoice on a SageMaker endpoint.

Pairs with the deploy code in ``sagemaker/`` (OmniVoice handler). Selected via
``tts.backend: omnivoice_sagemaker`` in a config.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from collections.abc import AsyncIterable
from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar

import numpy as np
from fastthaig2p import normalize as normalize_for_thai_tts
from livekit.agents import tts, utils
from livekit.agents.tts import AudioEmitter, SynthesizedAudio, SynthesizeStream
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils import shortuuid

from plugins.thai_tts_phrase_tokenizer import ThaiTTSPhraseTokenizer

logger = logging.getLogger(__name__)

OMNIVOICE_SAMPLE_RATE = 24000


def _passthrough(text: str) -> str:
    return text


class OmniVoiceSageMakerTTS(tts.TTS):
    def __init__(
        self,
        *,
        endpoint_name: str = "omnivoice-tts",
        region: str = "us-west-2",
        language: str = "Thai",
        num_step: int = 16,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=OMNIVOICE_SAMPLE_RATE,
            num_channels=1,
        )
        self._endpoint_name = endpoint_name
        self._region = region
        self._language = language
        self._num_step = num_step
        self._client = None

        # Thai text needs the Thai normalizer + phrase tokenizer; other languages
        # must NOT get Thai processing (it garbles them, e.g. spelling English out
        # letter by letter), so use a generic sentence tokenizer and pass text through.
        if language.lower() in ("th", "thai"):
            self._normalize = normalize_for_thai_tts
            self._sentence_tokenizer = ThaiTTSPhraseTokenizer(min_phrase_len=10)
        else:
            from livekit.agents import tokenize

            self._normalize = _passthrough
            self._sentence_tokenizer = tokenize.basic.SentenceTokenizer()

    @property
    def model(self) -> str:
        return f"sagemaker:{self._endpoint_name}"

    @property
    def provider(self) -> str:
        return "omnivoice_sagemaker"

    def _ensure_client(self):
        if self._client is not None:
            return
        import boto3

        self._client = boto3.client("sagemaker-runtime", region_name=self._region)
        logger.info(f"SageMaker TTS client ready (endpoint={self._endpoint_name})")

    def _generate_audio(self, text: str) -> np.ndarray:
        """Call SageMaker endpoint to generate audio."""
        self._ensure_client()

        payload = json.dumps(
            {
                "text": text,
                "language": self._language,
                "num_step": self._num_step,
            }
        )

        response = self._client.invoke_endpoint(
            EndpointName=self._endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            Body=payload,
        )

        result = json.loads(response["Body"].read())
        audio_bytes = base64.b64decode(result["audio_base64"])
        return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return _SageMakerChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        return _SageMakerPrefetchStream(
            tts=self,
            conn_options=conn_options,
        )


class _SageMakerChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: OmniVoiceSageMakerTTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._sm_tts = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        self._sm_tts._ensure_client()

        request_id = shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=OMNIVOICE_SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/pcm",
            stream=False,
        )

        normalized_text = self._sm_tts._normalize(self._input_text)
        audio_np = await asyncio.get_event_loop().run_in_executor(
            None, self._sm_tts._generate_audio, normalized_text
        )

        pcm_bytes = (audio_np * 32767).astype(np.int16).tobytes()
        output_emitter.push(pcm_bytes)


class _SageMakerPrefetchStream(SynthesizeStream):
    """Streaming TTS with sentence-level prefetch via SageMaker endpoint."""

    _tts_request_span_name: ClassVar[str] = "sagemaker_prefetch_stream"

    def __init__(
        self, *, tts: OmniVoiceSageMakerTTS, conn_options: APIConnectOptions
    ) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._sm_tts = tts

    async def _metrics_monitor_task(
        self, event_aiter: AsyncIterable[SynthesizedAudio]
    ) -> None:
        async for _ in event_aiter:
            pass

    async def _run(self, output_emitter: AudioEmitter) -> None:
        self._sm_tts._ensure_client()
        sent_stream = self._sm_tts._sentence_tokenizer.stream()

        request_id = shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=OMNIVOICE_SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        segment_id = shortuuid()
        output_emitter.start_segment(segment_id=segment_id)

        async def _forward_input() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sent_stream.flush()
                    continue
                sent_stream.push_text(data)
            sent_stream.end_input()

        async def _synthesize_pipeline() -> None:
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts")

            pipeline_start = time.perf_counter()
            chunk_count = 0

            # Synthesize and emit each phrase as it streams in from the LLM, so the
            # first audio is produced right after the first phrase (not after the
            # whole reply). The LLM keeps generating later phrases on its own task
            # while a phrase synthesizes, so that overlap is free.
            async for ev in sent_stream:
                text = self._sm_tts._normalize(ev.token.strip())
                if not text:
                    continue
                audio_np = await loop.run_in_executor(
                    executor, self._sm_tts._generate_audio, text
                )
                if audio_np.size == 0:
                    continue
                chunk_count += 1
                if chunk_count == 1:
                    logger.info(
                        "TTS  | chunk=1 ttfb=%.3fs audio=%.3fs",
                        time.perf_counter() - pipeline_start,
                        len(audio_np) / OMNIVOICE_SAMPLE_RATE,
                    )
                pcm_bytes = (audio_np * 32767).astype(np.int16).tobytes()
                output_emitter.push(pcm_bytes)
                output_emitter.flush()
                await asyncio.sleep(0)

            logger.info(
                "TTS  | TOTAL chunks=%d %.3fs",
                chunk_count,
                time.perf_counter() - pipeline_start,
            )
            executor.shutdown(wait=False)

        tasks = [
            asyncio.create_task(_forward_input()),
            asyncio.create_task(_synthesize_pipeline()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await sent_stream.aclose()
            await utils.aio.cancel_and_wait(*tasks)
