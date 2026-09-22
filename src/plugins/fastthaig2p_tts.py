"""LiveKit TTS plugin wrapping FastThaiG2P (Kokoro-82M, Thai-finetuned).

Runs entirely on local CPU via ONNX — no remote endpoint needed.
Normalization is handled internally by FastThaiG2P's G2P pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncIterable
from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar

import numpy as np
from livekit.agents import tts, utils
from livekit.agents.tts import AudioEmitter, SynthesizedAudio, SynthesizeStream
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils import shortuuid

from plugins.thai_tts_phrase_tokenizer import ThaiTTSPhraseTokenizer

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000

_PUNCTUATION_RE = re.compile(r"[^\w\s฀-๿]", re.UNICODE)


def _strip_punctuation(text: str) -> str:
    """Remove punctuation that confuses TTS (quotes, parentheses, etc.)."""
    return _PUNCTUATION_RE.sub("", text)


class FastThaiG2PTTS(tts.TTS):
    def __init__(
        self,
        *,
        model_path: str | None = None,
        voicepack_path: str | None = None,
        config_path: str | None = None,
        speed: float = 1.0,
        intra_op_threads: int = 0,
        min_phrase_len: int = 10,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
        )
        self._model_path = model_path
        self._voicepack_path = voicepack_path
        self._config_path = config_path
        self._speed = speed
        self._intra_op_threads = intra_op_threads
        self._tts_engine = None
        self._sentence_tokenizer = ThaiTTSPhraseTokenizer(min_phrase_len=min_phrase_len)

    @property
    def model(self) -> str:
        return "fastthaig2p:kokoro-82m-thai"

    @property
    def provider(self) -> str:
        return "fastthaig2p"

    def _ensure_model(self):
        if self._tts_engine is not None:
            return
        from fastthaig2p import TTS as _TTS

        kwargs = {"speed": self._speed, "intra_op_threads": self._intra_op_threads}
        if self._model_path:
            kwargs["model_path"] = self._model_path
        if self._voicepack_path:
            kwargs["voicepack_path"] = self._voicepack_path
        if self._config_path:
            kwargs["config_path"] = self._config_path

        self._tts_engine = _TTS(**kwargs)
        logger.info("FastThaiG2P TTS engine loaded (ONNX/CPU)")

    def _generate_audio(self, text: str) -> np.ndarray:
        """Synchronous audio generation for a single text chunk."""
        text = _strip_punctuation(text)
        if not text.strip():
            return np.array([], dtype=np.float32)
        try:
            audio = self._tts_engine.generate(text)
        except ValueError as e:
            logger.warning("TTS generation failed for chunk (likely too long): %s", e)
            return np.array([], dtype=np.float32)
        return audio

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return _FastThaiG2PChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        return _FastThaiG2PPrefetchStream(
            tts=self,
            conn_options=conn_options,
        )


class _FastThaiG2PChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: FastThaiG2PTTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._fg2p_tts = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        self._fg2p_tts._ensure_model()

        request_id = shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/pcm",
            stream=False,
        )

        audio_np = await asyncio.get_event_loop().run_in_executor(
            None, self._fg2p_tts._generate_audio, self._input_text
        )

        if audio_np.size > 0:
            pcm_bytes = (audio_np * 32767).astype(np.int16).tobytes()
            output_emitter.push(pcm_bytes)


class _FastThaiG2PPrefetchStream(SynthesizeStream):
    """Streaming TTS with phrase-level prefetch.

    LLM tokens are buffered into Thai phrases via ThaiTTSPhraseTokenizer.
    Each phrase is synthesized in a thread pool with 1-ahead prefetch to
    minimize inter-phrase gaps.
    """

    _tts_request_span_name: ClassVar[str] = "fastthaig2p_prefetch_stream"

    def __init__(self, *, tts: FastThaiG2PTTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._fg2p_tts = tts

    async def _metrics_monitor_task(
        self, event_aiter: AsyncIterable[SynthesizedAudio]
    ) -> None:
        async for _ in event_aiter:
            pass

    async def _run(self, output_emitter: AudioEmitter) -> None:
        self._fg2p_tts._ensure_model()
        sent_stream = self._fg2p_tts._sentence_tokenizer.stream()

        request_id = shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=SAMPLE_RATE,
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
                text = ev.token.strip()
                if not text:
                    continue
                audio_np = await loop.run_in_executor(
                    executor, self._fg2p_tts._generate_audio, text
                )
                if audio_np.size == 0:
                    continue
                chunk_count += 1
                if chunk_count == 1:
                    logger.info(
                        "TTS  | chunk=1 ttfb=%.3fs audio=%.3fs",
                        time.perf_counter() - pipeline_start,
                        len(audio_np) / SAMPLE_RATE,
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
