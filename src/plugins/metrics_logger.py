"""Pipeline metrics logger for voice AI latency monitoring.

Tracks per-turn timing for:
- STT (complete): time to transcribe user speech
- LLM TTFT: time to first token from LLM
- LLM complete: total LLM generation time
- TTS TTFB: time to first byte of synthesized audio
- TTS complete: total TTS synthesis time
- TTFA: user stops speaking → first audio playback begins

Publishes metrics to the LiveKit room via data messages so the web UI can display them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from livekit import rtc
from livekit.agents.metrics.base import AgentMetrics, LLMMetrics, STTMetrics, TTSMetrics
from livekit.agents.voice.events import AgentStateChangedEvent

logger = logging.getLogger("pipeline.metrics")


class PipelineMetrics:
    def __init__(self, room: rtc.Room | None = None) -> None:
        self._room = room
        self._turn_start: float | None = None
        self._stt_duration: float | None = None
        self._llm_ttft: float | None = None
        self._llm_done: float | None = None
        self._first_audio: float | None = None
        self._turn_count: int = 0

    def _publish(self, data: dict) -> None:
        """Publish metrics to the room via data channel."""
        if self._room is None or not self._room.isconnected():
            return
        payload = json.dumps({"type": "metrics", **data}).encode()
        _task = asyncio.ensure_future(  # noqa: RUF006
            self._room.local_participant.publish_data(payload, reliable=True)
        )

    def on_agent_state_changed(self, event: AgentStateChangedEvent) -> None:
        if event.new_state == "thinking" and event.old_state == "listening":
            self._turn_start = time.perf_counter()
            # Note: STT has already finished by the time we enter "thinking", so the
            # STT duration is captured from its metrics event (which may fire just
            # before or after this) — don't reset it here.
            self._llm_ttft = None
            self._llm_done = None
            self._first_audio = None
        elif event.new_state == "speaking" and self._first_audio is None:
            self._first_audio = time.perf_counter()
            self._turn_count += 1
            self._log_ttfa()
        elif event.old_state == "speaking" and self._llm_done is not None:
            tts_total = time.perf_counter() - self._llm_done
            logger.info("TTS  | total=%.3fs", tts_total)
            self._publish({"tts_total": round(tts_total * 1000)})

    def on_metrics_collected(self, metrics: AgentMetrics) -> None:
        if isinstance(metrics, STTMetrics):
            self._on_stt_metrics(metrics)
        elif isinstance(metrics, LLMMetrics):
            self._on_llm_metrics(metrics)
        elif isinstance(metrics, TTSMetrics):
            self._on_tts_metrics(metrics)

    def _on_stt_metrics(self, metrics: STTMetrics) -> None:
        # metrics.duration is the STT processing time, reported the same way for the
        # local (Typhoon) and SageMaker (Qwen3-ASR) plugins.
        self._stt_duration = metrics.duration
        logger.info(
            "STT  | duration=%.3fs audio=%.3fs",
            metrics.duration,
            metrics.audio_duration,
        )
        self._publish({"stt": round(metrics.duration * 1000)})

    def _on_llm_metrics(self, metrics: LLMMetrics) -> None:
        now = time.perf_counter()
        self._llm_ttft = metrics.ttft
        self._llm_done = now
        tps = metrics.tokens_per_second
        logger.info(
            "LLM  | ttft=%.3fs duration=%.3fs tokens=%d tps=%.1f",
            metrics.ttft,
            metrics.duration,
            metrics.completion_tokens,
            tps,
        )
        self._publish(
            {
                "llm_ttft": round(metrics.ttft * 1000),
                "llm_total": round(metrics.duration * 1000),
            }
        )

    def _on_tts_metrics(self, metrics: TTSMetrics) -> None:
        logger.info(
            "TTS  | ttfb=%.3fs duration=%.3fs audio=%.3fs chars=%d",
            metrics.ttfb,
            metrics.duration,
            metrics.audio_duration,
            metrics.characters_count,
        )
        self._publish(
            {
                "tts_ttfb": round(metrics.ttfb * 1000),
                "tts_total": round(metrics.duration * 1000),
            }
        )

    def _log_ttfa(self) -> None:
        if self._turn_start is None or self._first_audio is None:
            return

        # `turn_start` is the listening -> thinking transition, i.e. after STT has
        # finished, so `response` covers only the LLM + TTS legs. STT ran before it,
        # timed separately by the STT plugin. Perceived latency (user stops speaking
        # -> agent starts) is roughly STT + response; it excludes the VAD/endpointing
        # silence wait, which happens before `turn_start`.
        response = self._first_audio - self._turn_start
        stt = self._stt_duration or 0.0
        tts_ttfb = (self._first_audio - self._llm_done) if self._llm_done else 0.0
        perceived = stt + response

        logger.info(
            "TTFA | turn=%d %.3fs = stt %.3fs + response %.3fs "
            "(llm_ttft %.3fs, tts_ttfb %.3fs)",
            self._turn_count,
            perceived,
            stt,
            response,
            self._llm_ttft or 0.0,
            tts_ttfb,
        )
        self._publish(
            {
                "tts_ttfb": round(tts_ttfb * 1000),
                "ttfa": round(perceived * 1000),
            }
        )
