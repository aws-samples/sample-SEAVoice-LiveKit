"""LiveKit STT plugin wrapping Typhoon ASR (FastConformer-Transducer for Thai)."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from functools import partial

import numpy as np

# NeMo references np.sctypes, which NumPy 2.0 no longer provides; add a shim.
if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [bool, object, bytes, str, np.void],
    }

import soundfile as sf
from livekit.agents import stt, utils
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

logger = logging.getLogger(__name__)


class TyphoonSTT(stt.STT):
    def __init__(
        self,
        *,
        model_name: str = "scb10x/typhoon-asr-realtime",
        device: str = "auto",
        noise_reduce: bool = True,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )
        self._model_name = model_name
        self._device = device
        self._model = None
        self._noise_reduce = noise_reduce

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "typhoon"

    def _ensure_model(self):
        if self._model is not None:
            return
        import nemo.collections.asr as nemo_asr
        import torch

        device = self._device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading Typhoon ASR model on {device}")
        self._model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=self._model_name,
            map_location=device,
        )

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        self._ensure_model()

        frames = utils.merge_frames(buffer)
        audio_data = (
            np.frombuffer(frames.data, dtype=np.int16).astype(np.float32) / 32768.0
        )

        if frames.sample_rate != 16000:
            import librosa

            audio_data = librosa.resample(
                audio_data, orig_sr=frames.sample_rate, target_sr=16000
            )

        if self._noise_reduce:
            import noisereduce as nr

            audio_data = nr.reduce_noise(
                y=audio_data, sr=16000, stationary=True, prop_decrease=0.75
            )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            sf.write(f.name, audio_data, 16000)
            transcriptions = await asyncio.get_event_loop().run_in_executor(
                None, partial(self._model.transcribe, audio=[f.name])
            )

        result = transcriptions[0] if transcriptions else ""
        text = (result[0] if result else "") if isinstance(result, list) else result

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    language="th",
                    text=text,
                    confidence=1.0,
                )
            ],
        )
