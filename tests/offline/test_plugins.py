"""Offline tests for the plugins (no network / no live AWS).

Persistent Bedrock session, SageMaker STT/TTS request+response shaping, and the
telco tool handlers / scenario switching.
"""

import base64
import json
from pathlib import Path

import numpy as np

from plugins import telco_tools
from plugins.bedrock_session import (
    _KEEPALIVE_TIMEOUT_S,
    PersistentSession,
    _PersistentClientCM,
)
from plugins.omnivoice_sagemaker_tts import OmniVoiceSageMakerTTS
from plugins.qwen_sagemaker_stt import QwenSageMakerSTT

REPO_ROOT = Path(__file__).resolve().parents[2]


# --- bedrock_session: persistent, keep-alive client ---


class _FakeRealCM:
    """Stand-in for the real aioboto3 client context manager."""

    def __init__(self) -> None:
        self.enters = 0
        self.exits = 0
        self.client = object()

    async def __aenter__(self):
        self.enters += 1
        return self.client

    async def __aexit__(self, *exc):
        self.exits += 1
        return False


async def test_client_opened_once_and_reused_across_turns():
    real = _FakeRealCM()
    cm = _PersistentClientCM(lambda: real)
    async with cm as c1:
        pass
    async with cm as c2:
        pass
    async with cm as c3:
        pass
    assert real.enters == 1, "client must be opened exactly once, not per turn"
    assert real.exits == 0, "per-turn exit must be a no-op (pool kept alive)"
    assert c1 is c2 is c3, "the same cached client is returned every turn"


async def test_aexit_never_suppresses_exceptions():
    cm = _PersistentClientCM(_FakeRealCM)
    raised = False
    try:
        async with cm:
            raise ValueError("boom")
    except ValueError:
        raised = True
    assert raised, "__aexit__ must not swallow the body's exception"


async def test_aclose_performs_deferred_real_exit_and_is_idempotent():
    real = _FakeRealCM()
    cm = _PersistentClientCM(lambda: real)
    async with cm:
        pass
    await cm.aclose()
    assert real.exits == 1, "aclose performs the deferred real __aexit__"
    await cm.aclose()  # idempotent
    assert real.exits == 1


class _FakeIncomingConfig:
    def __init__(self) -> None:
        self.user_agent_extra = "livekit-plugins-aws"
        self.connector_args = {"force_close": False}


def test_keepalive_injected_into_config():
    aio_cfg = PersistentSession(region_name="us-west-2")._with_keepalive(
        _FakeIncomingConfig()
    )
    assert aio_cfg.connector_args["keepalive_timeout"] == _KEEPALIVE_TIMEOUT_S
    assert aio_cfg.connector_args["force_close"] is False  # existing args preserved
    assert aio_cfg.user_agent_extra == "livekit-plugins-aws"


def test_keepalive_above_typical_turn_cadence():
    # Must exceed the ~18-21s inter-turn gap or the socket dies every turn anyway.
    assert _KEEPALIVE_TIMEOUT_S > 21.0


# --- SageMaker STT/TTS plugins (fake boto3 client) ---


class _FakeBody:
    def __init__(self, payload: dict) -> None:
        self._data = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._data


class _FakeClient:
    """Captures the last invoke_endpoint call and returns a canned response."""

    def __init__(self, response: dict) -> None:
        self.response = response
        self.last_call = None

    def invoke_endpoint(self, **kwargs):
        self.last_call = kwargs
        return {"Body": _FakeBody(self.response)}


def test_stt_shapes_chat_request_and_parses_transcript():
    stt = QwenSageMakerSTT(endpoint_name="qwen3-asr", region="us-west-2", language="th")
    # The vLLM DLC returns an OpenAI chat/completions response; Qwen3-ASR prefixes the
    # detected language before <asr_text>, which the plugin must strip.
    stt._client = _FakeClient(
        {"choices": [{"message": {"content": "language Thai<asr_text>สวัสดีค่ะ"}}]}
    )

    text = stt._transcribe((np.zeros(320, dtype=np.int16)).tobytes())
    assert text == "สวัสดีค่ะ"

    body = json.loads(stt._client.last_call["Body"])
    assert body["model"] == "Qwen/Qwen3-ASR-0.6B"
    content = body["messages"][0]["content"][0]
    assert content["type"] == "audio_url"
    assert content["audio_url"]["url"].startswith("data:audio/wav;base64,")
    assert stt._client.last_call["EndpointName"] == "qwen3-asr"


def test_stt_provider_and_model():
    stt = QwenSageMakerSTT(endpoint_name="qwen3-asr")
    assert stt.provider == "qwen_sagemaker"
    assert stt.model == "sagemaker:qwen3-asr"


def test_tts_constructs_and_parses_response():
    tts = OmniVoiceSageMakerTTS(
        endpoint_name="omnivoice-tts", region="us-west-2", language="English"
    )
    samples = np.array([0, 16384, -16384, 32767], dtype=np.int16)
    tts._client = _FakeClient(
        {
            "audio_base64": base64.b64encode(samples.tobytes()).decode(),
            "sample_rate": 24000,
        }
    )

    audio = tts._generate_audio("hello")
    assert audio.dtype == np.float32
    assert len(audio) == 4
    assert audio.min() >= -1.0 and audio.max() <= 1.0

    body = json.loads(tts._client.last_call["Body"])
    assert body["text"] == "hello"
    assert body["language"] == "English"


def test_tts_provider_and_model():
    tts = OmniVoiceSageMakerTTS(endpoint_name="omnivoice-tts")
    assert tts.provider == "omnivoice_sagemaker"
    assert tts.model == "sagemaker:omnivoice-tts"


def test_tts_only_applies_thai_processing_to_thai():
    from plugins.thai_tts_phrase_tokenizer import ThaiTTSPhraseTokenizer

    # Non-Thai must NOT get Thai processing (it garbles text, e.g. spelling English
    # out letter by letter).
    th = OmniVoiceSageMakerTTS(endpoint_name="e", language="Thai")
    en = OmniVoiceSageMakerTTS(endpoint_name="e", language="English")
    assert isinstance(th._sentence_tokenizer, ThaiTTSPhraseTokenizer)
    assert not isinstance(en._sentence_tokenizer, ThaiTTSPhraseTokenizer)
    assert en._normalize("Hello, how are you?") == "Hello, how are you?"


# --- telco tools + scenario switching ---


def test_set_scenario_path_switches_fixtures():
    telco_tools.set_scenario_path(REPO_ROOT / "scenarios" / "telco_intl.json")
    try:
        # The intl fixtures use Latin names, not Thai.
        assert telco_tools.load_fixtures()["customers"]["2749"]["name"] == "Somying Jaidee"
    finally:
        telco_tools.set_scenario_path(REPO_ROOT / "scenarios" / "telco_th.json")


def test_open_ticket_returns_summary_key():
    telco_tools.set_scenario_path(REPO_ROOT / "scenarios" / "telco_intl.json")
    try:
        result = telco_tools.open_ticket("2749", "billing", "Overcharged this month")
        assert result["summary"] == "Overcharged this month"
        assert result["status"] == "open"
        assert result["ticket_id"]
    finally:
        telco_tools.set_scenario_path(REPO_ROOT / "scenarios" / "telco_th.json")


def test_open_ticket_unknown_customer():
    telco_tools.set_scenario_path(REPO_ROOT / "scenarios" / "telco_intl.json")
    try:
        assert telco_tools.open_ticket("0000", "other", "x")["error"] == "customer_not_found"
    finally:
        telco_tools.set_scenario_path(REPO_ROOT / "scenarios" / "telco_th.json")
