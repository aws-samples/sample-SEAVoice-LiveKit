"""Offline tests for agent.py wiring (no network).

Backend selectors, prompt builder, scenario detection, LLM construction, session
options, and the connection prewarm.
"""

import pytest

import agent as agent_mod
from plugins.bedrock_session import PersistentSession

# --- backend selectors ---


def test_tts_backend_omnivoice_sagemaker():
    from plugins.omnivoice_sagemaker_tts import OmniVoiceSageMakerTTS

    tts = agent_mod._make_tts(
        {"tts": {"backend": "omnivoice_sagemaker", "endpoint_name": "omnivoice"}}
    )
    assert isinstance(tts, OmniVoiceSageMakerTTS)


def test_tts_unknown_backend_raises():
    with pytest.raises(ValueError, match=r"tts\.backend"):
        agent_mod._make_tts({"tts": {"backend": "nope"}})


def test_stt_backend_qwen_sagemaker():
    from plugins.qwen_sagemaker_stt import QwenSageMakerSTT

    stt = agent_mod._make_stt(
        {"stt": {"backend": "qwen_sagemaker", "endpoint_name": "qwen3-asr"}}
    )
    assert isinstance(stt, QwenSageMakerSTT)


def test_stt_unknown_backend_raises():
    with pytest.raises(ValueError, match=r"stt\.backend"):
        agent_mod._make_stt({"stt": {"backend": "nope"}})


# --- prompt builder + scenario detection ---


def test_build_system_prompt_joins_persona_and_voice_rules():
    prompt = agent_mod._build_system_prompt(
        {"prompt": {"persona": "You are helpful.", "voice_rules": "Speak plainly."}}
    )
    assert prompt == "You are helpful.\n\nSpeak plainly."


def test_build_system_prompt_persona_only():
    assert agent_mod._build_system_prompt({"prompt": {"persona": "Hi."}}) == "Hi."


def test_scenario_has_tools():
    assert agent_mod._scenario_has_tools({"scenario": "scenarios/telco_th.json"})
    assert agent_mod._scenario_has_tools({"scenario": "scenarios/telco_intl.json"})
    assert agent_mod._scenario_has_tools({}) is False


# --- LLM construction + session options ---


def test_persistent_connection_on_by_default():
    llm, session = agent_mod._make_llm({"llm": {"model": "m", "region": "us-west-2"}})
    assert isinstance(session, PersistentSession)
    # The plugin must actually use our session, not its own default.
    assert llm._session is session


def test_persistent_connection_can_be_disabled():
    _llm, session = agent_mod._make_llm(
        {"llm": {"model": "m", "region": "us-west-2", "persistent_connection": False}}
    )
    assert session is None


def test_reasoning_effort_still_applied_with_persistent_session():
    llm, _ = agent_mod._make_llm(
        {"llm": {"model": "m", "region": "us-west-2", "reasoning_effort": "low"}}
    )
    assert llm.chat.__name__ == "_patched_chat"


def test_session_options_preemptive_default_on():
    assert agent_mod._session_options({})["preemptive_generation"] is True


def test_session_options_preemptive_can_be_disabled():
    opts = agent_mod._session_options({"preemptive_generation": False})
    assert opts["preemptive_generation"] is False


# --- connection prewarm ---


class _FakeStream:
    def __init__(self) -> None:
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def aclose(self):
        self.closed = True


class _FakeLLM:
    def __init__(self, raise_on_chat=False) -> None:
        self.chat_calls = 0
        self.last_stream = None
        self._raise = raise_on_chat

    def chat(self, chat_ctx=None):
        self.chat_calls += 1
        if self._raise:
            raise RuntimeError("bedrock unreachable")
        self.last_stream = _FakeStream()
        return self.last_stream


async def test_prewarm_drives_stream_and_closes_it():
    llm = _FakeLLM()
    await agent_mod._cache_prewarm(llm, "system prompt")
    assert llm.chat_calls == 1
    assert llm.last_stream.closed is True


async def test_prewarm_swallows_errors():
    llm = _FakeLLM(raise_on_chat=True)
    # Best-effort: prewarm must never fail the session.
    await agent_mod._cache_prewarm(llm, "system prompt")
    assert llm.chat_calls == 1
