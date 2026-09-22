"""Config-driven Thai voice AI agent.

Usage:
    uv run python src/agent.py console --config configs/telco_th.yaml
    uv run python src/agent.py dev --config configs/chat_th.yaml
    uv run python src/agent.py console  # defaults to configs/telco_th.yaml
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
)
from livekit.agents.stt import StreamAdapter
from livekit.plugins import aws, silero

from plugins.fastthaig2p_tts import FastThaiG2PTTS
from plugins.metrics_logger import PipelineMetrics
from plugins.typhoon_stt import TyphoonSTT

logger = logging.getLogger("agent")

load_dotenv(".env.local")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "telco_th.yaml"


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate a YAML config file."""
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def _build_system_prompt(config: dict[str, Any]) -> str:
    """Assemble system prompt from persona and voice rules."""
    prompt_cfg = config.get("prompt", {})
    parts = []
    if persona := prompt_cfg.get("persona"):
        parts.append(persona.strip())
    if voice_rules := prompt_cfg.get("voice_rules"):
        parts.append(voice_rules.strip())
    return "\n\n".join(parts)


def _make_llm(config: dict[str, Any]):
    """Create LLM instance from config.

    Returns ``(llm, bedrock_session_or_None)``. When ``llm.persistent_connection``
    is enabled (default), the Bedrock client is built through a ``PersistentSession``
    that reuses one connection across turns; the caller MUST close it on shutdown
    via ``await session.aclose_all()``.
    """
    llm_cfg = config.get("llm", {})
    model = llm_cfg.get("model", "openai.gpt-oss-120b-1:0")
    region = llm_cfg.get("region", "us-west-2")
    reasoning_effort = llm_cfg.get("reasoning_effort")
    performance_config = llm_cfg.get("performance_config")

    bedrock_session = None
    if llm_cfg.get("persistent_connection", True):
        from plugins.bedrock_session import PersistentSession

        bedrock_session = PersistentSession(region_name=region)
        llm_instance = aws.LLM(model=model, region=region, session=bedrock_session)
    else:
        llm_instance = aws.LLM(model=model, region=region)

    if reasoning_effort or performance_config:
        _original_chat = llm_instance.chat

        def _patched_chat(**kwargs):
            stream = _original_chat(**kwargs)
            if reasoning_effort:
                stream._opts["additionalModelRequestFields"] = {
                    "reasoning_effort": reasoning_effort
                }
            if performance_config:
                stream._opts["performanceConfig"] = performance_config
            return stream

        llm_instance.chat = _patched_chat

    return llm_instance, bedrock_session


def _make_tts(config: dict[str, Any]):
    """Create the TTS backend selected by ``tts.backend``.

    Backends: ``fastthaig2p`` (local, default) and ``omnivoice_sagemaker``. The
    SageMaker backend is imported lazily so a local-only install never needs its
    dependencies.
    """
    tts_cfg = config.get("tts", {})
    backend = tts_cfg.get("backend", "fastthaig2p")

    if backend == "fastthaig2p":
        return FastThaiG2PTTS(
            speed=tts_cfg.get("speed", 1.0),
            intra_op_threads=tts_cfg.get("intra_op_threads", 0),
            model_path=tts_cfg.get("model_path"),
            voicepack_path=tts_cfg.get("voicepack_path"),
            config_path=tts_cfg.get("config_path"),
        )
    if backend == "omnivoice_sagemaker":
        from plugins.omnivoice_sagemaker_tts import OmniVoiceSageMakerTTS

        return OmniVoiceSageMakerTTS(
            endpoint_name=tts_cfg["endpoint_name"],
            region=tts_cfg.get("region", "us-west-2"),
            language=tts_cfg.get("language", "Thai"),
            num_step=tts_cfg.get("num_step", 16),
        )
    raise ValueError(f"Unknown tts.backend: {backend!r}")


def _make_stt(config: dict[str, Any]):
    """Create the STT backend selected by ``stt.backend``.

    Backends: ``typhoon`` (local, default) and ``qwen_sagemaker``.
    """
    stt_cfg = config.get("stt", {})
    backend = stt_cfg.get("backend", "typhoon")

    if backend == "typhoon":
        return TyphoonSTT(device=stt_cfg.get("device", "cpu"))
    if backend == "qwen_sagemaker":
        from plugins.qwen_sagemaker_stt import QwenSageMakerSTT

        return QwenSageMakerSTT(
            endpoint_name=stt_cfg["endpoint_name"],
            region=stt_cfg.get("region", "us-west-2"),
            language=stt_cfg.get("language", "en"),
        )
    raise ValueError(f"Unknown stt.backend: {backend!r}")


def _make_vad(config: dict[str, Any]):
    """Create VAD instance from config."""
    vad_cfg = config.get("vad", {})
    return silero.VAD.load(
        activation_threshold=vad_cfg.get("activation_threshold", 0.55),
        min_speech_duration=vad_cfg.get("min_speech_duration", 0.08),
        min_silence_duration=vad_cfg.get("min_silence_duration", 0.55),
    )


def _scenario_has_tools(config: dict[str, Any]) -> bool:
    """Return True if the config points at a scenario file that declares tools.

    The tool implementations live in the ``@function_tool`` methods of the agent
    class below; the scenario file's ``tools`` list only decides whether the
    tool-enabled agent (vs. the plain chat agent) is built.
    """
    scenario_path_str = config.get("scenario")
    if not scenario_path_str:
        return False

    scenario_path = REPO_ROOT / scenario_path_str
    with scenario_path.open("r", encoding="utf-8") as f:
        scenario = json.load(f)

    return bool(scenario.get("tools"))


def _session_options(config: dict[str, Any]) -> dict[str, Any]:
    """AgentSession kwargs derived from config.

    ``preemptive_generation`` (default on) runs the LLM during the endpointing
    silence wait so the first round-trip overlaps the wait; the turn detector still
    gates release of the reply.
    """
    return {
        "preemptive_generation": config.get("preemptive_generation", True),
    }


async def _cache_prewarm(llm_instance, system_prompt: str) -> None:
    """Fire one throwaway request through the pipeline's own LLM before turn 1.

    Warms the persistent Bedrock connection (TCP/TLS pool) so the first real turn
    is already warm instead of paying a cold handshake, and primes the prompt-cache
    prefix for backends that support it. Best-effort: never fails the session.
    """
    try:
        from livekit.agents import llm as _llm

        chat_ctx = _llm.ChatContext()
        chat_ctx.add_message(role="system", content=system_prompt)
        chat_ctx.add_message(role="user", content="hello")
        stream = llm_instance.chat(chat_ctx=chat_ctx)
        async for _ in stream:
            pass
        await stream.aclose()
        logger.info("cache_prewarm complete (persistent connection warmed)")
    except Exception as e:  # best-effort: never fail the session
        logger.warning("cache_prewarm skipped: %s", e)


def _create_agent_class(config: dict[str, Any], llm):
    """Dynamically create an Agent class with tools from config."""
    system_prompt = _build_system_prompt(config)

    if _scenario_has_tools(config):

        class ConfiguredAgent(Agent):
            def __init__(self) -> None:
                super().__init__(llm=llm, instructions=system_prompt)

            @function_tool
            async def check_balance(self, context: RunContext, customer_id: str):
                """Look up the customer's outstanding bill balance and due date.

                Args:
                    customer_id: The customer's 4-digit ID, e.g. '2749'.
                """
                from plugins.telco_tools import check_balance

                return check_balance(customer_id)

            @function_tool
            async def check_plan(self, context: RunContext, customer_id: str):
                """Look up the customer's current plan: monthly fee and quotas.

                Args:
                    customer_id: The customer's 4-digit ID, e.g. '2749'.
                """
                from plugins.telco_tools import check_plan

                return check_plan(customer_id)

            @function_tool
            async def check_usage(self, context: RunContext, customer_id: str):
                """Look up this month's usage (voice and data) and remaining quota.

                Args:
                    customer_id: The customer's 4-digit ID, e.g. '2749'.
                """
                from plugins.telco_tools import check_usage

                return check_usage(customer_id)

            @function_tool
            async def open_ticket(
                self,
                context: RunContext,
                customer_id: str,
                category: str,
                summary: str,
            ):
                """Open a support ticket for the customer's issue.

                Args:
                    customer_id: The customer's 4-digit ID, e.g. '2749'.
                    category: One of: billing, network, device, plan_change, other.
                    summary: A one-sentence summary of the issue, in the customer's language.
                """
                from plugins.telco_tools import open_ticket

                return open_ticket(customer_id, category, summary)

        return ConfiguredAgent
    else:

        class ChatAgent(Agent):
            def __init__(self) -> None:
                super().__init__(llm=llm, instructions=system_prompt)

        return ChatAgent


# --- Select the config ---
# We intercept `--config` before the LiveKit CLI parses argv. In `dev`/`start` mode
# the CLI spawns worker subprocesses that re-import this module with a fresh argv, so
# the choice is stashed in an env var (inherited by those subprocesses) rather than
# relying on argv alone — otherwise workers would silently fall back to the default.
for i, arg in enumerate(sys.argv):
    if arg == "--config" and i + 1 < len(sys.argv):
        os.environ["SEAVOICE_CONFIG"] = sys.argv[i + 1]
        sys.argv.pop(i)
        sys.argv.pop(i)
        break

_config_env = os.environ.get("SEAVOICE_CONFIG")
_config_path = Path(_config_env) if _config_env else DEFAULT_CONFIG
if not _config_path.is_absolute():
    _config_path = REPO_ROOT / _config_path

CONFIG = load_config(_config_path)
logger.info("Loaded config: %s", _config_path.name)

LLM_INSTANCE, BEDROCK_SESSION = _make_llm(CONFIG)
AgentClass = _create_agent_class(CONFIG, LLM_INSTANCE)
SYSTEM_PROMPT = _build_system_prompt(CONFIG)

# Point the telco tool fixtures at this config's scenario file.
if _scenario := CONFIG.get("scenario"):
    from plugins.telco_tools import set_scenario_path

    set_scenario_path(REPO_ROOT / _scenario)

server = AgentServer()

# Keep strong references to fire-and-forget background tasks (e.g. prewarm) so the
# event loop doesn't garbage-collect them before they complete.
_BACKGROUND_TASKS: set[asyncio.Task] = set()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = _make_vad(CONFIG)
    proc.userdata["stt"] = _make_stt(CONFIG)
    proc.userdata["tts"] = _make_tts(CONFIG)


server.setup_fnc = prewarm


@server.rtc_session(agent_name=CONFIG.get("agent_name", "my-agent"))
async def voice_agent(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    vad = ctx.proc.userdata["vad"]
    stt = ctx.proc.userdata["stt"]
    tts = ctx.proc.userdata["tts"]

    session = AgentSession(
        stt=StreamAdapter(stt=stt, vad=vad),
        tts=tts,
        vad=vad,
        **_session_options(CONFIG),
    )

    # Close the persistent Bedrock client cleanly when the job ends.
    if BEDROCK_SESSION is not None:
        ctx.add_shutdown_callback(BEDROCK_SESSION.aclose_all)

    metrics = PipelineMetrics(room=ctx.room)
    session.on(
        "metrics_collected", lambda event: metrics.on_metrics_collected(event.metrics)
    )
    session.on("agent_state_changed", metrics.on_agent_state_changed)

    # Log raw LLM output for debugging
    session.on(
        "conversation_item_added",
        lambda event: logger.info(
            "LLM  | role=%s text=%r", event.item.role, event.item.text_content
        ),
    )

    await session.start(agent=AgentClass(), room=ctx.room)
    await ctx.connect()

    # Warm the persistent LLM connection + prompt cache while the room settles,
    # so the first user turn doesn't pay a cold handshake. Fire-and-forget, but
    # keep a reference so the task isn't garbage-collected before it finishes.
    if CONFIG.get("llm", {}).get("prewarm", True):
        task = asyncio.create_task(_cache_prewarm(LLM_INSTANCE, SYSTEM_PROMPT))
        _BACKGROUND_TASKS.add(task)
        task.add_done_callback(_BACKGROUND_TASKS.discard)


if __name__ == "__main__":
    cli.run_app(server)
