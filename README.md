# SEAVoice-LiveKit

A real-time, multilingual voice AI agent built on [LiveKit Agents](https://github.com/livekit-examples/agent-starter-python) with reusable **LiveKit STT/TTS plugins** you can drop into your own agent.

SEVoice LiveKit provides speech plugins for six regional languages: English, Thai, Vietnamese, Indonesian, Malay, and Filipino. Every language has voice interfaces available on Amazon SageMaker; this example uses vLLM-accelerated [Qwen3-ASR](https://huggingface.co/collections/Qwen/qwen3-asr) for speech-to-text (STT) and [OmniVoice](https://github.com/k2-fsa/OmniVoice) for text-to-speech (TTS). Thai also has local, CPU-based voice plugins using [Typhoon ASR](https://arxiv.org/abs/2601.13044) for STT and [FastThaiG2P/Kokoro-82M](https://github.com/awslabs/FastThaiG2P) for TTS. The LLM is provisioned through Amazon Bedrock; this example uses GPT-OSS 120B, but you can point it at [any supported Bedrock model](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html).

Our contribution is `src/plugins/` for local and AWS-based STT, LLM and TTS inferfaces. They slot into any LiveKit `AgentSession`. Everything else (configs, scenarios, the web UI) is here to demonstrate them. Two **example scenarios** ship for each language: a `telco` customer-support agent (with tools) and a free-form `chat` agent.

**Example scenario**: Mango Mobile, a fictional mobile carrier with customer account lookup, plan details, usage tracking, and ticket creation via voice.

> You run this on your own machine against your own AWS account. The LLM (Bedrock) and any SageMaker speech backends run in the cloud; the Thai local pipeline runs on your CPU.

## Architecture

```
Mic  →  [ STT ] →  [ LLM ] →  [ TTS ] → Speaker
       pluggable   Bedrock   pluggable
```

The LLM runs via Amazon Bedrock (cloud) and VAD is run locally, while STT and TTS can run either locally or on SageMaker (cloud), selected per config in YAML.

| Stage | Model(s) | Runs on |
|-------|----------|---------|
| **STT** | Typhoon ASR (Thai) · Qwen3-ASR (all languages) | Local CPU **or** SageMaker GPU |
| **LLM** | [Any Bedrock model](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html) | Amazon Bedrock |
| **TTS** | FastThaiG2P/Kokoro-82M (Thai) · OmniVoice (all languages) | Local CPU **or** SageMaker GPU |
| **VAD** | Silero | Local CPU |

### Latency optimizations

A voice agent is judged on time-to-first-audio, so the pipeline is tuned to overlap and shorten every leg:

- **vLLM-accelerated STT** Qwen3-ASR runs on AWS's vLLM SageMaker DLC (`vllm:server-sagemaker-cuda`), ~60ms server-side inference vs ~1s for plain `transformers`.
- **Persistent Bedrock connection** one bedrock-runtime client + TCP/TLS pool is reused across turns (keep-alive tuned above the inter-turn gap), avoiding a fresh handshake every turn; warm LLM TTFT drops from ~986ms to ~610ms p50. (`llm.persistent_connection`)
- **Preemptive generation** the LLM runs *during* the endpointing silence wait, so its first round-trip overlaps the wait instead of starting after it. (`preemptive_generation`)
- **Greeting-time prewarm** a throwaway LLM call while the room connects warms the connection and primes the prompt cache, so turn 1 isn't cold. (`llm.prewarm`)
- **Streaming per-phrase TTS** each phrase is synthesized and played as the LLM streams it (sentence tokenizer and prefetch), so first audio lands after the *first phrase*, not the whole reply.
- **Local Thai path** Typhoon ASR and FastThaiG2P run on CPU, removing the network round-trip entirely for Thai.

See [Performance](#performance) for measured numbers.

## Languages

Each language ships two example scenarios: `telco_*` (customer support, with tools) and `chat_*` (free-form assistant, no tools). These are demonstrations; swap in your own prompt, tools, and `agent_name`.

| Language | Telco config | Chat config | STT / TTS |
|----------|-------------|-------------|-----------|
| Thai (local) | `telco_th.yaml` | `chat_th.yaml` | Typhoon / FastThaiG2P |
| Thai (SageMaker) | `telco_th_sagemaker.yaml` | `chat_th_sagemaker.yaml` | Qwen3-ASR / OmniVoice |
| English | `telco_en.yaml` | `chat_en.yaml` | Qwen3-ASR / OmniVoice | 
| Vietnamese | `telco_vi.yaml` | `chat_vi.yaml` | Qwen3-ASR / OmniVoice |
| Indonesian | `telco_id.yaml` | `chat_id.yaml` | Qwen3-ASR / OmniVoice |
| Malay | `telco_ms.yaml` | `chat_ms.yaml` | Qwen3-ASR / OmniVoice |
| Filipino | `telco_fil.yaml` | `chat_fil.yaml` | Qwen3-ASR / OmniVoice |

## Prerequisites

- Python 3.11+ and the [uv](https://docs.astral.sh/uv/) package manager
- An AWS account with Amazon Bedrock access
- [LiveKit CLI](https://docs.livekit.io/intro/basics/cli/) (`lk`) + `livekit-server` for web app mode (`brew install livekit-cli livekit`)
- For non-Thai languages only: an Amazon SageMaker execution role and GPU quota to host the speech models. See [`sagemaker/README.md`](sagemaker/README.md)

## Quickstart

### 1. Install

```bash
git clone https://github.com/awslabs/sample-SEAVoice-LiveKit.git
cd sample-SEAVoice-LiveKit
uv sync

cp .env.example .env.local
# Edit .env.local:
# LIVEKIT_URL=ws://127.0.0.1:7880   
# LIVEKIT_API_KEY=devkey
# LIVEKIT_API_SECRET=secret
# AWS_ACCESS_KEY_ID=<your-key>   
# AWS_SECRET_ACCESS_KEY=<your-secret>   
# AWS_DEFAULT_REGION=<your-region>

uv run python src/agent.py download-files   # one-time: VAD + turn-detector models
```

### 2a. Sagemaker (All Languages)

Deploy the speech endpoints once (full guide in [`sagemaker/README.md`](sagemaker/README.md)):

```bash
export SAGEMAKER_ROLE=arn:aws:iam::<account>:role/<your-sagemaker-role>
uv run python sagemaker/deploy_stt.py   # STT for all languages (Qwen3-ASR)
uv run python sagemaker/deploy_tts.py   # TTS for all languages (OmniVoice)
```

Then run any language in your terminal (mic → agent → speakers):

```bash
uv run python src/agent.py console --config configs/telco_en.yaml   # English telco
uv run python src/agent.py console --config configs/telco_vi.yaml   # Vietnamese telco
uv run python src/agent.py console --config configs/chat_id.yaml    # Indonesian chat
```

### 2b. Local Voice Interfaces (Thai Only)

No endpoints needed; the Thai speech models run on your CPU:

```bash
uv run python src/agent.py console                                 # Thai telco (default)
uv run python src/agent.py console --config configs/chat_th.yaml   # Thai free chat
```

### 3. Web App Mode 

A browser UI with live transcripts and pipeline metrics over WebRTC. Pass the same `--config` to the agent and the web server.

```bash
pkill -f "livekit-server"; pkill -f "src/agent.py"; pkill -f "web/server.py"   # clean slate
livekit-server --dev                                             # terminal 1
uv run python src/agent.py dev --config configs/telco_en.yaml    # terminal 2
uv run python web/server.py  --config configs/telco_en.yaml      # terminal 3
open http://localhost:8080                                       # click the mic to connect
```

Web app mode requires `localhost` for mic permissions; for remote servers use SSH tunneling or console mode.

## Speech Backends

Each backend is a standard LiveKit `stt.STT` / `tts.TTS` plugin in `src/plugins/` — import it and pass it to your own `AgentSession`, no config system required. In this sample they're chosen per config via `stt.backend` and `tts.backend`.

**STT** (`stt.backend`)

| Backend | Model | Hosting |
|---------|-------|---------|
| `typhoon` (default) | Typhoon ASR (Thai) | Local CPU |
| `qwen_sagemaker` | Qwen3-ASR on vLLM (multilingual) | SageMaker GPU |

**TTS** (`tts.backend`)

| Backend | Model | Hosting |
|---------|-------|---------|
| `fastthaig2p` | Kokoro-82M, Thai-finetuned (ONNX) | Local CPU |
| `omnivoice_sagemaker` | OmniVoice (646 languages, voice cloning) | SageMaker GPU |

Deploying, verifying, and tearing down the SageMaker endpoints, including the execution role and GPU quota, is documented in **[`sagemaker/README.md`](sagemaker/README.md)**. GPU endpoints bill until deleted.

## Configuration

The agent is fully configured via YAML in `configs/`. Pass `--config <path>` to select one.

```yaml
agent_name: telco-agent           # LiveKit agent registration name
scenario: scenarios/telco_th.json # Scenario file (tools + fixtures); omit for no-tools chat

preemptive_generation: true       # run the LLM during the endpointing wait

llm:
  model: openai.gpt-oss-120b-1:0  # any supported Bedrock model id
  region: us-west-2
  reasoning_effort: low           # low/medium/high (for reasoning models)
  persistent_connection: true     # reuse one Bedrock connection across turns
  prewarm: true                   # warm the connection before turn 1

tts:
  backend: fastthaig2p            # fastthaig2p | qwen_sagemaker | omnivoice_sagemaker
  speed: 1.0
  # endpoint_name / region / language   # required for the SageMaker backends

stt:
  backend: typhoon                # typhoon | qwen_sagemaker
  device: cpu                     # cpu/cuda/mps
  # endpoint_name / region / language   # required for the SageMaker backend

vad:
  activation_threshold: 0.55
  min_speech_duration: 0.08
  min_silence_duration: 0.55

prompt:
  persona: |                      # Agent personality and behavior rules
    ...
  voice_rules: |                  # TTS output formatting rules
    ...
```

## Performance

The same 30-utterance Thai benchmark (5 buckets), run on both the local and cloud
speech backends so they're comparable. Apple M4 client, GPT-OSS 120B on Bedrock
(`reasoning_effort=low`), persistent Bedrock connection and preemptive generation.
Overall p50 latency:

| Metric (ms, p50) | Local (Typhoon + FastThaiG2P, CPU) | Cloud (Qwen3-ASR vLLM + OmniVoice, SageMaker) |
|------------------|-----------------------------------:|---------------------------------------------:|
| STT | 300 | 375 |
| LLM TTFT | 617 | 689 |
| LLM Total | 2,107 | 2,702 |
| TTS Total | 1,869 | 2,288 |
| **TTFA** | **3,670** | **4,512** |
| E2E | 4,422 | 5,184 |

**Reading these numbers:** `TTFA` (time to first audio) is what the user actually waits for; `E2E` is time to the *last* audio, i.e. the whole reply synthesized, and is **not** perceived latency. The "overall" figures are weighted toward tool turns, which make **two** LLM round-trips before any audio — a single-turn *no-tool* reply is ~1s faster (TTFA ≈ 3.1–3.8s) than a *tool* turn (≈ 4.5–4.9s). And a **live call is faster still than TTFA here**: the benchmark runs stages sequentially, whereas the agent's preemptive generation runs the LLM *during* the endpointing wait, and you hear the first phrase while the rest streams. The LLM leg also varies run-to-run (Bedrock reasoning).

Backend notes: serving Qwen3-ASR on AWS's vLLM DLC brings its server-side inference to ~60ms (SageMaker `ModelLatency`), so STT p50 is ~375ms including network, vs ~940ms on the plain-`transformers` endpoint; the persistent Bedrock connection cut warm LLM TTFT from ~986ms to ~610ms p50. Tool-call accuracy is 100% on both. Cloud latency also depends on your network distance to the endpoint region.

```bash
uv run python scripts/bench_run.py --config configs/telco_th.yaml            # local
uv run python scripts/bench_run.py --config configs/telco_th_sagemaker.yaml  # cloud
uv run python scripts/bench_run.py --gap 15                                  # production cadence
```

The benchmark builds the LLM the same way the agent does (persistent Bedrock connection), so its per-stage latencies match a live call — but it measures stages sequentially, so live *perceived* latency is a bit lower (preemptive generation overlaps the LLM with the endpointing wait).

## Project Structure

```
src/
  agent.py                      # Config-driven agent entrypoint
  plugins/
    typhoon_stt.py              # Typhoon ASR LiveKit STT plugin (local, default)
    qwen_sagemaker_stt.py       # Qwen3-ASR LiveKit STT plugin (SageMaker)
    fastthaig2p_tts.py          # FastThaiG2P/Kokoro-82M LiveKit TTS plugin (local ONNX, default)
    omnivoice_sagemaker_tts.py  # OmniVoice LiveKit TTS plugin (SageMaker, any language)
    bedrock_session.py          # Persistent, keep-alive Bedrock connection
    thai_tts_phrase_tokenizer.py # Phrase chunking for streaming TTS
    telco_tools.py              # Mock telco tool handlers
    metrics_logger.py           # Pipeline latency tracking

configs/
  telco_*.yaml                  # Telco scenario (tools): th (local + _sagemaker), en, vi, id, ms, fil
  chat_*.yaml                   # Free chat (no tools): th (local + _sagemaker), en, vi, id, ms, fil

scenarios/
  telco_th.json                 # Thai telco fixtures + tool definitions
  telco_intl.json               # Language-neutral fixtures (shared by non-Thai telco configs)

sagemaker/                      # SageMaker speech endpoints (see sagemaker/README.md)
web/                            # Browser UI (index.html) + token server (server.py)
scripts/                        # bench_run.py (latency + accuracy), bench_synth.py (fixtures)
```
