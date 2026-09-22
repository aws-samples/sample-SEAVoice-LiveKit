"""Latency benchmark — simulates the real streaming pipeline.

Feeds pre-recorded WAV files through the same pipeline as agent.py:
  STT (full) → LLM (streaming) → Phrase tokenizer → TTS (chunked with prefetch)

It builds the LLM the same way agent.py does (via ``_make_llm``), so it uses the
persistent Bedrock connection — the warm per-turn latency it reports matches what a
live conversation sees. The warmup opens that connection before turn 1, mirroring
the agent's greeting-time prewarm. Use ``--gap`` to insert an idle pause between
turns (production turns are ~15-20 s apart); this only changes results when the
persistent connection is disabled, in which case each turn pays a cold handshake.

Measures per turn: STT, LLM TTFT, LLM Total, TTS TTFB, TTS Total, TTFA (time to
first audio) and E2E (to last audio), plus tool-call accuracy. Both are measured
from audio-in and exclude the VAD/endpointing silence wait that precedes them in a
live call.

Usage:
    uv run python scripts/bench_run.py
    uv run python scripts/bench_run.py --config configs/telco_th.yaml
    uv run python scripts/bench_run.py --gap 15   # simulate production cadence
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from fastthaig2p import normalize as normalize_for_tts
from livekit import rtc
from livekit.agents import llm

from plugins import telco_tools
from plugins.thai_tts_phrase_tokenizer import ThaiTTSPhraseTokenizer

load_dotenv(".env.local")

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "tests" / "fixtures" / "bench" / "manifest.json"


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    vs = sorted(values)
    k = (len(vs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(vs) - 1)
    return vs[lo] + (vs[hi] - vs[lo]) * (k - lo)


async def run_benchmark(skip_warmup: bool, config_path: Path, gap: float) -> int:
    from agent import _build_system_prompt, _make_llm, _make_stt, _make_tts, load_config

    config = load_config(config_path)
    print(
        f"[bench] config: {config_path.name} (llm={config.get('llm', {}).get('model')})"
    )

    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    print(f"[bench] {len(entries)} utterances")
    print()

    # --- Load models (same construction path as the live agent) ---
    print("[bench] Loading STT model...")
    stt = _make_stt(config)

    print("[bench] Loading TTS model...")
    tts = _make_tts(config)
    # Local backends load a model; SageMaker backends open a client. Warm whichever.
    if hasattr(tts, "_ensure_model"):
        tts._ensure_model()
    elif hasattr(tts, "_ensure_client"):
        tts._ensure_client()

    print("[bench] Loading LLM...")
    llm_instance, _bedrock_session = _make_llm(config)

    system_prompt = _build_system_prompt(config)

    # Tool definitions
    from livekit.agents import function_tool

    @function_tool
    def check_balance(customer_id: str):
        """Look up the customer's outstanding bill balance and due date.

        Args:
            customer_id: The customer's 4-digit ID, e.g. '2749'.
        """
        return {}

    @function_tool
    def check_plan(customer_id: str):
        """Look up the customer's current plan: monthly fee and quotas.

        Args:
            customer_id: The customer's 4-digit ID, e.g. '2749'.
        """
        return {}

    @function_tool
    def check_usage(customer_id: str):
        """Look up this month's usage (voice and data) and remaining quota.

        Args:
            customer_id: The customer's 4-digit ID, e.g. '2749'.
        """
        return {}

    @function_tool
    def open_ticket(customer_id: str, category: str, summary: str):
        """Open a support ticket for the customer's issue.

        Args:
            customer_id: The customer's 4-digit ID, e.g. '2749'.
            category: One of: billing, network, device, plan_change, other.
            summary: A one-sentence summary of the issue, in the customer's language.
        """
        return {}

    tool_defs = [check_balance, check_plan, check_usage, open_ticket]
    phrase_tokenizer = ThaiTTSPhraseTokenizer(min_phrase_len=10)
    tts_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts")

    # Warmup
    if not skip_warmup:
        print("[bench] warmup turn...")
        warmup_path = REPO_ROOT / entries[0].get(
            "path", "tests/fixtures/bench/00_greet.wav"
        )
        if warmup_path.exists():
            audio, sr = sf.read(str(warmup_path))
            audio = np.asarray(audio, dtype=np.float32)
            if sr != 16000:
                import librosa

                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
            frame = rtc.AudioFrame(
                data=(audio * 32767).astype(np.int16).tobytes(),
                sample_rate=sr,
                num_channels=1,
                samples_per_channel=len(audio),
            )
            await stt._recognize_impl(buffer=[frame])
        # Warm the persistent LLM connection + prompt cache, like the agent's
        # greeting-time prewarm, so turn 1 isn't cold.
        warmup_ctx = llm.ChatContext()
        warmup_ctx.add_message(role="system", content=system_prompt)
        warmup_ctx.add_message(role="user", content="hello")
        warmup_stream = llm_instance.chat(chat_ctx=warmup_ctx)
        async for _ in warmup_stream:
            pass
        await warmup_stream.aclose()
        print("[bench] warmup done")

    print()
    print("=" * 100)
    print(
        f"{'#':<4} {'bucket':<8} {'STT':>5} {'TTFT':>5} {'LLM':>5} "
        f"{'TTFB':>5} {'TTFA':>5} {'TTS':>5} {'E2E':>5}  {'tool':>14}  {'ok'}"
    )
    print("-" * 100)

    results: list[dict] = []

    for entry in entries:
        # Idle pause between turns to mimic real conversational cadence.
        if gap and results:
            await asyncio.sleep(gap)

        path = REPO_ROOT / entry.get(
            "path",
            f"tests/fixtures/bench/{entry['idx']:02d}_{entry['bucket']}.wav",
        )
        if not path.exists():
            print(f"  [{entry['idx']:02d}] SKIP — {path.name} not found")
            continue

        audio, sr = sf.read(str(path))
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        audio = np.asarray(audio, dtype=np.float32)

        if sr != 16000:
            import librosa

            audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        else:
            audio_16k = audio

        e2e_start = time.perf_counter()

        # --- STT ---
        stt_start = time.perf_counter()
        frame = rtc.AudioFrame(
            data=(audio_16k * 32767).astype(np.int16).tobytes(),
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=len(audio_16k),
        )
        event = await stt._recognize_impl(buffer=[frame])
        stt_text = event.alternatives[0].text if event.alternatives else ""
        stt_ms = (time.perf_counter() - stt_start) * 1000

        # --- LLM (streaming) → Phrase tokenizer → TTS (chunked, interleaved) ---
        conversation: list[dict] = [{"role": "user", "content": stt_text}]

        chat_ctx = llm.ChatContext()
        chat_ctx.add_message(role="system", content=system_prompt)
        chat_ctx.add_message(role="user", content=stt_text)

        llm_start = time.perf_counter()
        ttft = None
        llm_text = ""
        tools_called: list[str] = []
        ttfa = None
        loop = asyncio.get_event_loop()

        # Accumulate LLM tokens and detect phrase boundaries in real-time
        text_buffer = ""
        tts_futures: list[asyncio.Future] = []
        first_phrase_submitted = False
        first_tts_submit_time: float | None = None

        stream = llm_instance.chat(chat_ctx=chat_ctx, tools=tool_defs)
        async for chunk in stream:
            if ttft is None:
                ttft = (time.perf_counter() - llm_start) * 1000
            if chunk.delta and chunk.delta.content:
                content = chunk.delta.content
                llm_text += content
                text_buffer += content

                # Check if we have a complete phrase
                candidate_phrases = phrase_tokenizer.tokenize(
                    normalize_for_tts(text_buffer)
                )
                # If tokenizer found >1 phrase, the first ones are complete
                if len(candidate_phrases) > 1:
                    for phrase in candidate_phrases[:-1]:
                        if phrase.strip():
                            # Submit to TTS immediately
                            future = loop.run_in_executor(
                                tts_executor, tts._generate_audio, phrase
                            )
                            tts_futures.append(future)
                            if not first_phrase_submitted:
                                first_phrase_submitted = True
                                first_tts_submit_time = time.perf_counter()
                    # Keep only the incomplete last part
                    text_buffer = candidate_phrases[-1]

            if chunk.delta and chunk.delta.tool_calls:
                for tc in chunk.delta.tool_calls:
                    if tc.name and tc.name not in tools_called:
                        tools_called.append(tc.name)
        await stream.aclose()

        llm_total_ms = (time.perf_counter() - llm_start) * 1000
        llm_ttft_ms = ttft or 0

        # --- If tools were called, execute them and do a second LLM turn ---
        if tools_called and not llm_text:
            # Execute the tool
            tool_handlers = {
                "check_balance": telco_tools.check_balance,
                "check_plan": telco_tools.check_plan,
                "check_usage": telco_tools.check_usage,
                "open_ticket": telco_tools.open_ticket,
            }
            tool_name = tools_called[0]
            handler = tool_handlers.get(tool_name)
            # Use default args since we can't parse from the stream easily
            if tool_name == "open_ticket":
                tool_result = handler("2749", "other", "ลูกค้าแจ้งปัญหา") if handler else {}
            else:
                tool_result = handler("2749") if handler else {}
            conversation.append(
                {
                    "role": "tool_call",
                    "name": tool_name,
                    "args": {"customer_id": "2749"},
                }
            )
            conversation.append(
                {"role": "tool_result", "name": tool_name, "result": tool_result}
            )

            # Second LLM turn: feed tool result back as user context
            chat_ctx.add_message(
                role="user",
                content=f"ผลจากระบบ: {json.dumps(tool_result, ensure_ascii=False)}\n\nกรุณาตอบลูกค้าตามข้อมูลข้างต้น",
            )

            llm2_start = time.perf_counter()
            stream2 = llm_instance.chat(chat_ctx=chat_ctx, tools=tool_defs)
            async for chunk in stream2:
                if chunk.delta and chunk.delta.content:
                    content = chunk.delta.content
                    llm_text += content
                    text_buffer += content

                    candidate_phrases = phrase_tokenizer.tokenize(
                        normalize_for_tts(text_buffer)
                    )
                    if len(candidate_phrases) > 1:
                        for phrase in candidate_phrases[:-1]:
                            if phrase.strip():
                                future = loop.run_in_executor(
                                    tts_executor, tts._generate_audio, phrase
                                )
                                tts_futures.append(future)
                                if not first_phrase_submitted:
                                    first_phrase_submitted = True
                                    first_tts_submit_time = time.perf_counter()
                        text_buffer = candidate_phrases[-1]
            await stream2.aclose()

            llm_total_ms += (time.perf_counter() - llm2_start) * 1000

        # Flush remaining buffer as final phrase
        remaining = normalize_for_tts(text_buffer.strip())
        if remaining:
            future = loop.run_in_executor(tts_executor, tts._generate_audio, remaining)
            tts_futures.append(future)
            if not first_phrase_submitted:
                first_phrase_submitted = True
                first_tts_submit_time = time.perf_counter()

        # Record final assistant response
        if llm_text:
            conversation.append({"role": "assistant", "content": llm_text})

        # If still no text (shouldn't happen now), use placeholder
        if not tts_futures:
            future = loop.run_in_executor(tts_executor, tts._generate_audio, "ขออภัยค่ะ")
            tts_futures.append(future)

        # Await TTS results — first one gives us TTFA and TTS TTFB
        tts_start = time.perf_counter()
        for future in tts_futures:
            await future
            if ttfa is None:
                ttfa = (time.perf_counter() - e2e_start) * 1000

        tts_total_ms = (time.perf_counter() - tts_start) * 1000
        # TTS TTFB = time from first TTS submission to first TTS completion
        if first_tts_submit_time and ttfa:
            tts_ttfb_ms = ttfa - (first_tts_submit_time - e2e_start) * 1000
        else:
            tts_ttfb_ms = tts_total_ms
        e2e_ms = (time.perf_counter() - e2e_start) * 1000
        ttfa_ms = ttfa or e2e_ms
        # Time from start to first phrase ready for TTS
        first_sentence_ms = (
            (first_tts_submit_time - e2e_start) * 1000
            if first_tts_submit_time
            else ttfa_ms
        )

        # --- Evaluate tool call ---
        expected = entry["expected_tool"]
        if expected == "":
            tool_ok = "✓" if not tools_called else "✗"
        else:
            tool_ok = "✓" if expected in tools_called else "✗"

        tools_str = ",".join(tools_called) or "-"

        print(
            f"{entry['idx']:>3}  {entry['bucket']:<8} "
            f"{stt_ms:>5.0f} {llm_ttft_ms:>5.0f} {llm_total_ms:>5.0f} "
            f"{tts_ttfb_ms:>5.0f} {ttfa_ms:>5.0f} {tts_total_ms:>5.0f} {e2e_ms:>5.0f}  "
            f"{tools_str:>14}  {tool_ok}"
        )

        results.append(
            {
                "idx": entry["idx"],
                "bucket": entry["bucket"],
                "expected_tool": expected,
                "gold_text": entry["text"],
                "stt_text": stt_text,
                "llm_text": llm_text,
                "tools_called": tools_called,
                "stt_ms": stt_ms,
                "llm_ttft_ms": llm_ttft_ms,
                "llm_total_ms": llm_total_ms,
                "tts_ttfb_ms": tts_ttfb_ms,
                "first_sentence_ms": first_sentence_ms,
                "ttfa_ms": ttfa_ms,
                "tts_total_ms": tts_total_ms,
                "e2e_ms": e2e_ms,
                "tool_ok": tool_ok == "✓",
                "conversation": conversation,
            }
        )

    print("=" * 100)
    print()

    # Save raw results
    out_path = REPO_ROOT / "logs" / "bench_results.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[bench] Raw results: {out_path}")
    print()

    _print_summary(results)
    tts_executor.shutdown(wait=False)
    if _bedrock_session is not None:
        await _bedrock_session.aclose_all()
    return 0


def _print_summary(results: list[dict]) -> None:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        buckets[r["bucket"]].append(r)
    buckets["overall"] = results[:]
    buckets["with_tools"] = [r for r in results if r["expected_tool"]]
    buckets["no_tools"] = [r for r in results if not r["expected_tool"]]

    metrics = [
        ("stt_ms", "STT"),
        ("llm_ttft_ms", "LLM TTFT"),
        ("llm_total_ms", "LLM Total"),
        ("first_sentence_ms", "1st Sent"),
        ("tts_ttfb_ms", "TTS TTFB"),
        ("ttfa_ms", "TTFA"),
        ("tts_total_ms", "TTS Total"),
        ("e2e_ms", "E2E"),
    ]

    print("## Latency (ms)")
    print()
    print(
        "TTFA (time to first audio) is what the user waits for; E2E is time to the\n"
        "LAST audio (whole reply synthesized), not perceived latency. The live agent\n"
        "is faster still: preemptive generation overlaps the LLM with the endpointing\n"
        "wait. Compare the no_tools rows (one LLM round-trip) vs with_tools (two).\n"
    )
    print(
        f"{'metric':<10} {'bucket':<12} {'n':>3}  "
        f"{'mean':>7}  {'p50':>7}  {'p90':>7}  {'p95':>7}"
    )
    print("-" * 70)
    for key, label in metrics:
        for name in (
            "greet",
            "balance",
            "plan",
            "usage",
            "ticket",
            "with_tools",
            "no_tools",
            "overall",
        ):
            rows = buckets.get(name, [])
            values = [r[key] for r in rows if r.get(key) is not None]
            if not values:
                continue
            mean = statistics.fmean(values)
            p50 = percentile(values, 0.5)
            p90 = percentile(values, 0.9)
            p95 = percentile(values, 0.95)
            print(
                f"{label:<10} {name:<12} {len(values):>3}  "
                f"{mean:>6.0f}   {p50:>6.0f}   {p90:>6.0f}   {p95:>6.0f}"
            )
        print()

    # Tool-call accuracy
    print("## Tool-call Accuracy")
    print()
    print(f"{'bucket':<12} {'n':>3}  {'correct':>7}  {'wrong':>7}  {'acc':>6}")
    print("-" * 42)
    for name in (
        "greet",
        "balance",
        "plan",
        "usage",
        "ticket",
        "overall",
    ):
        rows = buckets.get(name, [])
        correct = sum(1 for r in rows if r["tool_ok"])
        total = len(rows)
        wrong = total - correct
        acc = correct / total * 100 if total else 0
        print(f"{name:<12} {total:>3}  {correct:>6}    {wrong:>6}    {acc:>5.1f}%")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument(
        "--config",
        default="configs/telco_th.yaml",
        help="Path to YAML config (default: configs/telco_th.yaml)",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=0.0,
        help="Idle seconds between turns to mimic production cadence (default: 0)",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    sys.exit(asyncio.run(run_benchmark(args.skip_warmup, config_path, args.gap)))
