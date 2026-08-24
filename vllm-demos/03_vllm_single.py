#!/usr/bin/env python3
"""Demo 03 — vLLM, sequential requests.

Spawns the Jetson vLLM Docker container itself (no separate launch script),
serves Qwen2.5-3B GPTQ-Int4 on :8000, and streams each prompt in prompts.txt
one at a time (the same prompts 02_ollama.py's sequential leg uses). Reports
TTFT + TPS per request plus an aggregate summary, then shuts the container down.

Usage:
    python3 03_vllm_single.py
    VLLM_PORT=8002 python3 03_vllm_single.py
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import httpx

from bench import Report, highlight_prompt, load_prompts, print_summary, stream_request, write_stats

from vllm_serve import resolve, vllm_server

PROMPTS = load_prompts()
SEQUENTIAL_MAX_TOKENS = 120  # matches 02_ollama.py


async def sequential_demo(base_url: str, model: str) -> None:
    print(f"{len(PROMPTS)} sequential requests")
    print()
    # Warmup: the first request pays one-time GPU/kernel setup cost.
    async with httpx.AsyncClient() as client:
        await stream_request(client, base_url, model, "Warmup.", max_tokens=16)
    measurements = []
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"  [{i}] {highlight_prompt(prompt)}")
        print("      ", end="", flush=True)
        async with httpx.AsyncClient() as client:
            m = await stream_request(
                client, base_url, model, prompt,
                max_tokens=SEQUENTIAL_MAX_TOKENS,
                on_token=lambda text: print(text, end="", flush=True),
            )
        print()
        print(f"      TTFT {m.ttft_s * 1000:6.1f} ms  "
              f"{m.tps:6.2f} tok/s  ({m.completion_tokens} tok)")
        measurements.append(m)
    seq_report = Report(measurements=measurements, concurrency=1)
    seq_label = f"vLLM (GPTQ-Int4) — {len(PROMPTS)} sequential"
    print_summary(seq_label, seq_report)
    write_stats(seq_label, seq_report)


async def main() -> int:
    ap = argparse.ArgumentParser(
        description="Demo 03 — vLLM sequential requests (self-contained)"
    )
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--served-name", default=None)
    ap.add_argument("--container", default=None)
    ap.add_argument("--image", default=None)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--gpu-mem-util", default=None)
    ap.add_argument("--max-model-len", default=None)
    ap.add_argument("--max-num-seqs", default=None)
    ap.add_argument("--kv-cache-bytes", default=None)
    args = ap.parse_args()

    overrides = {k: v for k, v in vars(args).items() if v is not None}
    # One sequence keeps the memory-profiling peak small enough to load
    # reliably on the 8 GB Tegra.
    overrides.setdefault("max_num_seqs", "1")
    cfg = resolve(overrides)

    print(f"Demo 03 — vLLM serving Qwen2.5-3B GPTQ-Int4 ({len(PROMPTS)} sequential)")
    print(f"  image  : {cfg['image']}")
    print(f"  model  : {cfg['model_dir']}")
    print(f"  served : {cfg['served_name']}")
    print()

    try:
        async with vllm_server(overrides, reuse=True) as base_url:
            await sequential_demo(base_url, cfg["served_name"].split()[0])
    except Exception as e:  # noqa: BLE001
        print(f"\n  [!] {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
