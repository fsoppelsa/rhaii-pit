#!/usr/bin/env python3
"""Demo 04 — vLLM, concurrent requests (continuous batching).

Spawns the Jetson vLLM Docker container itself (no separate launch script),
serves Qwen2.5-3B GPTQ-Int4 on :8000, fires every prompt in prompts.txt
concurrently, and reports per-request TTFT/TPS plus the aggregate, then shuts
the container down.

This is the continuous-batching leg: vLLM interleaves the requests instead of
serializing them. Pair it with 03_vllm_single.py (sequential) for the contrast
in how the same model handles load.

Usage:
    python3 04_vllm_batch.py
    VLLM_PORT=8002 python3 04_vllm_batch.py
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import textwrap

from bench import highlight_prompt, load_prompts, print_summary, run_benchmark, write_stats
from vllm_serve import resolve, vllm_server

PROMPTS = load_prompts()

ANSWER_LINES = 4
ANSWER_PREFIX = "        "


class LiveAnswers:
    """Render each concurrent response in its own fixed terminal block."""

    def __init__(self, prompts: list[str]) -> None:
        self.prompts = prompts
        self.answers = [""] * len(prompts)
        self.live = sys.stdout.isatty()
        self.width = max(
            shutil.get_terminal_size((100, 24)).columns - len(ANSWER_PREFIX), 20
        )
        self.block_height = ANSWER_LINES + 2  # prompt, answer rows, spacer

    def start(self) -> None:
        if not self.live:
            print("  answers: streaming display requires an interactive terminal")
            return
        print("  answers (streaming):")
        for i, prompt in enumerate(self.prompts, 1):
            print(f"    [{i}] {highlight_prompt(prompt)}")
            print("\n" * ANSWER_LINES, end="")
            print()
        sys.stdout.write("\x1b[?25l")
        sys.stdout.flush()

    def update(self, index: int, text: str) -> None:
        self.answers[index] += text
        if not self.live:
            return
        lines = textwrap.wrap(
            self.answers[index].strip(),
            width=self.width,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [""]
        if len(lines) > ANSWER_LINES:
            lines = lines[:ANSWER_LINES]
            lines[-1] = lines[-1][:self.width - 1] + "…"
        lines.extend([""] * (ANSWER_LINES - len(lines)))

        target_row = index * self.block_height + 1
        total_rows = len(self.prompts) * self.block_height
        sys.stdout.write(f"\x1b[{total_rows - target_row}A\r")
        for row, line in enumerate(lines):
            sys.stdout.write(f"\x1b[2K{ANSWER_PREFIX}{line}")
            if row < ANSWER_LINES - 1:
                sys.stdout.write("\n")
        sys.stdout.write(f"\x1b[{total_rows - target_row - ANSWER_LINES + 1}B\r")
        sys.stdout.flush()

    def close(self) -> None:
        if self.live:
            sys.stdout.write("\x1b[?25h")
            sys.stdout.flush()


async def batch_demo(base_url: str, model: str) -> None:
    print(f"{len(PROMPTS)} concurrent requests (continuous batching)")
    print()
    live_answers = LiveAnswers(PROMPTS)
    live_answers.start()
    try:
        report = await run_benchmark(
            base_url,
            model,
            PROMPTS,
            max_tokens=96,
            concurrency=len(PROMPTS),
            on_token=live_answers.update,
        )
    finally:
        live_answers.close()
    if not live_answers.live:
        for i, measurement in enumerate(report.measurements, 1):
            print(f"    [{i}] {highlight_prompt(measurement.prompt)}")
            print(f"        {measurement.text.strip()}")
        print()
    for i, measurement in enumerate(report.measurements, 1):
        print(f"    req {i}: TTFT {measurement.ttft_s * 1000:6.1f} ms  "
              f"{measurement.tps:6.2f} tok/s  "
              f"({measurement.completion_tokens} tok)")
    batch_label = f"vLLM (GPTQ-Int4) — {len(PROMPTS)} concurrent"
    print_summary(batch_label, report)
    write_stats(batch_label, report)


async def main() -> int:
    ap = argparse.ArgumentParser(
        description="Demo 04 — vLLM concurrent batch (self-contained)"
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
    cfg = resolve(overrides)

    print(f"Demo 04 — vLLM serving Qwen2.5-3B GPTQ-Int4 ({len(PROMPTS)} concurrent)")
    print(f"  image  : {cfg['image']}")
    print(f"  model  : {cfg['model_dir']}")
    print(f"  served : {cfg['served_name']}")
    print()

    try:
        async with vllm_server(overrides, reuse=True) as base_url:
            await batch_demo(base_url, cfg["served_name"].split()[0])
    except Exception as e:  # noqa: BLE001
        print(f"\n  [!] {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
