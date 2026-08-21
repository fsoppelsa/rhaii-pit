#!/usr/bin/env python3
"""Streaming benchmark for a running vLLM OpenAI backend.

Measures TTFT, inter-token latency (ITL), and output tok/s from the SSE
stream of a single chat completion. No dependencies beyond stdlib.

Usage:
  bench.py [--url http://127.0.0.1:8000/v1] [--model NAME] [--prompt P]
           [--tokens N] [--json-out FILE]

--tokens requests an exact completion length via max_tokens; short
generations skew ITL low (no tail-ramp effects).
"""
import argparse
import json
import sys
import time
import urllib.request

PROMPTS = {
    "short": "Say exactly: hello world.",
    "explain": "Explain in about 200 words why the sky is blue.",
    "long": "Write a 500-word story about a lighthouse keeper who discovers a radio that broadcasts tomorrow's weather.",
    "reason": "A farmer has 17 sheep. All but 9 run away. How many are left? Think carefully, then answer.",
}


def sse_stream(url, model, prompt, max_tokens):
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0.0,
        }
    ).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    marks = []
    completion_tokens = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            usage = event.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens", completion_tokens)
            try:
                delta = event["choices"][0]["delta"]
            except (KeyError, IndexError):
                continue
            text = delta.get("content") or delta.get("reasoning")
            if text:
                marks.append(time.perf_counter())
    if not marks:
        raise RuntimeError("no content or reasoning deltas received")
    return t0, marks, completion_tokens or len(marks)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen_27b")
    ap.add_argument("--prompt", default="explain", help="one of: " + ",".join(PROMPTS))
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    prompt = PROMPTS.get(args.prompt, args.prompt)
    t0, marks, completion_tokens = sse_stream(
        args.url, args.model, prompt, args.tokens
    )

    ttft = marks[0] - t0
    chunk_count = len(marks)
    itls = [marks[i] - marks[i - 1] for i in range(1, chunk_count)]
    itls.sort()
    total = marks[-1] - t0
    med = itls[len(itls) // 2] if itls else float("nan")
    p90 = itls[int(len(itls) * 0.9)] if itls else float("nan")
    decode_tokens = max(completion_tokens - 1, 0)
    decode_time = marks[-1] - marks[0]
    result = {
        "model": args.model,
        "prompt": args.prompt,
        "completion_tokens": completion_tokens,
        "sse_chunks": chunk_count,
        "tokens_per_chunk": round(completion_tokens / chunk_count, 2),
        "ttft_s": round(ttft, 3),
        "median_chunk_itl_ms": round(med * 1000, 1),
        "p90_chunk_itl_ms": round(p90 * 1000, 1),
        "total_s": round(total, 3),
        "tok_s": round(decode_tokens / decode_time, 2) if decode_time > 0 else 0.0,
    }
    print(json.dumps(result, indent=2))
    if args.json_out:
        with open(args.json_out, "a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f"BENCH-FAILED: {e}", file=sys.stderr)
        sys.exit(1)
