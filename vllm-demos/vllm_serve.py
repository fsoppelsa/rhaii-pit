#!/usr/bin/env python3
"""Launch vLLM's OpenAI-compatible server from Python (Docker).

Shared by the vLLM single-request, batching, and comparison demos so there is
no separate launch script. Spawns the Jetson
vLLM container, drops caches while it loads,
waits for /v1/models, and tears the container down on exit. The KV cache is
capped explicitly (--kv-cache-memory-bytes) because vLLM's utilization-based
estimate over-allocates and hits NvMap ENOMEM on this 8 GB Tegra.
"""

from __future__ import annotations
import argparse
import asyncio
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx

DEFAULT_IMAGE = "ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin"
DEFAULT_MODEL_DIR = "/srv/share/vllm/models/Qwen2.5-3B-Instruct-GPTQ-Int4"
DEFAULT_SERVED_NAME = "edge3b orin2b"
DEFAULT_PORT = 8000
DEFAULT_CONTAINER = "vllm-demo"
DEFAULT_QUANTIZATION = "gptq"
DEFAULT_KV_CACHE_BYTES = "268435456"  # 256 MiB — explicit cap, see README
DEFAULT_GPU_MEM_UTIL = "0.50"
READY_TIMEOUT = 240
READY_POLL_S = 3.0


def resolve(overrides: dict | None = None) -> dict:
    """Merge explicit args, env, and defaults (mirrors the old launch script)."""
    o = overrides or {}

    def get(key: str, env: str, default):
        return o.get(key, os.environ.get(env, default))

    return {
        "image":          get("image",          "VLLM_IMAGE",          DEFAULT_IMAGE),
        "model_dir":      get("model_dir",      "VLLM_MODEL_DIR",      DEFAULT_MODEL_DIR),
        "served_name":    get("served_name",    "VLLM_SERVED_NAME",    DEFAULT_SERVED_NAME),
        "port":           int(get("port",           "VLLM_PORT",           DEFAULT_PORT)),
        "container":      get("container",      "VLLM_CONTAINER",      DEFAULT_CONTAINER),
        "gpu_mem_util":   get("gpu_mem_util",   "VLLM_GPU_MEM_UTIL",   DEFAULT_GPU_MEM_UTIL),
        "quantization":   get("quantization",   "VLLM_QUANTIZATION",   DEFAULT_QUANTIZATION),
        "max_model_len":  get("max_model_len",  "VLLM_MAX_MODEL_LEN",  "2048"),
        "max_num_seqs":   get("max_num_seqs",   "VLLM_MAX_NUM_SEQS",   "8"),
        "kv_cache_bytes": get("kv_cache_bytes", "VLLM_KV_CACHE_BYTES", DEFAULT_KV_CACHE_BYTES),
    }


def _build_cmd(cfg: dict) -> list[str]:
    return [
        "docker", "run", "-d",
        "--name", cfg["container"],
        "--runtime=nvidia", "--shm-size", "1g",
        "--ulimit", "memlock=-1:-1", "--ulimit", "stack=67108864:67108864",
        "-e", "NVIDIA_VISIBLE_DEVICES=all",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
        "-e", "PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync",
        "-e", "NCCL_P2P_DISABLE=1", "-e", "NCCL_IB_DISABLE=1",
        "-e", "NCCL_CUMEM_ENABLE=0", "-e", "NCCL_SHM_DISABLE=1",
        "-p", f"{cfg['port']}:8000",
        "-v", f"{cfg['model_dir']}:/model:ro",
        cfg["image"],
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "/model",
        "--served-model-name", *cfg["served_name"].split(),
        "--host", "0.0.0.0", "--port", "8000",
        "--quantization", cfg["quantization"],
        "--dtype", "float16",
        "--enforce-eager",
        "--gpu-memory-utilization", cfg["gpu_mem_util"],
        "--max-model-len", cfg["max_model_len"],
        "--max-num-seqs", cfg["max_num_seqs"],
        "--kv-cache-memory-bytes", cfg["kv_cache_bytes"],
    ]


def _drop_caches() -> None:
    """Best-effort: free Tegra unified memory held by page cache."""
    try:
        subprocess.run(
            ["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
            check=False, timeout=10,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:  # noqa: BLE001 — sudo absent or requires password
        pass


async def _docker(*args: str) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        "docker", *args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate()
    return proc.returncode, out.decode(errors="replace")


async def _docker_run(cfg: dict) -> None:
    proc = await asyncio.create_subprocess_exec(
        *_build_cmd(cfg),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"docker run failed:\n{out.decode(errors='replace').strip()}")


async def _container_running(cfg: dict) -> bool:
    code, out = await _docker("ps", "--format", "{{.Names}}")
    return cfg["container"] in out.split()


async def _logs_tail(cfg: dict, n: int = 80) -> str:
    _, out = await _docker("logs", "--tail", str(n), cfg["container"])
    return out


async def _stream_logs(cfg: dict, stop: asyncio.Event) -> None:
    """Print vLLM container logs live while it loads (docker logs -f)."""
    proc = await asyncio.create_subprocess_exec(
        "docker", "logs", "-f", cfg["container"],
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdout is not None
    try:
        while not stop.is_set():
            line = await proc.stdout.readline()
            if not line:
                break
            print(f"  [vllm] {line.decode(errors='replace').rstrip()}", flush=True)
    except asyncio.CancelledError:
        pass
    finally:
        try:
            proc.terminate()
            await proc.wait()
        except (ProcessLookupError, Exception):  # noqa: BLE001
            pass


def _summarize(logs: str) -> str:
    """One-line cause for the common startup failure, else a generic message."""
    for line in logs.splitlines():
        if "NvMapMemAlloc" in line or "would exceed allowed memory" in line:
            return (
                "not enough free unified memory — NvMap allocation failed while "
                "vLLM profiled the KV cache. The Jetson's 8 GB is shared with "
                "page cache; drop caches again (sync; echo 3 > /proc/sys/vm/drop_caches) "
                "or reboot. This run already drops caches every 3s during load."
            )
        if "Free memory on device" in line:
            return (
                "not enough free GPU memory — the 3B model needs ~3.7 GiB but the "
                "Jetson's unified memory is fragmented (long uptime). Reboot it, or "
                "lower --gpu-mem-util / --max-model-len / --max-num-seqs."
            )
    return "vLLM container exited during startup"


async def _rm(cfg: dict) -> None:
    await _docker("rm", "-f", cfg["container"])


async def _drop_loop(cfg: dict, stop: asyncio.Event) -> None:
    """Drop caches immediately, then every 3s while the container loads.

    NvMap ENOMEM workaround: reading the ~2 GB safetensors refills the page
    cache; vLLM's memory profiler runs right after weight loading, so the
    cache must stay dropped through the whole load, not just at start.
    """
    _drop_caches()
    try:
        while True:
            try:
                await asyncio.wait_for(stop.wait(), timeout=3.0)
                return
            except asyncio.TimeoutError:
                _drop_caches()
    except asyncio.CancelledError:
        pass


@asynccontextmanager
async def vllm_server(overrides: dict | None = None, reuse: bool = False) -> AsyncIterator[str]:
    """Start the vLLM container, wait for /v1/models, yield base URL, stop it."""
    cfg = resolve(overrides)
    if not os.path.isdir(cfg["model_dir"]):
        raise FileNotFoundError(f"model_dir not found: {cfg['model_dir']}")
    base_url = f"http://127.0.0.1:{cfg['port']}/v1"
    if reuse and await _container_running(cfg):
        # Use an already-running server and leave it up on exit.
        deadline = time.monotonic() + READY_TIMEOUT
        while time.monotonic() < deadline:
            try:
                async with httpx.AsyncClient() as client:
                    r = await client.get(base_url + "/models", timeout=2.0)
                if r.status_code == 200:
                    break
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(READY_POLL_S)
        else:
            raise RuntimeError(
                f"reuse: container {cfg['container']} up but not ready on "
                f"{base_url} within {READY_TIMEOUT}s"
            )
        yield base_url
        return

    await _rm(cfg)  # clear any stale container with this name
    _drop_caches()  # free page cache before the container grabs unified memory
    await _docker_run(cfg)

    stop = asyncio.Event()
    drop_task = asyncio.create_task(_drop_loop(cfg, stop))
    log_task = asyncio.create_task(_stream_logs(cfg, stop))

    base_url = f"http://127.0.0.1:{cfg['port']}/v1"
    try:
        deadline = time.monotonic() + READY_TIMEOUT
        while time.monotonic() < deadline:
            if not await _container_running(cfg):
                logs = await _logs_tail(cfg)
                raise RuntimeError(f"{_summarize(logs)}\n\n{logs.strip()}")
            try:
                async with httpx.AsyncClient() as client:
                    r = await client.get(base_url + "/models", timeout=2.0)
                if r.status_code == 200:
                    break
            except Exception:  # noqa: BLE001 — server still starting
                pass
            await asyncio.sleep(READY_POLL_S)
        else:
            logs = await _logs_tail(cfg)
            raise RuntimeError(
                f"vLLM not ready on {base_url} within {READY_TIMEOUT}s:\n{logs.strip()}"
            )
        yield base_url
    finally:
        stop.set()
        for task in (drop_task, log_task):
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        await _rm(cfg)


def _build_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="vLLM server control (Jetson Orin)")
    ap.add_argument(
        "action", nargs="?", choices=["up", "down", "logs"], default="up",
        help="up: start & leave running (default); down: stop; logs: last 60 lines",
    )
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--served-name", default=None)
    ap.add_argument("--container", default=None)
    ap.add_argument("--image", default=None)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--gpu-mem-util", default=None)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--max-model-len", default=None)
    ap.add_argument("--max-num-seqs", default=None)
    ap.add_argument("--kv-cache-bytes", default=None)
    return ap.parse_args()


async def _up(cfg: dict) -> None:
    if not os.path.isdir(cfg["model_dir"]):
        raise FileNotFoundError(f"model_dir not found: {cfg['model_dir']}")
    await _rm(cfg)
    _drop_caches()
    await _docker_run(cfg)
    base_url = f"http://127.0.0.1:{cfg['port']}/v1"
    print(f"  starting vLLM ({cfg['container']}) on {base_url} ...", flush=True)
    stop = asyncio.Event()
    drop_task = asyncio.create_task(_drop_loop(cfg, stop))
    log_task = asyncio.create_task(_stream_logs(cfg, stop))
    deadline = time.monotonic() + READY_TIMEOUT
    try:
        while time.monotonic() < deadline:
            if not await _container_running(cfg):
                logs = await _logs_tail(cfg)
                raise RuntimeError(f"{_summarize(logs)}\n\n{logs.strip()}")
            try:
                async with httpx.AsyncClient() as client:
                    r = await client.get(base_url + "/models", timeout=2.0)
                if r.status_code == 200:
                    break
            except Exception:  # noqa: BLE001 — not ready yet
                pass
            await asyncio.sleep(READY_POLL_S)
        else:
            logs = await _logs_tail(cfg)
            raise RuntimeError(
                f"vLLM not ready on {base_url} within {READY_TIMEOUT}s:\n{logs.strip()}"
            )
    finally:
        stop.set()
        for task in (drop_task, log_task):
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
    print(f"  vLLM up: {base_url}  (served as '{cfg['served_name']}')")
    _, ps_out = await _docker("ps", "--filter", f"name={cfg['container']}")
    print(ps_out.rstrip())


async def _down(cfg: dict) -> None:
    await _rm(cfg)
    print(f"  stopped container {cfg['container']}")


async def _logs(cfg: dict) -> None:
    code, out = await _docker("logs", "--tail", "60", cfg["container"])
    if code != 0:
        print(f"  (no logs: container {cfg['container']} not found)")
        return
    print(out)


if __name__ == "__main__":
    args = _build_cli()
    overrides = {
        k: v for k, v in vars(args).items()
        if k != "action" and v is not None
    }
    cfg = resolve(overrides)
    try:
        if args.action == "up":
            asyncio.run(_up(cfg))
        elif args.action == "down":
            asyncio.run(_down(cfg))
        elif args.action == "logs":
            asyncio.run(_logs(cfg))
    except RuntimeError as e:
        print(f"\n  [!] {e}")
        raise SystemExit(1)
