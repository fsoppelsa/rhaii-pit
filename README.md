# RHAII Inference Stack

Local GPU-backed vLLM server (Red Hat AI Inference or upstream vLLM) with an optional reasoning-hiding proxy and Open WebUI frontend, on Podman (RHEL-like hosts).

Jetson Orin variant: [`orin-vllm.sh`](orin-vllm.sh).

## Configuring

Every script has a `CONFIGURATION — edit these` block at the top, above a `DO NOT EDIT BELOW THIS LINE` separator. Edit the plain values above the line; leave everything below it. Where noted, an environment variable of the same purpose still overrides the value at runtime.

- [`inference-manage.sh`](inference-manage.sh) — model, backend (`UPSTREAM=0` for Red Hat / `1` for upstream), ports, cache dir, WebUI host/port
- [`rhaii-universal.sh`](rhaii-universal.sh) — image, container, per-model quant/memory tuning
- [`model-downloader.sh`](model-downloader.sh) — default model, cache dir, offline flag, token file
- [`start-webui.sh`](start-webui.sh) — image, branding, host/port, cert dir, feature toggles
- [`orin-vllm.sh`](orin-vllm.sh) — Jetson model, container, memory tuning

## Start / Stop

```bash
./inference-manage.sh start rhaii              # Red Hat AI Inference backend (headless)
./inference-manage.sh start upstream           # upstream vLLM backend (headless)
./inference-manage.sh start rhaii --with-proxy # + reasoning-hiding proxy (filtered API on :8001)
./inference-manage.sh start rhaii --with-ui    # + Open WebUI (and TLS proxy on :443)
./inference-manage.sh start rhaii --bearer KEY # require "Authorization: Bearer KEY" on /v1/*
./inference-manage.sh start rhaii --smoke-test # send a test prompt after boot; nonzero exit on failure
./inference-manage.sh stop                     # stop every running component
./inference-manage.sh smoke-test               # probe an already-running backend
```

Flags combine. Set `UPSTREAM=1` in the config block to default to upstream vLLM, or use `start upstream` for a one-off run. To change the model, edit `MODEL` at the top of the script. The settings table printed on startup summarizes the active configuration. Lower-level launchers:

```bash
./rhaii-universal.sh [--api-key KEY]                    # run the backend directly (env RHAII_UPSTREAM=1 for upstream)
HF_HUB_OFFLINE=0 HF_TOKEN=hf_xxx ./model-downloader.sh  # cache a model only
```

## Models

`MODEL` / `MODEL_KEY` is a catalog key, or set a raw Hugging Face id directly:

| key | model id |
|---|---|
| `deepseek_r1_qwen_14b_awq` | `casperhansen/deepseek-r1-distill-qwen-14b-awq` |
| `qwen3_4b` | `Qwen/Qwen3-4B-Instruct-2507` |
| `qwen3_14b` | `RedHatAI/Qwen3-14B-quantized.w4a16` |
| `granite_8b` | `ibm-granite/granite-3.3-8b-instruct` |
| `llama31_8b` | `meta-llama/Llama-3.1-8B-Instruct` |
| `whiterabbit_7b_awq` | `solidrust/WhiteRabbitNeo-7B-v1.5a-AWQ` |

Per-model quantization and memory defaults live in `rhaii-universal.sh`.

## model-downloader.sh

Downloads or verifies a model in the local HF cache (`$HOME/rhaii-cache`). Run automatically by the backend launcher, or standalone. Key env vars:

- `MODEL_KEY` / `MODEL` — catalog key or raw HF id
- `HF_HUB_OFFLINE` — `1` require local (default), `0` allow download
- `HF_TOKEN` / `HF_TOKEN_FILE` — token (file default `$HOME/HF_TOKEN`), needed when downloading
- `MODEL_DOWNLOAD_FORCE=1` — re-download even if already cached

## Reasoning proxy (reasoning_proxy.py)

OpenAI passthrough that strips `<think>…</think>` reasoning blocks from model output. Started by `--with-proxy`; the filtered API is served on `:8001`. Env vars:

- `UPSTREAM_BASE` (default `http://127.0.0.1:8000`) — the vLLM backend
- `LISTEN_HOST` / `LISTEN_PORT` (default `0.0.0.0:8001`)
- `REQUEST_TIMEOUT` (default `300`)
- `THINKING_PLACEHOLDER` — text substituted for removed reasoning (default empty)

## Consume from outside

OpenAI-compatible on `/v1/*`. Route port `8000` to the host and point clients at `http://<host>:8000/v1`. The API is open unless started with `--bearer KEY` / `--api-key KEY`.

```bash
curl http://<host>:8000/v1/chat/completions \
  -H "Content-Type: application/json" -H "Authorization: Bearer EMPTY" \
  -d '{"model":"qwen3_14b","messages":[{"role":"user","content":"Give me three Linux tips."}]}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://<host>:8000/v1", api_key="EMPTY")
print(client.chat.completions.create(
    model="qwen3_14b",
    messages=[{"role": "user", "content": "Hello"}],
).choices[0].message.content)
```

## Prerequisites

`./setup.sh` (RHEL-like hosts) installs `nvidia-container-toolkit`, generates CDI config, and sets the `container_use_devices` SELinux boolean.

## Future work

- Move the shared values (cache dir, ports, WebUI host, container names) into a single `rhaii-inference.conf` sourced by all scripts.
- Add a Mistral model to the catalog.
