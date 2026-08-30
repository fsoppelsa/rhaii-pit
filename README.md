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
- [`tools/benchmark.sh`](tools/benchmark.sh) — containerized performance and quality benchmarks

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


When starting a backend, `inference-manage.sh` prints the resolved model
runtime parameters before launch, including context window, batching limits,
sequence concurrency, memory utilization, dtype, quantization, and optional
offload or speculative-decoding settings. Values supplied through environment
variables are reflected in the summary.
Flags combine. Set `UPSTREAM=1` in the config block to default to upstream vLLM, or use `start upstream` for a one-off run. To change the model, edit `MODEL` at the top of the script. The settings table printed on startup summarizes the active configuration. Lower-level launchers:

```bash
./rhaii-universal.sh [--api-key KEY]                    # run the backend directly (env RHAII_UPSTREAM=1 for upstream)
HF_HUB_OFFLINE=0 HF_TOKEN=hf_xxx ./model-downloader.sh  # cache a model only
```

## Benchmarking

Install the benchmark images once, then benchmark the local backend or a remote
server. `--url` accepts a full HTTP(S) origin or a bare IP address; a bare IP
uses port `8000`.

```bash
./tools/benchmark.sh install
./tools/benchmark.sh benchmark                         # http://127.0.0.1:8000
./tools/benchmark.sh benchmark --url 192.0.2.10       # http://192.0.2.10:8000
./tools/benchmark.sh quality --url https://api.example.com --model <served-model-name>
```

Set `BENCHMARK_API_KEY` when the remote server requires bearer authentication.
The `quality` command runs the TrustyAI LM-Eval-compatible `lm_eval` harness
in a Podman container based on Red Hat UBI 9. Its default deterministic suite
is HellaSwag, ARC-Challenge, GSM8K, and MMLU; the server must support
completion logprobs. Override tasks with `BENCHMARK_QUALITY_TASKS`, or use
`BENCHMARK_QUALITY_LIMIT` to cap examples per task for a smoke test.

Performance results persist on the host in `logs/benchmarks` by default
with datetime-stamped names such as `benchmarks-20260830-143012.csv` and
`benchmarks-20260830-143012.json`. Quality results persist in
timestamped directories below `logs/benchmarks/quality`. Set
`BENCHMARK_RESULTS_DIR` to use another host directory; it is mounted into
each benchmark container at `/results`.

## Models

`MODEL` / `MODEL_KEY` is a catalog key, or set a raw Hugging Face id directly:

| key | model id |
|---|---|
| `deepseek_r1_qwen_14b_awq` | `casperhansen/deepseek-r1-distill-qwen-14b-awq` |
| `qwen3_4b` | `Qwen/Qwen3-4B-Instruct-2507` |
| `qwen3_14b` | `RedHatAI/Qwen3-14B-quantized.w4a16` |
| `qwen38_27b` | `Qwen/Qwen3.8-27B` (27B BF16; T4 needs CPU offload, upstream vLLM 0.27.1) |
| `qwen38_27b_int4` | `RedHatAI/Qwen3.8-27B-INT4` (T4-tuned W4A16; upstream vLLM 0.27.1) |
| `granite_8b` | `ibm-granite/granite-3.3-8b-instruct` |
| `llama31_8b` | `meta-llama/Llama-3.1-8B-Instruct` |
| `whiterabbit_7b_awq` | `solidrust/WhiteRabbitNeo-7B-v1.5a-AWQ` |

Per-model quantization and memory defaults live in `rhaii-universal.sh`.

### Qwen3.8-27B INT4 on one Tesla T4

`qwen38_27b_int4` defaults to the verified single-T4 profile: float16 compute, text-only mode, one sequence, 16,384-token context, grouped prefetch offload, native MTP speculative decoding, and fine-grained prefix caching. It requires a dedicated T4 because `GPU_MEMORY_UTILIZATION=0.99`.

```bash
MODEL_KEY=qwen38_27b_int4 RHAII_UPSTREAM=1 RHAII_FOLLOW_LOGS=0 \
  ./rhaii-universal.sh
```

Pi/OMP exposes this server as `fedyagpt/qwen_27b`; the OpenAI API model id is `qwen_27b`.

On the repository's 300-token benchmark (`bench.py`), the cache-warm profile measured 3.70 output tokens/s and 1.23 s TTFT. A 15,433-token retrieval prompt completed successfully in about three minutes. First startup can take several minutes while vLLM compiles kernels; later starts reuse the cache.

To suppress thinking generation rather than only hide it after generation, OpenAI chat requests can include:

```json
{"chat_template_kwargs":{"enable_thinking":false}}
```

The tuning variables remain overridable: `GPU_MEMORY_UTILIZATION`, `MAX_MODEL_LEN`, `MAX_NUM_BATCHED_TOKENS`, `MAX_NUM_SEQS`, `RHAII_LANGUAGE_MODEL_ONLY`, `RHAII_ENABLE_PREFIX_CACHING`, `RHAII_PREFIX_MATCH_UNIT`, `RHAII_OFFLOAD_GROUP_SIZE`, `RHAII_OFFLOAD_NUM_IN_GROUP`, `RHAII_OFFLOAD_PREFETCH_STEP`, and `RHAII_SPEC_CONFIG`.

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
