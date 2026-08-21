# NInfer Dual RTX 3090 NVLink

This fork extends NInfer-3090 with a native Linux graph-parallel route for **two RTX 3090s joined by
NVLink**. Dense Qwen3.8-27B and Qwen3.6-27B groupwise models physically split their packed MLP
weights across both cards, execute both shards concurrently, and reduce the secondary result over
NVLink. Enable it at runtime with `--devices PRIMARY,SECONDARY`; the same `sm_86` build continues to
support ordinary single-card execution with `--device PRIMARY`.

Qwen3.8-27B is the primary qualified dual-GPU target. The engine loads its native `.ninfer`
artifact, serves OpenAI- and Anthropic-compatible APIs, and supports paged KV, compatible-prefix
reuse, MTP speculative decoding, reasoning-effort control, ReplaySSM state transactions, and
concurrent cohorts through **C8**. Cross-device execution uses an eager graph schedule because CUDA
Graph capture is disabled for the multi-device round; single-card execution retains CUDA Graphs.

Community project, maintained on a best-effort basis. Issues and PRs are very welcome, but support
and feature requests are not guaranteed.

On one RTX 3090, Qwen3.8-27B supports a measured **171K-token INT8 context** with the standard
1 GiB safety headroom. The optional RotorQuant `rk8v4` cache raises this to **226K tokens with
1 GiB headroom**, or **247,872 tokens (about 248K)** in a tightly packed 300 MiB-headroom profile.
RotorQuant is an opt-in feature; INT8 remains the default because `rk8v4` is lossy.

This fork targets `sm_86`. Blackwell-only NVFP4/W4A4 execution is unavailable.

The goal is to make the fastest practical Qwen inference stack for dual RTX 3090 systems. PRs and
testing on other two-card NVLink configurations are welcome.

Base NInfer-3090 release notes: [v0.6.1](RELEASE_NOTES_0.6.1.md).

## Dual RTX 3090 graph parallelism

The dense 27B groupwise targets split every text-layer MLP across two peer-connected SM86 devices.
Both GPUs run their gate/up and down-projection shards concurrently; the secondary partial is
reduced into the primary residual over NVLink. Prompt prefill, ordinary decode, and MTP verification
all use the split. This is a physical weight split, not two complete model replicas.

```bash
./build-sm86/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Explain NVLink graph parallelism." \
  --max-context 8192 --max-new 256 --kv-dtype int8 \
  --devices 1,2
```

The two listed CUDA devices must have the same compute capability and bidirectional peer access.
CUDA device order can differ from `nvidia-smi` order, especially when an Ada display card is also
installed; use the probe in the [Linux guide](docs/rtx-3090-linux.md#dual-3090-nvlink) and select
the two CUDA indices whose PCI bus IDs identify the 3090s. The route physically shards the packed
main-model Q4/Q5 MLP weights 8,192/9,216. Attention projections and unsplit weights stay on the
primary, while paged KV is split by full-attention layer: five layers remain primary-local and
eleven layers, plus optional MTP KV, live on the secondary. CUDA Graph capture is disabled for this
cross-device schedule and an eager representative round is warmed at startup instead. The 35B-A3B
and 27B NVFP4 targets currently reject `--devices`.
See [GRAPH-MODE-README.md](GRAPH-MODE-README.md) for compile steps, startup memory peaks, and C2-C8
serving guidance.

To serve concurrent OpenAI/Anthropic-compatible requests across both cards:

```bash
./build-sm86/apps/ninfer-serve models/qwen3_8_27b.ninfer \
  --host 127.0.0.1 --port 8080 \
  --max-context 8192 --kv-capacity auto --kv-dtype int8 \
  --max-concurrency 8 --max-pending-requests 32 \
  --devices 1,2
```

### Measured dual-card results

The graph path was qualified on two RTX 3090s connected by NVLink, with a third RTX 4070 SUPER used
as the display card. Measurements used the Qwen3.8-27B groupwise-int artifact, INT8 KV, greedy
decoding, and CUDA runtime 13.1. The exact-count table used three measured repetitions after one
warmup. These throughput measurements predate KV-by-layer sharding and are retained as the
MLP-split baseline pending a full post-sharding benchmark rerun.

| Exact-count workload | One RTX 3090 | Dual RTX 3090 | Change |
|---|---:|---:|---:|
| 512-token prefill | 975.62 tok/s | **1,378.72 tok/s** | **+41.3%** |
| 2,048-token prefill | 997.66 tok/s | **1,407.94 tok/s** | **+41.1%** |
| 256-token decode | 38.85 tok/s | **45.63 tok/s** | **+17.5%** |
| 2,048 prefill + 256 decode | 986.95 / 38.30 tok/s | **1,393.78 / 45.31 tok/s** | **+41.2% / +18.3%** |

A post-sharding one-repetition smoke benchmark (after one warmup) measured 512-token prefill at
974.37 tok/s single-card versus 1,377.97 tok/s dual-card (**+41.4%**), and 64-token ordinary decode
at 38.82 tok/s versus 44.85 tok/s (**+15.5%**). Run a multi-repetition local benchmark before using
these smoke figures as publication results.

The ordinary dual-card layout held **11.09 GiB** of weights on the primary and **4.83 GiB** on the
secondary, compared with 15.92 GiB on one card. KV pages remain device-local but form one logical
aggregate entitlement across both layer pools. With native 262K per-request context, C8, INT8 KV,
and automatic sizing, the current layout resolved **803,328 aggregate resident tokens** while
retaining the default 1 GiB headroom on both cards (10.30 GiB primary and 17.47 GiB secondary
runtime reservations).

Concurrent HTTP requests also scale through the shared graph:

| Concurrency | Single-card aggregate | Dual-card aggregate | Change |
|---:|---:|---:|---:|
| C2 | 63.85 tok/s | **64.37 tok/s** | +0.8% |
| C4 | 102.44 tok/s | **114.82 tok/s** | +12.1% |
| C8 | 127.05 tok/s | **145.97 tok/s** | +14.9% |

Each concurrency measurement generated exactly 256 tokens per request with prefix reuse disabled.
At C8, the dual route delivered 3.20 times its single-request exact-count decode rate. Deterministic
ordinary and MTP3 generation produced identical text and identical MTP acceptance statistics in
single- and dual-card modes.

## Choose a platform

| Platform | Delivery | Guide |
|---|---|---|
| Linux | Docker image or native source build | [Linux build guide](docs/rtx-3090-linux.md) |
| Windows 11 | Prebuilt release archive | [Windows guide](docs/rtx-3090-windows.md) |

### Linux

The Dockerfile gives the shortest build path on Bazzite and other Linux distributions:

```bash
docker build --tag ninfer-3090:sm86 .
```

The Linux guide contains the GPU check, native Ubuntu build, model mount, server command, and Bash
launchers. Native Linux is the qualified dual-3090 path; this repository does not currently publish
a prebuilt Linux archive.

### Windows 11

The inherited Windows release is the base project's single-card distribution. The dual-NVLink
route in this fork is currently qualified on Linux only.

1. Download and unzip the latest [Windows release](https://github.com/Don-Chad/ninfer-3090/releases/latest).
2. Double-click `download-qwen38.bat` to download the model. Interrupted downloads resume.
3. Double-click one launcher:

| Launcher | Best for |
|---|---|
| `run-qwen38-c1.bat` | One interactive user, lowest latency, up to 64K context |
| `run-qwen38-c8.bat` | Multiple users or agents, highest aggregate throughput, 8K context |
| `run-qwen38-vision.bat` | Qwen3.8 image understanding, one user, 32K context, MTP3 |
| `run-qwen36-35b-vision.bat` | Image understanding with Qwen3.6-35B-A3B, one user, 32K context |

The API is then available at `http://127.0.0.1:8080/v1`. The Windows archive includes the required
applications and DLLs.

## Qwen3.8-27B single-card results

Qwen3.8-27B is validated from one through eight simultaneous users. ReplaySSM cuts the memory cost
of speculative decoding, allowing the faster MTP3 mode to remain enabled at C8. The table below is
the new sustained test: every request generated 1,024 tokens with CUDA Graphs enabled.

The prompts were **29-34 input tokens** and the server's maximum context window was **8,192 tokens
per request**. Each measured sequence therefore reached roughly 1,053-1,058 tokens including its
generated output. This is a long-output/decode benchmark, not an 8K-prompt or long-prefill test.
C1-C4 used an 8,192-token shared KV pool; C8 used 16,384 tokens so all eight requested outputs
could be admitted simultaneously.

| Cohort | Total output | End-to-end throughput | Decode throughput | MTP acceptance | Mean TTFT | Peak VRAM |
|---:|---:|---:|---:|---:|---:|---:|
| C1 | 1,024 tokens | **70.19 tok/s** | **71.00 tok/s** | 61.13% | 149 ms | 19,641 MiB |
| C2 | 2,048 tokens | **89.43 tok/s** | **90.66 tok/s** | 59.66% | 262 ms | 20,022 MiB |
| C4 | 4,096 tokens | **97.89 tok/s** | **100.28 tok/s** | 59.63% | 538 ms | 20,641 MiB |
| C8 | 8,192 tokens | **161.28 tok/s** | **165.33 tok/s** | 56.84% | 1,215 ms | 22,138 MiB |

C1 is the responsive choice for a single user. C8 delivers **2.3x the total throughput** when
several requests are active. The C8 long-output test uses a 16K shared KV pool so all eight
1,024-token responses can be admitted together.

### Prompt-processing speed

Prompt processing was tested separately with **4,362 fresh input tokens per request**, an 8,192-token
per-request context window, 512-token prefill chunks, INT8 KV, ReplaySSM/MTP3, CUDA Graphs, and
prefix reuse disabled. Each request generated only 16 tokens so the run measures prefill rather
than long decode.

| Cohort | Total fresh input | Aggregate prefill | Active-prefill speed | Mean TTFT | Peak VRAM |
|---:|---:|---:|---:|---:|---:|
| C1 | 4,362 tokens | **861.51 tok/s** | 893.98 tok/s | 4,893 ms | 19,114 MiB |
| C2 | 8,724 tokens | **853.86 tok/s** | 883.95 tok/s | 7,478 ms | 19,697 MiB |
| C4 | 17,448 tokens | **847.26 tok/s** | 874.49 tok/s | 12,692 ms | 20,894 MiB |
| C8 | 34,896 tokens | **844.10 tok/s** | 870.94 tok/s | 23,028 ms | 23,207 MiB |

`Aggregate prefill` is total fresh input tokens divided by the complete request-wave time, so it is
the user-facing throughput number. NInfer currently processes one long prefill at a time; cohort
batching accelerates decode, but does not multiply prompt ingestion. Consequently C1-C8 remain near
844-862 input tok/s while queued requests increase mean TTFT. `Active-prefill speed` excludes queue
waiting and measures only the server's recorded prefill phase.

### Optional RotorQuant KV for longer context

`rk8v4` is an **experimental, opt-in** KV-cache mode for Qwen3.8-27B. INT8 remains the default and
the recommended quality setting. RotorQuant applies the same normalized transform to queries and
keys, rotates values before four-bit storage, and reverses the value transform after attention.
This reduces the V-cache footprint while keeping keys at eight bits.

Add `--kv-dtype rk8v4` to either `ninfer.exe` or `ninfer-serve.exe`. For example, from Command
Prompt:

```bat
ninfer-serve.exe qwen3_8_27b.ninfer --max-context 131072 --kv-capacity auto --max-concurrency 1 --prefill-chunk 1024 --kv-dtype rk8v4 --spec mtp --draft-tokens 3 --lm-head-draft
```

On the development RTX 3090, C1 with MTP and CUDA Graphs disabled fit a **226,560-token** logical
context with the normal 1 GiB automatic-sizing headroom; 226,624 was rejected. The comparable INT8
boundary was 171,648 tokens, so `rk8v4` increased measured allocatable context by about **32%**.
This is an allocation plus short-execution boundary, not a full 226K prefill quality claim.

A second explicit-reservation test reduced operational headroom to approximately 300 MiB.
`rk8v4` successfully started and generated at **247,872 tokens**, leaving **302.97 MiB** physically
free after startup. The next 64-token page boundary, 247,936, left only 274.58 MiB because it crossed
a CUDA allocation granularity. Therefore 247,872 is the measured maximum that retained at least
300 MiB—not 2x INT8 capacity, but about **44% more** than the 171,648-token INT8/1 GiB baseline.
This tight profile leaves little tolerance for other GPU users and is not the recommended default.

The matched 1,024-token hard coding test reduced 4K KV payload from 140.38 MiB to 106.35 MiB and
decoded at 84.07 tok/s versus 85.72 tok/s for INT8. Quality was not equivalent: the RotorQuant
answer introduced a faulty nested-rollback design that the INT8 answer avoided, and both answers
hit their output limit. Use `rk8v4` only when its context gain is worth task-specific quality
validation; do not use it as the default for correctness-sensitive work.

### Qwen3.8 vision

The same Qwen3.8 artifact supports images. Start the server with `--vision`, MTP3, INT8 KV, and a
32K maximum context. The Windows archive includes `run-qwen38-vision.bat` for this profile.

A 1,920×1,080 image expanded to 2,074 prompt tokens and was read correctly. Measured TTFT was
3.29 seconds, decode reached 98.1 tok/s, MTP acceptance was 96.7%, and startup retained 2.16 GiB
free VRAM. The artifact also declares multi-image and video support; this release test directly
validated a single image.

## Qwen3.6-35B-A3B single-card results

Measured with the compact 20.84 GiB 35B-A3B artifact, a 4K shared INT8 group-64 paged KV pool,
CUDA Graphs, MTP3, greedy decoding, and no competing GPU workload:

| Concurrent requests | 128 output tokens each | Observed VRAM |
|---:|---:|---:|
| 1 | 162.7 aggregate tok/s | 22,427 MiB |
| 2 | 267.9 aggregate tok/s | 22,743 MiB |
| 4 | 366.2 aggregate tok/s | 23,377 MiB |
| 6 | 383.4 aggregate tok/s | 24,038 MiB |
| 8 | rejected at startup | about 503 MiB over the safe reservation limit |

A longer 512-token-per-request check reached **286.8 tok/s at C1** and **399.1 aggregate tok/s at
C2**. These short-prompt measurements include request-level timing and are not directly comparable
to v0.3.1's 1,500-token adaptive prompt-lookup benchmark.

Compatible-prefix reuse was validated end to end: a repeated 26-token prompt reused 24 tokens,
reducing measured prefill from 371 ms to 10 ms.

### Qwen3.6-35B vision

The compact 35B artifact includes its vision encoder and accepts images through the same OpenAI-
compatible API. Start the server with `--vision` and leave speculative decoding disabled. The
Windows archive includes `run-qwen36-35b-vision.bat` for this profile.

The safe RTX 3090 profile is **one request, 32K maximum context, INT8 KV, vision enabled, and MTP
disabled**. A current v0.6 test processed three 1,920×1,080 images correctly. Each image expanded
to a 2,081-token prompt; engine TTFT was 3.76–4.07 seconds, decode was about 159 tok/s, and peak
VRAM was 23,944 MiB. This leaves little room for another GPU workload.

MTP is intentionally off for this profile. At 32K, speculative recurrent state would exceed the
3090 memory budget; KV compression alone does not recover enough memory. Text-only 35B profiles can
still use MTP3 as documented above.

## Capabilities

- Native dual-RTX-3090 NVLink graph parallelism on Linux for dense 27B groupwise artifacts.
- Native SM86 CLI and server applications for Linux and Windows single-card use.
- An inherited prebuilt Windows single-card archive with tested launchers.
- OpenAI Chat Completions, Responses, and Anthropic-compatible APIs.
- ReplaySSM and MTP3 for higher throughput without exceeding 24 GB VRAM.
- `low`, `medium`, and `xhigh` reasoning modes.
- Qwen3.8 image understanding with ReplaySSM and MTP3.
- Prefix reuse for faster repeated or shared prompts.
- Qwen3.6-35B image understanding with a guarded 32K profile.
- Windows one-user and eight-user launchers with safe tested defaults.

## Supported artifacts

| Model | Artifact | Size | Notes |
|---|---|---:|---|
| Qwen3.6-35B-A3B v1 | [pinned compact artifact](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/tree/c8b8c1c0df4c74df3c190c6aa3a7f24dc614721c) | 20.84 GiB | **Recommended for RTX 3090; text C1-C6 at 4K and vision C1 at 32K** |
| Qwen3.6-35B-A3B v2 | [current upstream artifact](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | 21.22 GiB | Reader supported by v0.5+; includes DFlash payload and is not the measured 3090 artifact |
| Qwen3.6-27B | [groupwise artifact](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | 16.29 GiB | Single-card supported; eligible for the dual Q4/Q5 MLP split |
| **Qwen3.8-27B** | [official NInfer groupwise artifact](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | 16.96 GiB | **Qualified on dual 3090s; single-card validated at C1-C8/MTP3** |

NInfer-3090 v0.5 and newer recognize both v1 and v2 container magic. The current 21.22 GiB v2
artifact contains additional DFlash weights and is not the artifact used for the published RTX
3090 concurrency results. The pinned compact v1 artifact keeps the measured model payload and
omits DFlash, providing the known 24 GB memory profile.

## Models and platform support

Linux users build the applications from source or use the Docker image. Windows users can use the
prebuilt archive, which includes the applications and required DLLs. Both platforms require an
RTX 3090 or RTX 3090 Ti and a recent NVIDIA driver.

Dual graph mode additionally requires two matching SM86 cards, bidirectional CUDA peer access, and
an NVLink bridge. It is currently qualified on Linux. A third display GPU is supported; discover
the CUDA ordinals with `tools/multi_gpu_probe.cu` instead of assuming `nvidia-smi` table order.

Download the [official Qwen3.8 artifact](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) as
`models/qwen3_8_27b.ninfer`. Windows users can run `download-qwen38.bat` instead.

For Qwen3.6-35B-A3B, the smaller
[pinned container-v1 artifact](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/tree/c8b8c1c0df4c74df3c190c6aa3a7f24dc614721c)
is recommended on a 24 GB card. Releases v0.5 and newer also read the larger container-v2 file.
An `artifact magic is not NInfer version 1` message means the executable is outdated, not that the
current model download is necessarily corrupt.

Developers can build from source on Windows or Linux. Windows uses Visual Studio 2022 and vcpkg.
Linux uses GCC 13 with system packages or the pinned vcpkg manifest. Both builds require CUDA 12.8
or newer and CMake 3.28 or newer.

See the [Windows build guide](docs/rtx-3090-windows.md) or the
[Linux build guide](docs/rtx-3090-linux.md). Ordinary Windows release users do not need these tools.

## Qwen3.8 reasoning effort

Qwen3.8-27B supports distinct reasoning-effort modes. `medium` uses the model's normal thinking
prompt. `xhigh` injects the checkpoint's extended deliberation instruction, asking it to validate
assumptions and consider alternatives. This is a real prompt-template change, not a sampling alias.

| Value | Qwen3.8 behavior |
|---|---|
| `none` | Disable thinking |
| `low` | Keep reasoning brief and focused |
| `medium` | Use normal Qwen3.8 thinking |
| `xhigh` | Use extended deliberation and verification |

OpenAI Chat Completions accepts a top-level `reasoning_effort` field:

```json
{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "Solve this carefully..."}],
  "reasoning_effort": "xhigh",
  "max_tokens": 4096
}
```

OpenAI Responses uses `"reasoning": {"effort": "xhigh"}`. Anthropic Messages uses
`"output_config": {"effort": "xhigh"}`. For the native CLI, pass
`--reasoning-effort low|medium|xhigh`; use `--no-thinking` instead of an effort to disable
reasoning. Chat Completions returns hidden reasoning separately as `message.reasoning_content`.

## Serving APIs

The server supports:

- OpenAI Chat Completions;
- OpenAI Responses Core with streaming and local continuation state;
- Anthropic Messages;
- compatible-prefix reuse;
- prompt-rendered function tools and parsed tool calls;
- bounded pending-request admission and JSONL request logs.

See [HTTP serving](docs/serving.md) and [CLI usage](docs/cli.md).

## How cohort batching works

The C number is the maximum number of requests NInfer can run together. C1 favors one interactive
user; C8 can combine up to eight active requests into each GPU step for much higher total output.

Follow-up requests do not need to arrive at the same instant. When a running request finishes, the
next waiting request can join at a safe generation boundary. Empty or finished lanes are skipped,
so a C8 server also works normally with only one, two, or four active users.

This is deliberately more bounded than datacenter-style dynamic batching. The maximum number of
users and GPU memory are chosen when the server starts. In return, memory use stays predictable on
a 24 GB card. Single-card execution reuses CUDA Graphs; dual-card execution uses the warmed eager
cross-device schedule.

## Current limits

- One process owns one model on one RTX 3090, or one dense-27B graph split across an ordered pair
  of peer-connected SM86 GPUs.
- Concurrency is fixed at startup and limited to 1-8 by the API; compact 35B fits C1-C6 and
  Qwen3.8-27B fits C8/8K with MTP3 through ReplaySSM.
- The shared KV pool is fixed at startup and is not divided statically among request lanes.
- In dual-27B mode, every lane reserves matching logical page counts in the primary five-layer and
  secondary eleven-layer KV pools; admission is limited by whichever physical pool fills first.
- This is bounded small-scale batching, not preemptive large-scale continuous batching.
- Multi-GPU execution is limited to the dense 27B groupwise Q4/Q5 MLP graph; there is no CPU/GPU
  weight offload, 35B graph split, or NVFP4 graph split.
- Tool calls are returned to the client but are not executed by NInfer.
- NVFP4 A4 and TMA kernels require Blackwell and are unavailable on SM86.
- The paged runtime exposes BF16, INT8, and experimental opt-in `rk8v4` KV. INT8 remains the
  quality-default path.

## Validation

The v0.6.0 Windows gate covered Qwen3.8 generation, materialization, request memory, admission,
paged KV, prefix reuse, speculative rounds, and SM86 W8 Linear paths.

The v0.6.1 Linux source gate completed all 245 Docker compile and link steps with CUDA 13.1 on
Ubuntu 24.04. The dual-GPU extension was then qualified with the real Qwen3.8-27B groupwise-int
artifact on two NVLinked RTX 3090s. Materialization, device-selection, reduction, attention,
paged-KV, capacity, Q4/Q5 graph-shard, and benchmark-contract tests passed. Ordinary and MTP3
sharded-KV generation completed with deterministic requested output, four simultaneous HTTP lanes
completed correctly, and the 803,328-token automatic pool survived warmup and retained-state
shutdown. The earlier MLP throughput and C2-C8 serving measurements above predate KV-by-layer
sharding and are retained as a baseline pending a full post-sharding benchmark rerun.

## Upstream

This repository is based on [Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090), which
ports [Neroued/ninfer](https://github.com/Neroued/ninfer) from RTX 5090/`sm_120a` to RTX
3090/`sm_86`. This fork adds compact dual-device materialization, Q4/Q5 MLP shard kernels, NVLink
activation/result exchange, primary/secondary memory accounting, dual-device CLI/server/benchmark
selection, concurrency qualification, and mixed-display-GPU discovery.

## Contributors

- [airtonix](https://github.com/airtonix) added Linux and Docker build and release support in
  [PR #1](https://github.com/Don-Chad/ninfer-3090/pull/1).

## License

Apache License 2.0. See [LICENSE](LICENSE).
