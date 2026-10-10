# NInfer 3× RTX 3090 (PCIe P2P)

This fork extends NInfer with a native Linux graph-parallel route for **three RTX 3090s
connected over PCIe P2P (no NVLink required)**. Dense Qwen3.8-27B and Qwen3.6-27B
groupwise models physically split their packed MLP weights across all three cards,
execute all three shards concurrently, and reduce the partials over PCIe P2P DMA.
Enable it at runtime with `--devices PRIMARY,SECONDARY,TERTIARY`; the same `sm_86`
build continues to support ordinary single-card execution with `--device PRIMARY`
and the two-card graph split with `--devices PRIMARY,SECONDARY`.

The target platform is NASoWaffle: EPYC 74F3, 3× RTX 3090 24 GB, driver 610.43.02
with the aikitoria P2P modules. All six directed P2P pairs measure **26 GB/s**
(uniform, no asymmetry). The intent was to beat the llama.cpp tensor-split
baseline (qwen38-27b @ llama-swap :8088) on decode; the [measured results](#benchmark-results)
show the graph-parallel approach does not reach that bar over PCIe P2P, and the
fork is closed at that state.

Community project, maintained on a best-effort basis. Issues and PRs are very
welcome, but support and feature requests are not guaranteed.

This fork targets `sm_86`. Blackwell-only NVFP4/W4A4 execution is unavailable.

Base NInfer-3090 release notes: [v0.6.1](RELEASE_NOTES_0.6.1.md).

## 3× RTX 3090 graph parallelism

The dense 27B groupwise targets split every text-layer MLP across three
peer-connected SM86 devices. All three GPUs run their gate/up and down-projection
shards concurrently; the two secondary partials are reduced into the primary
residual over PCIe P2P. Prompt prefill, ordinary decode, and MTP verification
all use the split. This is a physical weight split, not three complete model
replicas.

```bash
./build-sm86/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Explain graph parallelism over PCIe P2P." \
  --max-context 8192 --max-new 256 --kv-dtype int8 \
  --devices 0,1,2
```

The three listed CUDA devices must have the same compute capability and
bidirectional peer access. CUDA device order can differ from `nvidia-smi`
order, especially when a display card is also installed; use the probe in the
[Linux guide](docs/rtx-3090-linux.md) and select the three CUDA indices whose
PCI bus IDs identify the 3090s.

### Weight split

| Mode | MLP intermediate split | KV layers | MTP KV |
|---|---|---|---|
| 1× (single card) | — | all 16 | local |
| 2× (`--devices 0,1`) | 8,192 / 9,216 | 5 / 11 | secondary |
| **3×** (`--devices 0,1,2`) | **6,144 / 6,144 / 5,120** | **5 / 6 / 5** | **tertiary** |

All three split boundaries are 1024-aligned. Attention projections, embeddings,
norms, and the output head stay on the primary (rank 0). Paged KV is split by
full-attention layer; MTP KV lives on the last rank in both 2× and 3× modes.

### Per-layer execution flow (3×)

1. Rank 0 launches its gate/up + down GEMM on the 6,144-row shard.
2. Hidden state fans out to rank 1 and rank 2 via `cudaMemcpyPeerAsync`
   (direct from rank 0; no relay).
3. Ranks 1 and 2 run their 6,144-row and 5,120-row GEMMs concurrently.
4. Both partials return to rank 0 via `cudaMemcpyPeerAsync`.
5. `residual_add_three_launch` sums primary + secondary + tertiary into the
   residual (scalar 3-input kernel; the reduction is memory-bound at BF16
   tensor sizes).

CUDA Graph capture is disabled for the cross-device schedule; an eager
representative round is warmed at startup instead. Single-card execution
retains CUDA Graphs.

The 35B-A3B and 27B NVFP4 targets currently reject `--devices`.

See [GRAPH-MODE-README.md](GRAPH-MODE-README.md) for compile steps, startup
memory peaks, and serving guidance.

### Serving

To serve concurrent OpenAI/Anthropic-compatible requests across all three cards:

```bash
./build-sm86/apps/ninfer-serve models/qwen3_8_27b.ninfer \
  --host 127.0.0.1 --port 8080 \
  --max-context 8192 --kv-capacity auto --kv-dtype int8 \
  --max-concurrency 8 --max-pending-requests 32 \
  --devices 0,1,2
```

The 2× path remains fully supported with `--devices 0,1`.

## P2P bandwidth (measured)

Measured on NASoWaffle (EPYC 74F3, driver 610.43.02 + aikitoria, CUDA 12.8.1):

| Pair | BW (GB/s) | Latency (µs) |
|---|---:|---:|
| 0→1 | 26.0 | 5.5 |
| 0→2 | 26.4 | 5.8 |
| 1→0 | 26.2 | 5.2 |
| 1→2 | 26.4 | 5.2 |
| 2→0 | 26.4 | 5.5 |
| 2→1 | 26.4 | 5.4 |

All pairs ≥ 25 GB/s, uniform (±0.4 GB/s spread), no asymmetry. Topology is
all NODE/PHB (CPU root complex) — no switch-level paths, but the EPYC 74F3
fabric handles all three equally. Kernel params: `iommu=pt`,
`pcie_acs_override=downstream`.

## Build

No GPU in the development container; build in a Docker container on the host
(Unraid shell):

```bash
docker run --rm --gpus all \
  -v /mnt/user/appdata/hermes-agent/workspace/git/ninfer-3x-3090:/src -w /src \
  nvidia/cuda:12.8.1-devel-ubuntu24.04 bash -c '
    apt-get update -qq && apt-get install -y -qq cmake ninja-build gcc-13 g++-13 \
      pkg-config python3 \
      libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libcurl4-openssl-dev &&
    cmake -S . -B build-sm86 -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=86 -DNINFER_BUILD_APPS=ON \
      -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=ON &&
    cmake --build build-sm86 --parallel 4'
```

The `build-sm86/` directory persists on the host filesystem across `--rm`
container runs, so incremental rebuilds are fast.

### Focused tests

```bash
docker run --rm --gpus all \
  -v /mnt/user/appdata/hermes-agent/workspace/git/ninfer-3x-3090:/src -w /src \
  nvidia/cuda:12.8.1-devel-ubuntu24.04 bash -c '
    apt-get update -qq && apt-get install -y -qq cmake ninja-build gcc-13 g++-13 \
      pkg-config python3 \
      libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libcurl4-openssl-dev &&
    cmake --build build-sm86 --parallel 4 &&
    ctest --test-dir build-sm86 -R "ninfer_(device|graph_shard|artifact_materialization|kv_capacity)" \
      --output-on-failure'
```

### Smoke test (3×, requires free VRAM)

```bash
./build-sm86/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Hello" --max-new 8 --devices 0,1,2
```

Stop llama-swap (or any other GPU workload) before running the smoke test —
the 27B artifact needs ~54 GB across the three cards.

## Benchmark results

Measured on NASoWaffle (EPYC 74F3, PCIe P2P 26 GB/s), Q4-groupwise artifact,
INT8 KV cache, MTP3, max context 2048:

| System | Split | Decode (t/s) | Prefill 512 (t/s) | Prefill 2048 (t/s) | Spec acc |
|---|---|---:|---:|---:|---:|
| NInfer 2× (original fork, `--devices 0,1`) | graph-parallel | 45.26 | 1,092 | 1,110 | 37.7% |
| **NInfer 3× (this fork, `--devices 0,1,2`)** | graph-parallel | **48.06** | **1,185** | **1,192** | 36.6% |
| llama.cpp `swift38-27b` (Q8_0, `--tensor-split 0.30,0.36,0.34`, MTP3) | tensor-split | ~67 (4K ctx) | — | ~1,470 (20K ctx) | MTP3 |

The 3× port is a small win over the 2× fork (+6% decode, +8% prefill), but
neither graph-parallel variant reaches the llama.cpp tensor-split baseline:
tensor-parallel decode is ~40% faster, and the gap is the same with 2 and 3
GPUs. Graph-parallel MLP splitting was designed for NVLink (~300 GB/s); at
26 GB/s the per-layer sequential `cudaMemcpyPeerAsync` round-trips
(2 copies/layer in 2×, 4 in 3×, × 48 layers) dominate batch-1 decode, and the
primary rank carries all attention plus its MLP shard, capping how much work
adding a third card removes.

**Verdict:** the 3× port is correct and works, but graph parallelism over
PCIe P2P is not competitive with tensor parallelism on this hardware. This
fork is closed in its current state. For serving on NASoWaffle, the
llama.cpp tensor-split remains the faster option.

Benchmark commands:

```bash
# NInfer 3×
./build-sm86/bench/ninfer_bench --weights models/qwen3_8_27b.ninfer \
  -p 512,2048 -n 256 -r 3 --warmup 1 --kv-dtype int8 \
  --mtp-draft-tokens 3 --lm-head-draft --devices 0,1,2

# NInfer 2× (original fork, same command with --devices 0,1)

# llama.cpp
python3 llama/benchmarks/llama_bench.py --models swift38-27b --bench throughput,ttft
```

## Choose a platform

| Platform | Delivery | Guide |
|---|---|---|
| Linux | Docker image or native source build | [Linux build guide](docs/rtx-3090-linux.md) |
| Windows 11 | Prebuilt release archive (single-card only) | [Windows guide](docs/rtx-3090-windows.md) |

### Linux

The Dockerfile gives the shortest build path on Bazzite and other Linux
distributions:

```bash
docker build --tag ninfer-3090:sm86 .
```

The Linux guide contains the GPU check, native Ubuntu build, model mount,
server command, and Bash launchers. Native Linux is the qualified multi-GPU
path; this repository does not currently publish a prebuilt Linux archive.

### Windows 11

The inherited Windows release is the base project's single-card distribution.
The multi-GPU graph route in this fork is currently Linux-only.

## Single-card results (inherited from 2× fork)

Qwen3.8-27B is validated from one through eight simultaneous users on a single
RTX 3090. ReplaySSM cuts the memory cost of speculative decoding, allowing the
faster MTP3 mode to remain enabled at C8.

| Cohort | Total output | End-to-end throughput | Decode throughput | MTP acceptance | Mean TTFT | Peak VRAM |
|---:|---:|---:|---:|---:|---:|---:|
| C1 | 1,024 tokens | **70.19 tok/s** | **71.00 tok/s** | 61.13% | 149 ms | 19,641 MiB |
| C2 | 2,048 tokens | **89.43 tok/s** | **90.66 tok/s** | 59.66% | 262 ms | 20,022 MiB |
| C4 | 4,096 tokens | **97.89 tok/s** | **100.28 tok/s** | 59.63% | 538 ms | 20,641 MiB |
| C8 | 8,192 tokens | **161.28 tok/s** | **165.33 tok/s** | 56.84% | 1,215 ms | 22,138 MiB |

On one RTX 3090, Qwen3.8-27B supports a measured **171K-token INT8 context**
with the standard 1 GiB safety headroom. The optional RotorQuant `rk8v4` cache
raises this to **226K tokens with 1 GiB headroom**.

## Capabilities

- Native 3× RTX 3090 PCIe P2P graph parallelism on Linux for dense 27B
  groupwise artifacts (6144/6144/5120 MLP split, 5/6/5 KV layers).
- 2× RTX 3090 graph parallelism preserved (`--devices 0,1`).
- Single-card execution with CUDA Graphs (`--device 0`).
- OpenAI Chat Completions, Responses, and Anthropic-compatible APIs.
- ReplaySSM and MTP3 for higher throughput without exceeding 24 GB VRAM.
- `low`, `medium`, and `xhigh` reasoning modes.
- Qwen3.8 image understanding with ReplaySSM and MTP3.
- Prefix reuse for faster repeated or shared prompts.
- Qwen3.6-35B image understanding with a guarded 32K profile.

## Supported artifacts

| Model | Artifact | Size | Notes |
|---|---|---:|---|
| Qwen3.6-35B-A3B v1 | [pinned compact artifact](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/tree/c8b8c1c0df4c74df3c190c6aa3a7f24dc614721c) | 20.84 GiB | **Recommended for RTX 3090; text C1-C6 at 4K and vision C1 at 32K** |
| Qwen3.6-35B-A3B v2 | [current upstream artifact](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | 21.22 GiB | Reader supported by v0.5+; includes DFlash payload |
| Qwen3.6-27B | [groupwise artifact](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | 16.29 GiB | Single-card supported; eligible for 2×/3× MLP split |
| **Qwen3.8-27B** | [official NInfer groupwise artifact](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | 16.96 GiB | **Primary 3× target; single-card validated at C1-C8/MTP3** |

## Models and platform support

Linux users build the applications from source or use the Docker image.
Windows users can use the prebuilt archive (single-card only).

Multi-GPU graph mode requires 2–3 matching SM86 cards with bidirectional CUDA
peer access. NVLink is not required; PCIe P2P DMA is sufficient (26 GB/s
measured on the target platform). It is currently qualified on Linux. A
display GPU is supported; discover the CUDA ordinals with
`tools/multi_gpu_probe.cu` instead of assuming `nvidia-smi` table order.

Download the [official Qwen3.8 artifact](https://huggingface.co/neroued/Qwen3.8-27B-NInfer)
as `models/qwen3_8_27b.ninfer`.

## Qwen3.8 reasoning effort

Qwen3.8-27B supports distinct reasoning-effort modes. `medium` uses the
model's normal thinking prompt. `xhigh` injects the checkpoint's extended
deliberation instruction. This is a real prompt-template change, not a
sampling alias.

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

OpenAI Responses uses `"reasoning": {"effort": "xhigh"}`. Anthropic Messages
uses `"output_config": {"effort": "xhigh"}`. For the native CLI, pass
`--reasoning-effort low|medium|xhigh`; use `--no-thinking` to disable
reasoning.

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

The C number is the maximum number of requests NInfer can run together. C1
favors one interactive user; C8 can combine up to eight active requests into
each GPU step for much higher total output.

Follow-up requests do not need to arrive at the same instant. When a running
request finishes, the next waiting request can join at a safe generation
boundary. Empty or finished lanes are skipped, so a C8 server also works
normally with only one, two, or four active users.

This is deliberately more bounded than datacenter-style dynamic batching. The
maximum number of users and GPU memory are chosen when the server starts. In
return, memory use stays predictable on a 24 GB card. Single-card execution
reuses CUDA Graphs; multi-card execution uses the warmed eager cross-device
schedule.

## Current limits

- One process owns one model on one RTX 3090, or one dense-27B graph split
  across an ordered set of 2–3 peer-connected SM86 GPUs.
- Concurrency is fixed at startup and limited to 1-8 by the API.
- The shared KV pool is fixed at startup and is not divided statically among
  request lanes.
- In multi-GPU mode, every lane reserves matching logical page counts in the
  per-rank KV pools; admission is limited by whichever physical pool fills
  first.
- This is bounded small-scale batching, not preemptive large-scale continuous
  batching.
- Multi-GPU execution is limited to the dense 27B groupwise Q4/Q5 MLP graph;
  there is no CPU/GPU weight offload, 35B graph split, or NVFP4 graph split.
- Tool calls are returned to the client but are not executed by NInfer.
- NVFP4 A4 and TMA kernels require Blackwell and are unavailable on SM86.
- The paged runtime exposes BF16, INT8, and experimental opt-in `rk8v4` KV.
  INT8 remains the quality-default path.

## Upstream

This repository is based on [devon-caron/ninfer-dual-3090-nvlink](https://github.com/devon-caron/ninfer-dual-3090-nvlink),
which extends [Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090)
with dual-GPU graph-parallel MLP splitting. This fork generalizes the 2-rank
graph-parallel path to 3 ranks over PCIe P2P (no NVLink required): N-endpoint
`DeviceContext`, 3-way MLP split (6144/6144/5120), 3-way KV cache (5/6/5),
`residual_add_three` reduction, N-way weight materialization, and `--devices`
accepting 1–3 indices. The 2× path remains fully functional.

## Contributors

- [airtonix](https://github.com/airtonix) added Linux and Docker build and
  release support in [PR #1](https://github.com/Don-Chad/ninfer-3090/pull/1).

## License

Apache License 2.0. See [LICENSE](LICENSE).
