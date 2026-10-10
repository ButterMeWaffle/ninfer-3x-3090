# Multi-GPU Graph Mode Build Guide

This guide builds NInfer's graph-parallel path for 2–3 RTX 3090s connected over
PCIe P2P. NVLink is **not** required; the code uses `cudaMemcpyPeerAsync`
(CUDA P2P DMA) for all cross-device transfers and checks
`cudaDeviceCanAccessPeer` at startup. The same `sm_86` binaries support
single-GPU, 2-card, and 3-card execution; there is no separate graph-mode
CMake option. Graph mode is selected at runtime with `--devices CUDA_A`
(single), `--devices CUDA_A,CUDA_B` (2-card), or
`--devices CUDA_A,CUDA_B,CUDA_C` (3-card).

The current graph path supports dense Qwen3.6/Qwen3.8 27B groupwise `.ninfer`
artifacts. It physically shards every main-model MLP's packed Q4 gate/up and
Q5 down weights across the selected 3090s, executes all shards concurrently,
and reduces the partials over PCIe P2P. It also shards the growing paged KV
cache by attention layer. The split is used for prompt prefill, ordinary
decode, and MTP verification. The 35B-A3B and 27B NVFP4 targets are not
supported by this path.

## Requirements

- Linux; the commands below target Ubuntu 24.04.
- 2–3 RTX 3090 or RTX 3090 Ti cards with bidirectional CUDA peer access
  (NVLink optional; PCIe P2P is sufficient).
- An NVIDIA driver that supports the installed CUDA Toolkit and exposes P2P
  DMA. On systems without NVLink, the driver must include P2P BAR
  enablement (e.g. the aikitoria P2P modules for the 610.x driver series).
- CUDA Toolkit 12.8 or newer.
- CMake 3.28 or newer, Ninja, and GCC/G++ 13.
- Enough free memory during startup for the temporary packing layout described
  below.

Run every command from the repository root.

## 1. Check the physical GPUs and P2P

Before building, confirm that the 3090s are visible and that P2P is enabled:

```bash
nvidia-smi --query-gpu=index,pci.bus_id,name,memory.total --format=csv
nvidia-smi topo -m
```

For P2P bandwidth (pulls the CUDA devel image):

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  /usr/local/cuda/extras/demo_suite/p2pBandwidthLatencyTest
```

Read the "P2P DMA Bandwidth Matrix (GB/s)" — all directed pairs should be
non-zero. Target ≥ 15 GB/s per pair; ≥ 25 GB/s is comfortable. On the
reference NASoWaffle system (EPYC 74F3, driver 610.43.02 + aikitoria), all
six pairs measure **26 GB/s**.

Do not assume that a CUDA device index matches the index printed by the main
`nvidia-smi` table. On a system with a display card, CUDA may enumerate the
display card as device 0 even when `nvidia-smi` lists it elsewhere. A
CUDA-level check is included after the build.

## 2. Install native build dependencies

Install CUDA 12.8 or newer from NVIDIA, then install the host dependencies:

```bash
sudo apt-get update
sudo apt-get install --yes \
  build-essential gcc-13 g++-13 cmake ninja-build pkg-config python3 \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
  libcurl4-openssl-dev
```

Select GCC 13 for C++, C, and CUDA host compilation:

```bash
export CC=/usr/bin/gcc-13
export CXX=/usr/bin/g++-13
export CUDACXX=/usr/local/cuda/bin/nvcc
export CUDAHOSTCXX=/usr/bin/g++-13
```

Verify the toolchain before configuring:

```bash
cmake --version
"$CUDACXX" --version
"$CXX" --version
```

## 3. Configure the SM86 build

Configure a release build for the RTX 3090 architecture:

```bash
cmake -S . -B build-sm86 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_CUDA_COMPILER="$CUDACXX" \
  -DCMAKE_CUDA_HOST_COMPILER="$CUDAHOSTCXX" \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DNINFER_BUILD_APPS=ON \
  -DBUILD_TESTING=OFF \
  -DNINFER_BUILD_BENCHMARKS=OFF
```

Keep `CMAKE_CUDA_ARCHITECTURES=86`. Building for `89` targets the Ada display
card rather than the RTX 3090 kernels and cannot be used for the multi-GPU
graph path.

## 4. Compile

```bash
cmake --build build-sm86 --parallel 4
```

CUDA compilation can consume substantial host memory. Increase `--parallel`
only if the machine has enough RAM. A successful build produces:

```text
build-sm86/apps/ninfer
build-sm86/apps/ninfer-serve
```

Confirm that both applications start:

```bash
./build-sm86/apps/ninfer --help
./build-sm86/apps/ninfer-serve --help
```

## 5. Find the correct CUDA indices

Compile and run the repository's CUDA enumeration probe:

```bash
"$CUDACXX" tools/multi_gpu_probe.cu -o build-sm86/multi-gpu-probe
./build-sm86/multi-gpu-probe
```

Select the entries that:

1. Identify the RTX 3090s by name and PCI bus ID.
2. Report `sm=86`.
3. Report peer access as `1` in both directions.

For the reference NASoWaffle system (3× 3090, no display card):

```text
cuda=0 ... name=NVIDIA GeForce RTX 3090 sm=86 ...
cuda=1 ... name=NVIDIA GeForce RTX 3090 sm=86 ...
cuda=2 ... name=NVIDIA GeForce RTX 3090 sm=86 ...
peer 0->1=1  peer 1->0=1
peer 0->2=1  peer 2->0=1
peer 1->2=1  peer 2->1=1
```

Therefore its graph-mode device option is `--devices 0,1,2`. Use the indices
reported on your own machine; do not copy these numbers blindly. NInfer checks
that all selected devices have matching compute capability and bidirectional
peer access. Selecting a non-3090 and a 3090 is rejected before the model is
loaded.

The order matters:

- **Rank 0 (primary):** model execution, attention, embeddings, norms, output
  head, persistent linear-attention state, the first MLP shard, the first KV
  layer group, and the final MLP reduction.
- **Rank 1 (secondary):** the second MLP shard and the middle KV layer group.
- **Rank 2 (tertiary, 3-card only):** the third MLP shard and the last KV
  layer group, plus MTP KV when enabled.

## 6. Run a graph-mode smoke test

Replace the model path if the artifact is stored elsewhere:

```bash
# 3-card
./build-sm86/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "In one sentence, explain why the sky is blue." \
  --max-context 8192 \
  --max-new 128 \
  --kv-dtype int8 \
  --devices 0,1,2

# 2-card
./build-sm86/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "In one sentence, explain why the sky is blue." \
  --max-context 8192 \
  --max-new 128 \
  --kv-dtype int8 \
  --devices 0,1

# Single card (for comparison)
./build-sm86/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "In one sentence, explain why the sky is blue." \
  --max-context 8192 \
  --max-new 128 \
  --kv-dtype int8 \
  --device 0
```

Use deterministic or identical sampling settings when comparing output and
throughput. Initial graph-mode model loading takes longer because NInfer reads
the original packed arena onto the primary, repacks the selected MLP slices
over P2P, and then installs the compact primary arena. Compare steady-state
prefill and decode measurements after loading.

Stop any competing GPU workload (e.g. llama-swap) before the smoke test —
the 27B artifact needs ~17 GB per card in 3-card mode.

## How the weights and memory are split

### 3-card (6144 / 6144 / 5120)

For every main-model MLP, graph mode splits the 17,408 intermediate channels
across three ranks:

| Weight | Rank 0 (primary) | Rank 1 (secondary) | Rank 2 (tertiary) |
|---|---:|---:|---:|
| packed Q4 gate/up | `[12288,5120]` | `[12288,5120]` | `[10240,5120]` |
| packed Q5 down | `[5120,6144]` | `[5120,6144]` | `[5120,5120]` |

### 2-card (8192 / 9216)

| Weight | Rank 0 (primary) | Rank 1 (secondary) |
|---|---:|---:|
| packed Q4 gate/up | `[16384,5120]` | `[18432,5120]` |
| packed Q5 down | `[5120,8192]` | `[5120,9216]` |

Attention projections, embeddings, normalization, output-head, and optional
proposal weights remain on the primary. There is no full model replica on any
secondary rank. At each split MLP, NInfer fans the normalized hidden activation
out to all secondary/tertiary ranks, executes all shards concurrently,
collects the partials, and the primary sums them into the residual using
`residual_add_three` (3-card) or `residual_add_two` (2-card).

### KV cache split

| Mode | Rank 0 | Rank 1 | Rank 2 | MTP KV |
|---|---|---|---|---|
| 3-card | layers 0–4 | layers 5–10 | layers 11–15 | rank 2 |
| 2-card | layers 0–4 | layers 5–15 | — | rank 1 |

The paged KV cache is a logical shared token budget backed by device-local
physical pools (2 or 3 depending on mode). Attention layers read and append
their rank's pool directly. Remote layers copy their Q/K/V and small
block-table selectors over P2P, execute GQA against the rank-local cache, and
copy the attention result back. Keeping complete layers together preserves the
qualified 24-query-head/4-KV-head kernels and avoids a head-sliced reduction.
Every admitted request reserves the same logical number of pages in all pools,
so concurrent lanes cannot overbook any card.

### Startup memory peak

Materialization deliberately reads the artifact from storage only once, but it
has a temporary startup peak:

- **Rank 0 (primary):** the original full packed arena, later replaced by the
  compact primary arena.
- **Rank 1 (secondary):** the compact primary staging arena plus the final
  secondary shard arena.
- **Rank 2 (tertiary, 3-card only):** the final tertiary shard arena.

NInfer checks all peaks before uploading any weight bytes and reports the CUDA
device/rank when there is not enough free VRAM. After repacking, the staging
arenas are released.

## Parallel queries

Use `--max-concurrency 2` through `--max-concurrency 8` on `ninfer-serve` to
batch independent requests through the same multi-card graph. Empty lanes are
skipped, so a C8 server can still serve one request without waiting for
eight. Example:

```bash
./build-sm86/apps/ninfer-serve models/qwen3_8_27b.ninfer \
  --host 127.0.0.1 --port 8080 \
  --max-context 8192 --kv-capacity auto --kv-dtype int8 \
  --max-concurrency 8 --max-pending-requests 32 \
  --devices 0,1,2
```

This improves aggregate throughput when multiple requests are ready together;
it does not launch one independent model replica per card. Size
`--kv-capacity` for the combined resident contexts. `--kv-capacity auto`
solves all device reservation curves and selects the largest page count that
fits all cards while retaining the configured headroom on each.

## Measured results

### 2-card (NVLink, inherited from the 2× fork)

The MLP throughput measurements below were collected before KV-by-layer
sharding was added. They remain useful as the graph-split baseline. The
platform was 2× RTX 3090 with NVLink, Qwen3.8-27B groupwise-int, INT8 KV,
greedy decoding, CUDA runtime 13.1.

| Workload | One RTX 3090 | Two RTX 3090s | Dual-card change |
|---|---:|---:|---:|
| 512-token prefill | 975.62 tok/s | 1,378.72 tok/s | +41.3% |
| 2,048-token prefill | 997.66 tok/s | 1,407.94 tok/s | +41.1% |
| 256-token decode | 38.85 tok/s | 45.63 tok/s | +17.5% |
| 2,048 prefill + 256 decode | 986.95 / 38.30 tok/s | 1,393.78 / 45.31 tok/s | +41.2% / +18.3% |

HTTP concurrency (256 tokens each, no prefix reuse):

| Concurrency | Single-card aggregate | Dual-card aggregate | Dual-card change |
|---:|---:|---:|---:|
| 2 | 63.85 tok/s | 64.37 tok/s | +0.8% |
| 4 | 102.44 tok/s | 114.82 tok/s | +12.1% |
| 8 | 127.05 tok/s | 145.97 tok/s | +14.9% |

### 3-card (PCIe P2P) — pending

3-card benchmarks are pending the first smoke test and `ninfer_bench` run on
the NASoWaffle 3× 3090 system. Run the benchmark matrix with:

```bash
./build-sm86/bench/ninfer_bench \
  --weights models/qwen3_8_27b.ninfer \
  -p 512,2048 -n 256 -pg '2048,256' -r 3 --warmup 1 \
  --kv-dtype int8 --devices 0,1,2
```

Compare against `--devices 0,1` (2-card) and `--device 0` (single-card) for
the full ladder. This command requires `-DNINFER_BUILD_BENCHMARKS=ON` at
configure time.

## 7. Start the HTTP server

```bash
./build-sm86/apps/ninfer-serve models/qwen3_8_27b.ninfer \
  --host 127.0.0.1 \
  --port 8080 \
  --max-context 8192 \
  --kv-capacity 8192 \
  --kv-dtype int8 \
  --devices 0,1,2
```

The OpenAI-compatible API is then available at `http://127.0.0.1:8080/v1`.

## Optional: build and run the focused tests

Tests can be enabled in a separate build directory so the release build
remains unchanged:

```bash
cmake -S . -B build-sm86-tests -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_CUDA_COMPILER="$CUDACXX" \
  -DCMAKE_CUDA_HOST_COMPILER="$CUDAHOSTCXX" \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DNINFER_BUILD_APPS=ON \
  -DBUILD_TESTING=ON \
  -DNINFER_BUILD_BENCHMARKS=OFF

cmake --build build-sm86-tests --parallel 4

ctest --test-dir build-sm86-tests --output-on-failure \
  -R 'ninfer_(artifact_materialization|device|serve_options|graph_shard_kernels|residual_add|gqa_attention|kv_capacity)_test'
```

The 3-rank test blocks in `test_device`, `test_artifact_materialization`, and
`test_graph_shard_kernels` are skipped when fewer than 3 peer-accessible GPUs
are available. The complete test suite also contains Blackwell-only NVFP4
tests. Those targets cannot execute on an RTX 3090 and are outside the
graph-mode path.

## Troubleshooting

### A non-3090 is being selected

Run `build-sm86/multi-gpu-probe` again and pass the CUDA indices that report
RTX 3090 and `sm=86`. Do not use the indices from `nvidia-smi` without
correlating their PCI bus IDs.

### NInfer reports missing bidirectional peer access

Check `nvidia-smi topo -m` and the P2P bandwidth matrix, then confirm that the
probe reports `peer A->B=1` and `peer B->A=1` for all pairs. On systems
without NVLink, verify that the driver's P2P modules are loaded
(`lsmod | grep -i p2p` or the aikitoria module name). Graph mode deliberately
refuses to fall back to staged host-memory transfers.

### A card runs out of memory

Make sure no other process occupies any selected 3090 during startup. The
loader temporarily needs the full source arena on the primary and the compact
shards on each secondary/tertiary rank. If loading succeeds but runtime
allocation is tight, reduce `--max-context`, `--kv-capacity`, or
`--max-concurrency`. The error reports the failing CUDA rank and byte counts.

### Graph mode is not 3× as fast

Only the dense main-model MLP is split. Attention, scheduling,
synchronization, and the final reduction remain on the primary device, and
every layer transfers an activation and partial results over PCIe P2P.
Consequently speedup is workload- and batch-size-dependent and cannot approach
N×. PCIe P2P (~26 GB/s) is significantly slower than NVLink (~300 GB/s), so
the 3-card gain over 2-card is expected to be modest for decode (T=1 is
latency-bound) and larger for prefill. Benchmark this compact weight path
against `--device PRIMARY` on the exact prompt, output length, sampling
settings, and concurrency you intend to serve.

Despite the name, this mode does not capture one CUDA Graph across all
devices. Cross-device CUDA Graph capture is disabled; NInfer warms an eager
graph-parallel round during startup instead.
