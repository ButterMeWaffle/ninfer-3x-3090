# Dual RTX 3090 Graph Mode Build Guide

This guide builds NInfer's dual-GPU graph-parallel path for two RTX 3090s connected by NVLink.
The same `sm_86` binaries support both single-GPU and graph-parallel execution; there is no separate
graph-mode CMake option. Graph mode is selected at runtime with `--devices CUDA_A,CUDA_B`.

The current graph path supports dense Qwen3.6/Qwen3.8 27B groupwise `.ninfer` artifacts. It
physically shards every main-model MLP's packed Q4 gate/up and Q5 down weights between the two
3090s, executes both shards concurrently, and reduces the secondary result over NVLink. It also
shards the growing paged KV cache by attention layer: the primary owns the first five
full-attention layers, while the secondary owns the remaining eleven and the optional MTP KV
layer. The split is used for prompt prefill, ordinary decode, and MTP verification. The 35B-A3B
and 27B NVFP4 targets are not supported by this path.

## Requirements

- Linux; the commands below target Ubuntu 24.04.
- Two RTX 3090 or RTX 3090 Ti cards connected by an NVLink bridge.
- An NVIDIA driver that supports the installed CUDA Toolkit.
- CUDA Toolkit 12.8 or newer.
- CMake 3.28 or newer, Ninja, and GCC/G++ 13.
- Enough free memory during startup for the temporary packing layout described below.

Run every command from the repository root.

## 1. Check the physical GPUs and NVLink

Before building, confirm that both 3090s are visible and that `nvidia-smi topo -m` shows an NVLink
connection between them:

```bash
nvidia-smi --query-gpu=index,pci.bus_id,name,memory.total --format=csv
nvidia-smi topo -m
```

Do not assume that a CUDA device index matches the index printed by the main `nvidia-smi` table.
On a system with a 4070 display card, CUDA may enumerate the 4070 as device 0 even when
`nvidia-smi` lists it elsewhere. A CUDA-level check is included after the build.

## 2. Install native build dependencies

Install CUDA 12.8 or newer from NVIDIA, then install the host dependencies:

```bash
sudo apt-get update
sudo apt-get install --yes \
  build-essential gcc-13 g++-13 cmake ninja-build pkg-config \
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

Keep `CMAKE_CUDA_ARCHITECTURES=86`. Building for `89` targets the Ada display card rather than the
RTX 3090 kernels and cannot be used for the dual-3090 graph path.

## 4. Compile

```bash
cmake --build build-sm86 --parallel 2
```

CUDA compilation can consume substantial host memory. Increase `--parallel` only if the machine
has enough RAM. A successful build produces:

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

Select the two entries that:

1. Identify the RTX 3090s by name and PCI bus ID.
2. Report `sm=86`.
3. Report peer access as `1` in both directions.

For this development machine, the important part of the output is:

```text
cuda=0 ... name=NVIDIA GeForce RTX 4070 SUPER sm=89 ...
cuda=1 ... name=NVIDIA GeForce RTX 3090 sm=86 ...
cuda=2 ... name=NVIDIA GeForce RTX 3090 sm=86 ...
peer 1->2=1
peer 2->1=1
```

Therefore its graph-mode device option is `--devices 1,2`. Use the indices reported on your own
machine; do not copy these numbers blindly. NInfer checks that both selected devices have matching
compute capability and bidirectional peer access. Selecting the 4070 and a 3090 is rejected before
the model is loaded.

The order matters: the first device is primary and owns model execution, the first five text KV
layers, persistent linear-attention state, and the final MLP reduction. The second device runs the
secondary MLP shard and owns the remaining eleven text KV layers plus MTP KV when enabled.

## 6. Run a graph-mode smoke test

Replace the model path if the artifact is stored elsewhere:

```bash
./build-sm86/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "In one sentence, explain why the sky is blue." \
  --max-context 8192 \
  --max-new 128 \
  --kv-dtype int8 \
  --devices 1,2
```

For comparison, run the identical command on only the primary 3090:

```bash
./build-sm86/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "In one sentence, explain why the sky is blue." \
  --max-context 8192 \
  --max-new 128 \
  --kv-dtype int8 \
  --device 1
```

Use deterministic or identical sampling settings when comparing output and throughput. Initial
graph-mode model loading takes longer because NInfer reads the original packed arena onto the
primary, repacks the selected MLP slices over NVLink, and then installs the compact primary arena.
Compare steady-state prefill and decode measurements after loading.

## How the weights and memory are split

For every main-model MLP, graph mode places the first 8,192 intermediate channels on the primary
and the remaining 9,216 channels on the secondary:

| Weight | Primary shape | Secondary shape |
|---|---:|---:|
| packed Q4 gate/up | `[16384,5120]` | `[18432,5120]` |
| packed Q5 down | `[5120,8192]` | `[5120,9216]` |

Attention projections, embeddings, normalization, output-head, and optional proposal weights
remain on the primary. There is no full model replica on the secondary. At each split MLP, NInfer
sends the normalized hidden activation to the secondary and returns one hidden-size partial result;
the primary adds both partials to the residual.

The paged KV cache is a logical shared token budget backed by two device-local physical pools.
Attention layers 0-4 read and append the primary pool directly. Layers 5-15 copy their Q/K/V and
small block-table selectors over NVLink, execute GQA against the secondary-local cache, and copy
the attention result back. Keeping complete layers together preserves the qualified
24-query-head/4-KV-head kernels and avoids a head-sliced reduction. Every admitted request reserves
the same logical number of pages in both pools, so concurrent lanes cannot overbook either card.

The final weight footprint is therefore asymmetric. The CLI reports primary and secondary weights,
sequence arenas, workspaces, KV payload, reservations, and remaining slack separately. This is not
a transparent 48 GB CUDA allocation: each cache page stays local to the card that owns its layers,
but admission exposes their combined capacity as one aggregate resident-token entitlement.

Materialization deliberately reads the artifact from storage only once, but it has a temporary
startup peak:

- primary: the original full packed arena, later replaced by the compact primary arena;
- secondary: the compact primary staging arena plus the final secondary shard arena.

NInfer checks both peaks before uploading any weight bytes and reports the CUDA device/rank when
there is not enough free VRAM. After repacking, the staging arenas are released.

## Parallel queries

Use `--max-concurrency 2` through `--max-concurrency 8` on `ninfer-serve` to batch independent
requests through the same two-card graph. Empty lanes are skipped, so a C8 server can still serve
one request without waiting for eight. Example:

```bash
./build-sm86/apps/ninfer-serve models/qwen3_8_27b.ninfer \
  --host 127.0.0.1 --port 8080 \
  --max-context 8192 --kv-capacity auto --kv-dtype int8 \
  --max-concurrency 8 --max-pending-requests 32 \
  --devices 1,2
```

This improves aggregate throughput when multiple requests are ready together; it does not launch
one independent model replica per card. Size `--kv-capacity` for the combined resident contexts.
`--kv-capacity auto` solves both device reservation curves and selects the largest page count that
fits both cards while retaining the configured headroom on each.

On the development system, ordinary INT8 KV with `--max-context 262144`,
`--max-concurrency 8`, and the default 1 GiB automatic headroom resolved to **803,328 aggregate
resident tokens**. Startup reserved 10.30 GiB on the primary and 17.47 GiB on the secondary,
leaving 1.91 GiB and 1.00 GiB free respectively. This pool can hold roughly three full 262K
contexts at once, or more shorter contexts whose combined page entitlements fit 803,328 tokens.
It does not grant 803K context to each request.

## Measured results on the development system

The MLP throughput measurements below were collected before KV-by-layer sharding was added. They
remain useful as the graph-split baseline, but are not a post-sharding benchmark. The current
implementation was functionally qualified on the device mapping shown above with the Qwen3.8 27B
groupwise-int artifact, INT8 KV cache, greedy decoding, and CUDA runtime 13.1. These numbers are a
reference for this particular machine, not a performance guarantee.

The exact-count product benchmark used three measured repetitions after one warmup:

| Workload | One RTX 3090 | Two RTX 3090s | Dual-card change |
|---|---:|---:|---:|
| 512-token prefill | 975.62 tok/s | 1,378.72 tok/s | +41.3% |
| 2,048-token prefill | 997.66 tok/s | 1,407.94 tok/s | +41.1% |
| 256-token decode | 38.85 tok/s | 45.63 tok/s | +17.5% |
| 2,048 prefill + 256 decode | 986.95 / 38.30 tok/s | 1,393.78 / 45.31 tok/s | +41.2% / +18.3% |

A post-sharding one-repetition smoke benchmark, after one warmup, measured 512-token prefill at
974.37 tok/s on one 3090 and 1,377.97 tok/s on two (**+41.4%**). A 64-token ordinary decode
measured 38.82 tok/s on one and 44.85 tok/s on two (**+15.5%**). Use the command below with at
least three repetitions for publication-quality local results.

Run that matrix locally with:

```bash
./build-sm86/bench/ninfer_bench \
  --weights models/qwen3_8_27b.ninfer \
  -p 512,2048 -n 256 -pg '2048,256' -r 3 --warmup 1 \
  --kv-dtype int8 --devices 1,2
```

This command requires `-DNINFER_BUILD_BENCHMARKS=ON` at configure time. Replace `--devices 1,2`
with `--device 1` for the matched single-card baseline.

The HTTP concurrency test issued independent requests with exactly 256 generated tokens each,
disabled prefix reuse, and reserved 2,048 KV tokens per resident request:

| Concurrency | Single-card aggregate | Dual-card aggregate | Dual-card change | Dual batch wall time |
|---:|---:|---:|---:|---:|
| 2 | 63.85 tok/s | 64.37 tok/s | +0.8% | 7.95 s |
| 4 | 102.44 tok/s | 114.82 tok/s | +12.1% | 8.92 s |
| 8 | 127.05 tok/s | 145.97 tok/s | +14.9% | 14.03 s |

At concurrency 8, graph mode delivered 3.20 times its single-request exact-count decode rate. The
latency of an individual busy-batch request rises because all lanes share the model execution, but
the measured aggregate rate did not collapse. Those measurements used the earlier primary-only KV
layout; use the automatic-capacity result above for the current sharded memory behavior.

A deterministic ordinary-decode smoke test generated identical 35-token text in single- and
dual-card modes. The same check with MTP window 3 and the optimized proposal head also generated
identical text and identical acceptance statistics; its measured prompt decode speed was 63.03
tok/s on one card and 68.15 tok/s on two.

Post-sharding smoke validation used the real Qwen3.8 artifact for ordinary decode and MTP3, then
issued four simultaneous HTTP requests through four independently bound KV rows. All four returned
the requested deterministic text. The 803,328-token automatic-capacity server also warmed and shut
down cleanly with retained request state.

## 7. Start the HTTP server

```bash
./build-sm86/apps/ninfer-serve models/qwen3_8_27b.ninfer \
  --host 127.0.0.1 \
  --port 8080 \
  --max-context 8192 \
  --kv-capacity 8192 \
  --kv-dtype int8 \
  --devices 1,2
```

The OpenAI-compatible API is then available at `http://127.0.0.1:8080/v1`.

## Optional: build and run the focused tests

Tests can be enabled in a separate build directory so the release build remains unchanged:

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

cmake --build build-sm86-tests --parallel 2

ctest --test-dir build-sm86-tests --output-on-failure \
  -R 'ninfer_(artifact_materialization|device|serve_options|graph_shard_kernels|residual_add|gqa_attention|linear_q4_a16|linear_q5_a16|linear_add_q5_a16|linear_swiglu_q4_a16)_test'
```

The complete test suite also contains Blackwell-only NVFP4 tests. Those targets cannot execute on
an RTX 3090 and are outside the graph-mode path.

## Troubleshooting

### The 4070 is being selected

Run `build-sm86/multi-gpu-probe` again and pass the two CUDA indices that report RTX 3090 and
`sm=86`. Do not use the indices from `nvidia-smi` without correlating their PCI bus IDs.

### NInfer reports missing bidirectional peer access

Check the physical bridge and `nvidia-smi topo -m`, then confirm that the probe reports both
`peer A->B=1` and `peer B->A=1`. Graph mode deliberately refuses to fall back to staged host-memory
transfers.

### A card runs out of memory

Make sure no other process occupies either selected 3090 during startup. The loader temporarily
needs the full source arena on the primary and the compact-primary staging plus secondary shards on
the secondary. If loading succeeds but runtime allocation is tight, reduce `--max-context`,
`--kv-capacity`, or `--max-concurrency`. The error reports the failing CUDA rank and byte counts.

### Graph mode is not twice as fast

Only the dense main-model MLP is split. Attention, scheduling, synchronization, and the final
reduction remain on the primary device, and every layer transfers an activation and a partial
result over NVLink. Consequently speedup is workload- and batch-size-dependent and cannot approach
2x. Prefill also uses the split because neither card owns a complete MLP. Benchmark this compact
weight path against `--device PRIMARY` on the exact prompt, output length, sampling settings, and
concurrency you intend to serve.

Despite the name, this mode does not capture one CUDA Graph across both devices. Cross-device CUDA
Graph capture is disabled; NInfer warms an eager graph-parallel round during startup instead.
