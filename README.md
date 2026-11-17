# 2fast2inkling

Run **Inkling 2 Small** (thinkingmachines/Inkling-2-Small, NVFP4, 594B total / 17B active MoE) on one node of
**8x NVIDIA RTX PRO 6000 Blackwell** (sm_120, PCIe only, no NVLink), and make it fast.

- **Stock SGLang cannot do this at all.** Its Inkling2 code fails on sm_120 in five places, whatever the flags (SM100-only
  expert path, NVLink-multicast-only all-reduces, SM100 decode attention, an FA4 dtype check, an SM100-sized
  shared-memory request). `patches/sm120` fixes all five.
- **Then it makes it faster**: the node is communication-bound over PCIe, so the patches cut the bytes and the cost
  of every cross-GPU reduction, and a memory split lets more requests run at once.
- **End to end** on a synthetic-query generation workload (long articles in, strict JSON out): **+42% queries per
  GPU-hour on text and +72% on figures** over the first working configuration, with outputs matching stock SGLang on
  4x B200 within ordinary serving noise.

The patches are six small diffs against SGLang's `dev-inkling-2-small` image, applied as bind mounts; nothing is
rebuilt. All speed changes are behind environment variables, so the same patched image runs the baseline.

## Results

**Workload:** 1,000 text calls (4,988 article chunks, ~8k-token prompts per call, ~3.2k generated tokens per call
incl. reasoning at effort "low") and 1,000 figure calls (one image + caption context each). Temperature 1.0, strict
JSON-schema output (xgrammar). One node, 8x RTX PRO 6000 (96 GB each), TP8 attention / EP8 experts.

### Baseline

| Configuration | Result |
|---|---|
| Stock SGLang image, any flags | **does not run**: crashes at load or on the first request (see [Why stock fails](#why-stock-sglang-fails-on-sm_120)) |
| First working config: patches for sm_120 only, NCCL all-reduces, SGLang default pools (`--swa-full-tokens-ratio 0.1 --mamba-full-memory-ratio 0.1 --mem-fraction-static 0.85`), client concurrency 256 | decode step **82 ms** per rank at 228 running requests (~2.8k decode tok/s); text **3,851** queries/GPU-h, figures **1,756** (grounded) / **1,865** (blind) |

### Speedups, step by step (decode)

| Step | Change | Decode ms/step (running reqs) | Decode tok/s (node) | Cumulative |
|---|---|---:|---:|---:|
| 0 | first working config (above) | 82 (228) | 2.8k | 1.00x |
| 1 | `INKRT_LMHEAD=replicated`: full LM head on every GPU, no FP32 logits all-gather | ~77 (228)* | ~3.0k | ~1.06x |
| 2 | `INKRT_MOE_AR=bf16`: expert combine all-reduced in BF16 instead of FP32 | ~69 (228)* | ~3.3k | ~1.19x |
| 3 | `INKRT_AR_GROUP=tp` + `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1`: Inkling's reductions on FlashInfer's PCIe-IPC all-reduce (~138 vs ~219 us per decode reduction) | ~61 (228)* | ~3.7k | ~1.34x |
| 4 | memory split `--mem-fraction-static 0.88 --swa-full-tokens-ratio 0.2 --mamba-full-memory-ratio 0.15 --max-running-requests 384`, client concurrency 384 | 81 (341), measured | **4.2k** | **1.51x** |

\* Steps 1-3 are derived from profiled kernel times at batch 228 (gather 6.3 ms replaced by a 1.5 ms GEMM; FP32 vs BF16
NCCL all-reduce 395 vs ~219 us x 48 layers; PCIe-IPC vs NCCL ~138 vs ~219 us x 96 reductions). Measured with steps
1-3 on: 58 ms/step at 184 running requests. Steps 0 and 4 are measured. Same-slice A/B of steps 1-3
together (502 text calls, concurrency 256): **3,463 -> 4,561 queries/GPU-h (+32%)**.

### End to end (same full workload, same node type)

| Arm | Config | Wall | Queries | Valid JSON | **Queries / GPU-hour** | Prompt tok/s | Generated tok/s |
|---|---|---:|---:|---:|---:|---:|---:|
| Text (1,000 calls) | baseline | 2,318 s | 19,837 | 100% | 3,851 | 3,622 | 1,368 |
| Text | **2fast2inkling** | 1,634 s | 19,835 | 100% | **5,463 (+42%)** | 5,138 | 1,963 |
| Figures, grounded (1,000) | baseline | 768 s | 2,996 | 99.9% | 1,756 | 2,297 | 1,387 |
| Figures, grounded | **2fast2inkling** | 446 s | 2,998 | 100% | **3,025 (+72%)** | 3,955 | 2,346 |
| Text | knobs on, pools 0.4/0.4, concurrency 256 (losing variant: 153 retractions) | 1,843 s | 19,850 | 100% | 4,848 (+26%) | | |

**Parity:** greedy outputs on 64 fixed prompts (48 text, 16 figures) against **stock SGLang on 4x B200** (the supported
configuration). On RTX at concurrency 1 the patched path is deterministic (a repeat is 64/64 identical); the speed
knobs move greedy tokens and logprobs by the same amount ordinary request batching does (mean |delta logprob| 0.024,
first divergences at near-ties), and B200 vs RTX sits at that same level (0.026-0.027). Details:
[docs/REPORT.md](docs/REPORT.md#5-parity).

## Where the time goes

Torch profiler, 8 ranks, 40 decode steps of the text workload at 228 running requests, first working config:

| Class | Share | ms/step | Notes |
|---|---:|---:|---|
| all-reduce / gather | 43% | 36 | expert combine FP32 395 us x48, attention BF16 219 us x48, FP32 logits all-gather 6.3 ms |
| expert GEMMs (Marlin) | 28% | 24 | ~80-90% of the 1.54 TB/s DRAM peak at 256-1024 tokens/step |
| attention | 18% | 14 | full-attention layers bandwidth-bound on KV at ~8.5k-token contexts; sliding-window layers small |
| other | 11% | 9 | norms, attention residuals, projections, short convolutions, routing |

After the patches (184 running): expert GEMMs 42%, all-reduce 22%, attention 21%; comm fell from 36 to 14 ms/step.

**Hardware notes.** The GPUs sit in PCIe-switch pairs (0-1, 2-3, 4-5, 6-7) on four NUMA nodes; every other pair is
cross-socket. P2P works between all eight. NCCL peaks at ~25.5 GB/s bus bandwidth (17 GB/s at decode sizes), and no
NCCL setting changes that (P2P level, protocol, channels). A hierarchical all-reduce cannot gain much here, because
each switch pair shares one CPU uplink. FlashInfer's PCIe-IPC kernels (staged pushes, 4+4 islands) do decode-size
reductions ~1.5x faster than NCCL.

## Why stock SGLang fails on sm_120

| # | Where | Failure | Fix |
|---|---|---|---|
| 1 | `inkling2_moe.py` NVFP4 scale swizzle | "NVFP4 expert computation requires an SM100, SM103 or SM107 GPU" | sm_12x takes Inkling's Hopper path: Marlin W4/W8 GEMMs with Triton fake-quantized NVFP4/MXFP8 activations (W4A4/W8A8 numerics kept) |
| 2 | Inkling all-reduces | "Inkling2 tensor parallelism requires NVLink multicast" | plain group all-reduces (NCCL or PCIe-IPC) |
| 3 | decode attention | SM100 CuTe rel-bias kernel; FA4 SM90 (wgmma) kernel for MTP verify | Inkling's Triton split-KV kernel |
| 4 | prefill attention (FA4 SM120) | `TypeError: All tensors must have the same data type` (FP32 out requested, kernel needs out dtype == q dtype) | BF16 output buffer, upcast into the FP32 buffer |
| 5 | `route.cuh` routing metadata | "Cannot set route metadata shared memory" above ~336 tokens (sort needs ~36 B/id; sm_120 allows 99 KB/block) | `torch.sort` fallback above 2,688 routed ids |

Inkling2 also refuses data-parallel attention and pipeline parallelism, and NVFP4 experts refuse expert-TP, so the
only layout is `--tp 8 --ep-size 8` with `fa4` attention. **Minimum is 8 GPUs**: 6 would need DP attention (8 KV heads
rule out TP6), and 4 cannot hold the experts (~94 GB per GPU on a 96 GB card).

## Quick start

Needs the public `lmsysorg/sglang:dev-inkling-2-small` image as an Apptainer SIF, the Inkling-2-Small checkpoint
(rev b0e7afb, ~367 GB) on fast local disk, and one 8x RTX PRO 6000 node.

```bash
# 1. build the patched files (copies six files out of the image and applies patches/sm120/*.diff)
scripts/build_patch_tree.sh --sif sglang-dev-inkling-2-small.sif ./tree

# 2. serve (bind-mounts ./tree, sets the env knobs, TP8/EP8)
SIF=sglang-dev-inkling-2-small.sif MODEL=/local/nvme/Inkling-2-Small PATCH_TREE=./tree PORT=30000 scripts/serve.sh

# 3. any OpenAI-compatible client; keep ~384 requests in flight for long-context text
curl -s localhost:30000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"inkling","messages":[{"role":"user","content":"What does PCSK9 do?"}],
       "chat_template_kwargs":{"reasoning_effort":"low"}}'
```

Environment knobs (all off by default in the patched code; `scripts/serve.sh` turns them on):

| Variable | Effect |
|---|---|
| `INKRT_LMHEAD=replicated` | full LM head on every rank (+1.6 GB/GPU), no logits all-gather |
| `INKRT_MOE_AR=bf16` | BF16 expert-combine all-reduce (default FP32) |
| `INKRT_AR_GROUP=tp` | route Inkling's reductions through SGLang's `tp` group, the only group its PCIe-IPC backend serves |
| `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1` | SGLang/FlashInfer PCIe-IPC all-reduce for decode-size BF16 reductions |

## Tools

| Tool | What it does |
|---|---|
| `tools/parity.py` | collect greedy tokens + top-5 logprobs from a server on a fixed prompt set; compare two collections |
| `tools/moe_probe.py` | 1-GPU bound probe of Inkling's expert path at EP4/6/8 shapes vs a DRAM read |
| `tools/arbench.py` | NCCL all-reduce microbenchmark (torchrun) |
| `tools/prof_summary.py` | kernel-class split of SGLang torch-profiler traces |
| `tools/steady.py`, `tools/metrics_delta.py` | steady-state and whole-run throughput from server logs / Prometheus metrics |
| `tools/smoke.py` | text, strict JSON and image smoke test |
| `tools/truncate.py` | layer-truncated checkpoint for single-GPU crash testing (outputs meaningless) |

## Method

Profile first, then bound each idea before building it (the kernel-autoresearch / kar workflow): the expert GEMMs and
attention were measured at 80-90% of DRAM bandwidth, so no kernel was rewritten; all-reduces were 43% of the step,
so the work went there. Throughput is end-to-end queries per GPU-hour on the full workload, plus same-slice A/Bs.
Parity: 64 prompts, temperature 0, 256 new tokens, first-divergence position, near-tie test (top-2 logprob gap < 0.5),
and |delta logprob| of the chosen token on the shared prefix, with a run-to-run noise floor measured the same way.

## Limitations and next steps

- Prefill reductions still use NCCL (they were ~39% of GPU time at 341 running requests). Letting PCIe-IPC take them
  (`SGLANG_PCIE_IPC_MAX_NUMEL`) and MTP speculative decoding are **untested**.
- Memory is the binding limit at long contexts (some retractions at 341 running); `--disable-radix-cache` and FP8 KV
  (after a quality check) are the next levers.
- DP attention for Inkling would roughly halve all-reduce bytes per token and allow 6-GPU replicas; it needs a model
  code change.
- The BF16 attention-output rounding (fix 4) and the BF16 combine are the only numerics that differ from the B200
  path by construction; both measured within serving noise.

Full write-up: [docs/REPORT.md](docs/REPORT.md). Licence: Apache-2.0 (matches the SGLang code the diffs apply to);
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
