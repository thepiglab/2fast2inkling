# Inkling 2 Small (NVFP4) on 8x RTX PRO 6000 (sm_120): making it run, then making it fast

Date: 2026-11-17. Model: thinkingmachines/Inkling-2-Small rev b0e7afb (NVFP4 experts in layers
0-39, MXFP8 experts in layers 40-47, BF16 attention; 48 layers, 384 experts top-8, 8 KV heads). Image: SGLang
`dev-inkling-2-small` (sglang 0.0.0.dev1+inkling2, torch 2.14.1+cu130, flashinfer 0.7.2, NCCL 2.30.7).
Workload: PigMed recipe v2 (pigmed-synth `c2` text layout, `figb`/`figg` figure layouts), reasoning effort low,
temperature 1.0, strict JSON-schema output.

## 1. Result in one table

Same recipe-v2 sample, same node (8x RTX PRO 6000, one node), reasoning effort low.
Baseline = the first configuration that runs at all (patch v4: sm_120 enablement only, SGLang defaults otherwise,
`--swa-full-tokens-ratio 0.1 --mamba-full-memory-ratio 0.1 --mem-fraction-static 0.85`, client concurrency 256). Optimized = patch v6 + comm knobs + memory split, client concurrency 384. Token rates are pigmed-synth's per-chunk token counts x chunks / wall time.

| arm (calls) | config | wall | queries | JSON-valid | **queries / GPU-hour** | prompt tok/s | generated tok/s | generated tok/s/GPU |
|---|---|---|---|---|---|---|---|---|
| c2 text (1,000 articles = 4,988 chunks) | baseline | 2,318 s | 19,837 | 100% | **3,851** | 3,622 | 1,368 | 171 |
| c2 text | **optimized** | 1,634 s | 19,835 | 100% | **5,463 (+42%)** | 5,138 | 1,963 | 245 |
| figg figures (1,000) | baseline | 768 s | 2,996 | 99.9% | **1,756** | 2,297 | 1,387 | 173 |
| figg figures | **optimized** | 446 s | 2,998 | 100% | **3,025 (+72%)** | 3,955 | 2,346 | 293 |
| c2 text | optimized knobs, pools 0.4/0.4, mem 0.88, concurrency 256 | 1,843 s | 19,850 | 100% | 4,848 (+26%) | | | |

Minimum GPU count that works: **8** (one node, TP8/EP8). Parity: inside SGLang's run-to-run noise against stock
SGLang on 4x B200 (section 5). Serve command: section 7.


## 2. Stock SGLang does not run Inkling on sm_120 at all

Inkling2 ships its own MoE, attention and communication code; `--moe-runner-backend` / `--fp4-gemm-backend` are
ignored. On RTX PRO 6000 (capability 12.0, PCIe only) the stock image fails in five places, whatever the flags:

| # | where | failure | fix in `patches/sm120` (bind-mounted over the image) |
|---|---|---|---|
| 1 | `inkling2_moe.py` `swizzle_nvfp4_scales` | raises "NVFP4 expert computation requires an SM100, SM103 or SM107 GPU" (only capability 9.x takes the portable path) | SM12x takes Inkling's Hopper path: Marlin W4/W8 GEMMs (mma.sync) with Triton fake-quantized NVFP4/MXFP8 activations, so the W4A4/W8A8 numerics are kept |
| 2 | `kernels/ops/inkling2/comm.py` | every all-reduce goes to NVLink multicast / symmetric memory: "Inkling2 tensor parallelism requires NVLink multicast" | plain group all-reduces on SM12x (NCCL, or FlashInfer PCIe-IPC, see section 4) |
| 3 | decode attention | non-sm90 takes an SM100 CuTe rel-bias decode kernel; MTP verify can take an FA4 SM90 (wgmma) kernel | SM12x uses Inkling's Triton split-KV `decode_partials` |
| 4 | prefill attention (FA4 SM120 runtime) | `TypeError: All tensors must have the same data type` (Inkling asks FP32 out, SM120 FA4 needs out dtype == q dtype) | BF16 output buffer upcast into the FP32 buffer (one extra BF16 rounding vs SM100) |
| 5 | `route.cuh` route_metadata | "Cannot set route metadata shared memory" at batch > 336 (one-CTA sort needs ~36 B/id; SM12x max 99 KB/block, SM100 227 KB) | torch.sort fallback above 2688 routed ids |

Also: Inkling2 refuses DP (`dp_size != 1`) and PP; NVFP4 experts refuse expert-TP, so the only layout is
`--tp 8 --ep-size 8`; attention must be `fa4` (the triton backend has no Inkling rel-bias support).
Apptainer trap: binding a file that does not exist in the image, or any `/tmp/<dir>`, puts the session into underlay
mode, which dies on the image's stale `/tmp/uv-setuptools-*.lock` (host `/tmp` is visible without a bind).

## 3. Where the time goes (torch profiler, 8 ranks, 40 decode steps, c2 at ~228 running requests)

Baseline (patch v4, NCCL): ~82 ms per decode step per rank.

| class | share | ms/step | notes |
|---|---|---|---|
| all-reduce / gather | 43% | 36 | MoE combine FP32 AR 395 us x48; attention o-proj BF16 AR 219 us x48; FP32 logits all-gather 6.3 ms (vocab-parallel LM head, 201k vocab) |
| Marlin MoE | 28% | 24 | ~80-90% of the 1.54 TB/s DRAM peak (measured on this GPU by a sibling project) at 256-1024 tokens/step (1-GPU probe at EP8 shapes) |
| attention | 18% | 14 | full layers ~811 us (KV ~1 GB/rank at ~8.5k context), SWA 102 us; bandwidth-bound |
| other | 11% | 9 | norms, attention residuals, projections, short convs, routing |

The PCIe fabric is the bottleneck: GPUs sit in PCIe-switch pairs (0-1, 2-3, 4-5, 6-7) on four NUMA nodes, the rest is
cross-socket. NCCL all-reduce peaks at ~25.5 GB/s bus bandwidth (17 GB/s at decode sizes) and no NCCL setting moved
it (P2P_LEVEL=SYS, PROTO=Simple: identical). Prefill is communication-bound too (~125 us/token of all-reduce vs
~19 us/token of expert GEMM at 8k-token chunks).

## 4. What made it faster

All knobs are in the patch set (`patches/sm120/*.diff`, called v6 below), off by
default, switched by environment variables, so the same patch serves the baseline and the fast config.

| change | mechanism | measured |
|---|---|---|
| KV/conv memory split | `--swa-full-tokens-ratio 0.1` starved the SWA pool (usage 0.86-0.99, prefill stalls, decode fell to ~0.9k tok/s); the conv-state cache capped `max_running_requests` at 228. For c2 (~10k tokens per request) the full-attention KV is what binds: `--mem-fraction-static 0.88 --swa-full-tokens-ratio 0.2 --mamba-full-memory-ratio 0.15` gives 3.9M full-KV tokens and 341 running (ratio 0.4/0.6 at mem 0.85 gave 2.5M tokens: full pool at 1.00, ~170 running, 129 retractions) | see table 1 |
| `INKRT_LMHEAD=replicated` | full LM head on every rank (+1.6 GB/GPU): no vocab-parallel FP32 logits all-gather (6.3 ms/step at bs 228); costs one 1.5 ms GEMM | -4.8 ms/step |
| `INKRT_MOE_AR=bf16` | expert combine all-reduced in BF16 (each rank's FP32 partial rounded once) instead of FP32: half the bytes | -8.7 ms/step on NCCL |
| `INKRT_AR_GROUP=tp` + `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1` | SGLang already ships FlashInfer's PCIe-IPC all-reduce (staged pushes, 4+4 island decomposition, for no-NVLink sm_120 hosts), but only for the group named `tp`; Inkling reduces on `attention_tp` / `moe_ep`. The patch routes them through `tp` (same 8 ranks). Decode-size BF16 reductions: ~138-144 us vs NCCL ~215 us; prefill-size ones stay on NCCL | comm 36 -> 14 ms/step |
| (all three comm knobs together) | | decode step 82 -> 58-63 ms per rank at bs ~185-205 |

A/B on the same 502-call c2 slice at client concurrency 256 (same memory split, ratio 0.4/0.4): knobs off 1,295 s,
3,463 queries/GPU-h; knobs on 981 s, **4,561 queries/GPU-h (+32%)**; JSON-valid 100% in both.

What did not move anything: NCCL `NCCL_P2P_LEVEL=SYS` (P2P is already on everywhere), `NCCL_PROTO=Simple`, more
channels: NCCL tops out at ~25.5 GB/s bus bandwidth on this fabric whatever the setting.


## 5. Parity

Method: 64 fixed prompts (48 recipe-v2 text chunks, 16 figures; seed 7), chat template, reasoning effort low,
temperature 0, 256 new tokens, top-5 logprobs (`tools/parity.py`). Compared: token agreement up to the first
divergence, whether the first divergence is a near-tie (reference top-2 logprob gap < 0.5), and |Δ logprob| of the
chosen token on the shared prefix. Reference: **stock SGLang on 4x B200** (TP4/EP4, no patch, the supported config). Stock SGLang has no working RTX config to compare against.

| comparison (concurrency 16 during collection) | mean prefix kept | first divergences at near-ties | mean \|Δ logprob\| | p99 |
|---|---|---|---|---|
| RTX knobs-off vs RTX knobs-off, repeated (run-to-run noise floor) | 18% | 56/64 | 0.024 | 0.249 |
| **B200 stock vs RTX knobs-off** (sm_120 enablement patch) | 17% | 57/64 | 0.026 | 0.258 |
| **B200 stock vs RTX all knobs** (fast config) | 15% | 59/64 | 0.027 | 0.262 |
| RTX knobs-off vs each knob alone: LM head / BF16 combine / PCIe-IPC | 20% / 17% / 16% | 54 / 58 / 60 | 0.025 / 0.027 / 0.022 | 0.25 / 0.28 / 0.25 |

Reading: SGLang is not batch-invariant, so at concurrency 16 two identical runs already diverge early, almost always
at near-ties. The B200-vs-RTX gap and every knob sit at that noise floor; the BF16 expert combine is the only one a
hair above it. Recipe-v2 quality held: JSON-valid 100% (c2) and 99.9% (figures) in every bench arm, accepted-query
rates unchanged.

At **concurrency 1** (fixed batch composition) the RTX path is deterministic: knobs-off vs a repeat is 64/64
identical, zero logprob difference. Knobs-off vs all knobs at concurrency 1: 0/64 identical, mean prefix 16%,
53/64 first divergences at near-ties, mean |Δ logprob| 0.024 (p99 0.27). So the comm knobs do change the numerics
(BF16 rounding of the expert combine, a different reduction order in the PCIe-IPC all-reduce, a different LM-head
GEMM shape), by the same amount that ordinary batching changes them (0.024 at concurrency 16). B200 stock (c16)
vs RTX knobs-off (c1): 0.027. A concurrency-1 B200 reference was not run.



## 6. Fewer GPUs

Weights alone decide this (per-GPU bytes from the loaded checkpoint, EP across the GPUs):

| GPUs | layout | expert bytes / GPU | fits 96 GB? | verdict |
|---|---|---|---|---|
| 8 | TP8 attention, EP8 experts | 45 GB (+2.4 GB attention/vocab/vision) | yes, ~31 GB left for KV | **minimum that works** |
| 6 | attention TP must divide the 8 KV heads, so TP6 is impossible; needs DP attention (DP3 x attn-TP2) + EP6 | 63 GB | yes | **not possible with this code**: Inkling2 raises on `dp_size != 1` and its decoder layer has no DP gather/scatter around the MoE. A model-code change (all-gather tokens before the experts, reduce-scatter after, replicated LM head); would also halve comm bytes per token. Not done tonight. |
| 4 | TP4/EP4 | 94 GB (NVFP4 40 x 1.72 + MXFP8 8 x 3.15) | no | only if the MXFP8 layers 40-47 are re-quantized to NVFP4 (-> ~83 GB experts, ~86 GB total), leaving ~8 GB for KV: ~50 sequences at c2 length. Breaks parity by construction and would be slower per GPU. Not recommended. |

One-GPU probes of the expert path at EP4/EP6 shapes (96/64 local experts) behave like EP8 (time scales with local
expert bytes), so a DP-attention 6-GPU build would be bandwidth-equivalent per GPU, with less comm.


## 7. Serve command

`scripts/serve.sh` (see the README quick start). Copy the weights to node-local disk first. It bind-mounts the
patched files built by `scripts/build_patch_tree.sh`, sets the four environment knobs and runs `sglang serve` with:

```
--tp 8 --ep-size 8 --attention-backend fa4 --disable-prefill-cuda-graph --page-size 128 --dtype bfloat16
--mamba-radix-cache-strategy extra_buffer --mem-fraction-static 0.88 --swa-full-tokens-ratio 0.2
--mamba-full-memory-ratio 0.15 --max-running-requests 384 --grammar-backend xgrammar --reasoning-parser inkling
--tool-call-parser inkling2 --enable-multimodal --context-length 131072
env: INKRT_MOE_AR=bf16 INKRT_LMHEAD=replicated INKRT_AR_GROUP=tp SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1
```

Client concurrency 384 (341 requests run, the rest queue). Server ready in 70-100 s with a warm JIT cache (first
start on a node ~5 min).

## 8. Dead ends and what is left

- **Expert GEMM kernel rewrite**: not pursued. A 1-GPU bound probe (`tools/moe_probe.py`, Inkling's own Marlin path at
  EP8 shapes, CUDA graphs, cold L2) moves ~1.2-1.4 TB/s for 256-1024 tokens/step (0.68 ms per NVFP4 layer at 256
  tokens, 1.16 ms per MXFP8 layer): ~80-90% of the 1.54 TB/s DRAM peak measured on this GPU. At most
  ~10-20% of the 28-42% MoE share is left for a kernel.
  Prefill MoE uses BF16 tensor cores (~184 TFLOP/s at 8k tokens); a native block-scaled FP4 `mma.sync` kernel for
  sm_120 is the only big kernel lever, and only on the prefill share (comm still dominates prefill).
- **Decode attention kernel**: Triton split-KV at ~1.2 TB/s on full-attention layers (811 us at ~8.5k context) and on
  SWA layers. Near bandwidth. FP8 KV would halve it but changes numerics (needs a quality check).
- **NCCL tuning**: P2P level, protocol, channels: no change (~25.5 GB/s ceiling). A topology-aware hierarchical
  all-reduce cannot beat the ring by much here: each PCIe-switch pair shares one CPU uplink (1.75 S -> 1.5 S per
  uplink).
- **High concurrency without the memory split**: concurrency 640 at ratio 0.4/0.6 and mem 0.85 filled the full KV pool
  (1.00), ran only ~170 requests and retracted 129. Concurrency is bounded by full-attention KV at c2 lengths.
- **Tooling traps found**: NCCL teardown hangs after a torchrun microbench on this host (use `os._exit`); apptainer
  binds of new files or `/tmp` subdirectories fail (underlay mode); SGLang's `route.cuh` shared-memory request is
  sized for SM100.

What is left, by expected value:
1. **DP attention for Inkling** (model change): removes the attention all-reduce (attention weights replicated,
   +5.6 GB/GPU), turns the expert combine into all-gather + reduce-scatter: about half the comm bytes per token,
   and makes **6-GPU replicas** possible (DP3 x attn-TP2, EP6).
2. **Memory**: `--disable-radix-cache` (c2 prompts barely share prefixes; the radix cache holds SWA/full KV that the
   running batch needs), and FP8 KV after a quality check.
3. **Native FP4 prefill MoE** for sm_120, once comm is lower. A sibling runtime found cuBLASLt 13.2 runs block-scaled
   NVFP4 W4A4 on sm_120 (E2M1 + VEC16_UE4M3 scales, device-pointer alpha for the dynamic activation scale), so start
   there (per-expert or batched GEMMs) before a hand-written `mma.sync` kernel. Note sm_120's ~99 KB opt-in shared
   memory per block.
4. **Prefill-size PCIe-IPC all-reduce** (`SGLANG_PCIE_IPC_MAX_NUMEL=40000000`): prefill reductions are still on NCCL
   (~4.2 ms per 75 MB, ~39% of GPU time in the c384 profile window, ~2.4 prompt tokens per generated token in this
   workload). The A/B started, prefilled at ~11.3k input tok/s, and was cut off when the GPU time
   ran out: **untested**. Check it for stream-ordering hangs
   (SGLang notes one IPC workspace serves one stream at a time).
5. **MTP (DSPARK) speculative decoding**: queued but not run (GPU time ran out). Expected small or negative here:
   a verify block multiplies the per-token all-reduce traffic, and the node is comm-bound per token.


## 9. Files

- `patches/sm120/*.diff`: the patch set (six SGLang files; never adds files, see section 2).
- `scripts/build_patch_tree.sh`, `scripts/serve.sh`.
- `tools/`: `parity.py`, `arbench.py`, `moe_probe.py`, `prof_summary.py`, `steady.py`, `metrics_delta.py`, `smoke.py`,
  `truncate.py` (layer-truncated checkpoint for 1-GPU crash testing).
