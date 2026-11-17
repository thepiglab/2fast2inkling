#!/usr/bin/env python3
"""Bound probe: Inkling2's SM12x (Hopper/Marlin) expert path for ONE EP rank at EP8 shapes (48 of 384 experts),
timed under CUDA graphs per token count, against a pure DRAM read of the same weight bytes.

Mirrors Inkling2Experts._forward_source (inkling-rt sm120 patch) minus the all-reduce: route_metadata -> fake-quantize
x -> Marlin w13 (gemv or gemm) -> SwiGLU + fake-quantize -> Marlin w2 -> FP32 source partial sums.
Random weights (timing only). Prints one JSON line per (dtype, tokens). Runs inside the SGLang image on one GPU.
"""
import json, sys
from functools import partial

import torch

from sglang.kernels.ops.inkling2 import experts_sm90, marlin_sm90
from sglang.kernels.ops.inkling2.route import ROUTE_MAX_IDS, route_metadata
from sglang.kernels.ops.moe.inkling_moe import compute_grouped_gemm_metadata

E, H, I, TOPK = 384, 4608, 2304, 8
EP = int(sys.argv[1]) if len(sys.argv) > 1 else 8
L = E // EP
TOKENS = [int(t) for t in (sys.argv[2].split(",") if len(sys.argv) > 2 else "1,8,32,64,128,228,256,512,1024,2048,8192".split(","))]
dev = torch.device("cuda")
torch.manual_seed(0)


def nvfp4(n, k):
    w = torch.randint(0, 256, (L, n, k // 2), dtype=torch.uint8, device=dev)
    s = (torch.rand(L, n, k // 16, device=dev) * 0.5 + 0.5).to(torch.float8_e4m3fn)
    return w, s


def mxfp8(n, k):
    w = (torch.randn(L, n, k, device=dev) * 0.5).to(torch.float8_e4m3fn)
    s = torch.randint(118, 130, (L, n, k // 32), dtype=torch.uint8, device=dev)
    return w, s


def build(kind):
    mk = nvfp4 if kind == "nvfp4" else mxfp8
    w13, s13 = mk(2 * I, H)
    w2, s2 = mk(H, I)
    raw = sum(t.numel() * t.element_size() for t in (w13, s13, w2, s2))
    rep = marlin_sm90.repack if kind == "nvfp4" else marlin_sm90.repack_mxfp8
    m13, m2 = rep(w13, s13), rep(w2, s2)
    del w13, s13, w2, s2
    marlin_bytes = sum(t.numel() * t.element_size() for m in (m13, m2) for t in m if t is not None)
    amax13 = torch.rand(L, device=dev) + 1.0
    amax2 = torch.rand(L, device=dev) + 1.0
    den = torch.ones(1, device=dev)
    return m13, m2, amax13, amax2, den, raw, marlin_bytes


def forward(x, weights, ids, P, kind):
    m13, m2, amax13, amax2, den = P
    tokens, top_k = ids.shape
    flat = ids.flatten()
    n = flat.numel()
    start, num_local = 0, L
    if n <= 2688:  # SM12x smem cap (inkling-rt patch v4)
        _, dst2src, _, counts, offsets, *_ = route_metadata(flat, weights.flatten(), start, num_local, E)
    else:
        sorted_ids, permutation = torch.sort(flat.to(torch.int16), stable=True)
        counts, offsets, _, _ = compute_grouped_gemm_metadata(sorted_ids, E, block_size_m=128)
        counts, offsets = counts[start:start + num_local], offsets[start:start + num_local + 1]
        dst2src = permutation.to(torch.int32)
    if marlin_sm90.uses_gemv(m13, tokens, num_local):
        matmul = partial(marlin_sm90.gemv, metadata=(counts, offsets, dst2src), tokens=tokens)
    else:
        routing = marlin_sm90.route(counts, offsets, n, tokens, dst2src)
        matmul = partial(marlin_sm90.gemm, routing=routing)
    caps = (4.0, 25.0)
    if kind == "nvfp4":
        xq, x_amax = experts_sm90.fake_quantize(x.contiguous())
    else:
        xq = experts_sm90.fake_quantize_mxfp8(x.contiguous())
    acc = matmul(xq, m13, 2 * I, top_k=top_k)
    if kind == "nvfp4":
        hidden, h_amax = marlin_sm90.swiglu_fake_quantize(acc, flat, weights.flatten(), (x_amax, amax13, den),
                                                          start, num_local, top_k, caps)
        scales = (h_amax, amax2, den)
    else:
        hidden = marlin_sm90.swiglu_fake_quantize_mxfp8(acc, flat, weights.flatten(), start, num_local, caps)
        scales = (None, None, None)
    down = matmul(hidden, m2, H, top_k=1)
    return marlin_sm90.source_partial(down, flat, scales, start, num_local, top_k)


def timed(fn, reps=30):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    ts = []
    for _ in range(reps):
        flush.zero_()  # cold L2
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return ts[len(ts) // 2]


props = torch.cuda.get_device_properties(0)
print(json.dumps({"gpu": props.name, "sms": props.multi_processor_count, "ep": EP, "local_experts": L}), flush=True)
# DRAM read floor probe: max over a buffer of the same size as the Marlin weights
for kind in ("nvfp4", "mxfp8"):
    m13, m2, amax13, amax2, den, raw, mbytes = build(kind)
    P = (m13, m2, amax13, amax2, den)
    big = torch.empty(mbytes // 4, dtype=torch.int32, device=dev).random_(0, 1 << 20)
    floor_ms = timed(lambda: big.amax())
    print(json.dumps({"kind": kind, "weight_bytes": raw, "marlin_bytes": mbytes, "read_floor_ms": round(floor_ms, 4),
                      "read_GBs": round(mbytes / floor_ms / 1e6, 1)}), flush=True)
    del big
    for T in TOKENS:
        x = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
        ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(T)]).to(torch.int32)
        w = torch.softmax(torch.randn(T, TOPK, device=dev), -1)
        try:
            ms = timed(lambda: forward(x, w, ids, P, kind))
            local_rows = int((ids < L).sum())
            print(json.dumps({"kind": kind, "tokens": T, "local_rows": local_rows, "ms": round(ms, 4),
                              "eff_GBs": round(mbytes / ms / 1e6, 1), "pct_floor": round(100 * floor_ms / ms, 1)}),
                  flush=True)
        except Exception as e:
            print(json.dumps({"kind": kind, "tokens": T, "error": repr(e)[:300]}), flush=True)
    del m13, m2
    torch.cuda.empty_cache()
