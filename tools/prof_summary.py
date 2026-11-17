#!/usr/bin/env python3
"""Summarize SGLang torch-profiler traces (chrome trace .json.gz per rank): GPU kernel time by class.

usage: prof_summary.py <dir with *.trace.json.gz> [--top 40]
Prints per-rank totals, the class split (% of GPU kernel time), and the top kernels. Kernel classes are regexes on the
kernel name; anything unmatched is 'other'. Python because it runs once per profile on a CPU node (reference tooling).
"""
import argparse, collections, glob, gzip, json, os, re

CLASSES = [
    ("allreduce", r"pcie_ipc|ipc_topo|rsag|nccl|all_?reduce|AllReduce|allgather|reduce_scatter|cross_device|one_?shot|multimem|_rank_sum|custom_ar"),
    ("moe_gemm", r"marlin|Marlin|grouped_gemm|moe_wna16|expert|fused_moe|cutlass.*grouped|_source_partial"),
    ("moe_route", r"route|topk|sort|src2dst|reorder|permut|_marlin_route|compute_.*metadata"),
    ("quant", r"fake_quantize|quantize|nvfp4|mxfp8|_e2m1|swiglu"),
    ("attention", r"flash|fmha|attn|attention|_decode_partials|combine_gate_sink|gate_sink|rel_bias|paged"),
    ("sconv", r"sconv|conv1d|causal_conv"),
    ("attn_res", r"attn_res|_attn_res|block_attn"),
    ("dense_gemm", r"gemm|Gemm|GEMM|nvjet|cublas|cutlass|sm90_xmma|sm100_xmma|ampere|linear|matmul|splitK"),
    ("norm", r"norm|rms|Norm"),
    ("sampling", r"sampl|softmax|argmax|logit|softcap|penalt|grammar|bitmask"),
    ("elementwise", r"elementwise|vectorized|copy|fill|cat|index|gather|scatter|add|mul|where|Memcpy|Memset"),
]
RX = [(n, re.compile(p)) for n, p in CLASSES]


def klass(name):
    for n, rx in RX:
        if rx.search(name):
            return n
    return "other"


def load(path):
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as f:
        return json.load(f)


ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--top", type=int, default=40)
a = ap.parse_args()
files = sorted(glob.glob(os.path.join(a.dir, "**", "*.json*"), recursive=True))
tot_cls, tot_k, per_rank = collections.Counter(), collections.Counter(), {}
cnt_k = collections.Counter()
for f in files:
    try:
        tr = load(f)
    except Exception as e:
        print("skip", f, e); continue
    ev = tr["traceEvents"] if isinstance(tr, dict) else tr
    ks = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    if not ks:
        continue
    t0 = min(e["ts"] for e in ks); t1 = max(e["ts"] + e["dur"] for e in ks)
    busy = sum(e["dur"] for e in ks)
    per_rank[os.path.basename(f)] = (len(ks), busy / 1e3, (t1 - t0) / 1e3)
    for e in ks:
        c = klass(e["name"]); tot_cls[c] += e["dur"]; tot_k[e["name"][:110]] += e["dur"]; cnt_k[e["name"][:110]] += 1
print("rank file | kernels | kernel ms | span ms | busy %")
for k, (n, b, s) in per_rank.items():
    print(f"{k[:60]:60s} {n:8d} {b:10.1f} {s:10.1f} {100 * b / max(s, 1e-9):6.1f}")
T = sum(tot_cls.values()) or 1
print("\nclass split (all ranks):")
for c, v in tot_cls.most_common():
    print(f"  {c:12s} {100 * v / T:6.2f}%  {v / 1e3:10.1f} ms")
print(f"\ntop {a.top} kernels:")
for k, v in tot_k.most_common(a.top):
    print(f"  {100 * v / T:6.2f}%  {v / 1e3:9.1f} ms  n={cnt_k[k]:7d}  avg={v / cnt_k[k]:8.1f} us  [{klass(k)}] {k}")
