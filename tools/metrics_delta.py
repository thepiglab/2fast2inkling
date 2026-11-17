#!/usr/bin/env python3
"""Throughput between two SGLang /metrics snapshots: prompt/generation tokens per second and finished requests.

usage: metrics_delta.py start.prom end.prom start_epoch end_epoch [gpus]
Counters are cumulative over finished requests, so use snapshots far apart (whole bench runs).
"""
import re, sys


def read(p):
    d = {}
    for line in open(p):
        m = re.match(r"^(sglang:(?:prompt_tokens_total|generation_tokens_total|num_requests_total))\{[^}]*\} ([0-9.e+]+)", line)
        if m:
            d[m.group(1)] = d.get(m.group(1), 0.0) + float(m.group(2))
    return d


a, b = read(sys.argv[1]), read(sys.argv[2])
dt = float(sys.argv[4]) - float(sys.argv[3])
g = int(sys.argv[5]) if len(sys.argv) > 5 else 8
p = b["sglang:prompt_tokens_total"] - a["sglang:prompt_tokens_total"]
o = b["sglang:generation_tokens_total"] - a["sglang:generation_tokens_total"]
n = b["sglang:num_requests_total"] - a["sglang:num_requests_total"]
print(f"wall {dt:.0f}s requests {n:.0f} prompt_tok/s {p / dt:.0f} gen_tok/s {o / dt:.0f} "
      f"gen_tok/s/GPU {o / dt / g:.0f} req/GPU-h {n / dt * 3600 / g:.0f}")
