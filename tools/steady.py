#!/usr/bin/env python3
"""Steady-state throughput from an SGLang server.log (TP0 lines): mean decode gen throughput over "Decode batch"
log lines whose #running-req >= frac * target, plus the token-weighted prefill rate over the same time window.

usage: steady.py server.log TARGET_RUNNING [frac=0.9]
"""
import re, sys
from datetime import datetime

log, target = sys.argv[1], int(sys.argv[2])
frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9
D = re.compile(r"\[(\S+ \S+) TP0[^\]]*\] Decode batch.*#running-req: (\d+).*gen throughput \(token/s\): ([0-9.]+)")
P = re.compile(r"\[(\S+ \S+) TP0[^\]]*\] Prefill batch.*#new-token: (\d+)")
dec, pre = [], []
for line in open(log, errors="ignore"):
    m = D.search(line)
    if m:
        dec.append((datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"), int(m.group(2)), float(m.group(3))))
        continue
    m = P.search(line)
    if m:
        pre.append((datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"), int(m.group(2))))
ss = [d for d in dec if d[1] >= frac * target]
if not ss:
    print(f"no steady window (max running {max((d[1] for d in dec), default=0)})"); sys.exit()
t0, t1 = ss[0][0], ss[-1][0]
span = max((t1 - t0).total_seconds(), 1)
ptoks = sum(n for t, n in pre if t0 <= t <= t1)
g = sum(d[2] for d in ss) / len(ss)
print(f"steady window {span:.0f}s ({len(ss)} decode logs, mean running {sum(d[1] for d in ss) / len(ss):.0f}): "
      f"decode {g:.0f} tok/s, prefill {ptoks / span:.0f} tok/s, total {g + ptoks / span:.0f} tok/s")
