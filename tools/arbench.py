#!/usr/bin/env python3
"""All-reduce microbench over the node's GPUs (torchrun): NCCL bf16 at decode-like sizes (B x 4608 x 2 bytes).
Reports per-size latency and bus bandwidth, plus p2p access matrix. Launch: torchrun --nproc-per-node N arbench.py"""
import os, sys, json, torch, torch.distributed as dist, time
dist.init_process_group("nccl"); r = dist.get_rank(); w = dist.get_world_size(); torch.cuda.set_device(r)
res = {"world": w}
if r == 0:
    res["p2p"] = [[int(torch.cuda.can_device_access_peer(i, j)) if i != j else 1 for j in range(w)] for i in range(w)]
import itertools
for dtype, B in itertools.product((torch.bfloat16, torch.float32), [1, 8, 32, 64, 128, 228, 256, 512, 1024, 2048, 4096]):
    x = torch.randn(B, 4608, dtype=dtype, device="cuda")
    for _ in range(5): dist.all_reduce(x)
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10): dist.all_reduce(x)
    for _ in range(3): g.replay()
    torch.cuda.synchronize(); t = time.perf_counter(); N = 20
    for _ in range(N): g.replay()
    torch.cuda.synchronize(); dt = (time.perf_counter() - t) / (N * 10)
    nbytes = x.numel() * x.element_size()
    res[f"{str(dtype)[6:]}_B{B}"] = {"us": round(dt * 1e6, 1), "busbw_GBs": round(nbytes * 2 * (w - 1) / w / dt / 1e9, 2)}
if r == 0: print("ARBENCH " + os.environ.get("NCCL_PROTO", "") + "/" + os.environ.get("NCCL_ALGO", "") + " " + json.dumps(res))
torch.cuda.synchronize(); sys.stdout.flush(); os._exit(0)  # NCCL teardown hangs on this host
