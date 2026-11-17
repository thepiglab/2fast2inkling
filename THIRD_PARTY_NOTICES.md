# Third-party notices

This repository is licensed under the Apache License 2.0 (see `LICENSE`). It contains **diffs** against, and tools
that import, the following third-party software. No third-party source file is redistributed in full; apply the
diffs to your own copy (`scripts/build_patch_tree.sh`).

| Component | Where it is used | Licence |
|---|---|---|
| **SGLang** (sgl-project/sglang), including the Inkling2 model code (`sglang/srt/models/inkling2*.py`, `sglang/srt/models/inkling_common/`, "Copyright 2026 SGLang Team") as shipped in the public `lmsysorg/sglang:dev-inkling-2-small` image | `patches/sm120/*.diff` modify six of its files; `tools/moe_probe.py` imports its kernels | Apache-2.0 |
| **FlashInfer** (flashinfer-ai/flashinfer) | its PCIe-IPC all-reduce is enabled through SGLang (`SGLANG_ENABLE_PCIE_IPC_ALLREDUCE`); FA4 SM120 attention via SGLang | Apache-2.0 |
| **PyTorch**, **NCCL**, **Triton** | runtime inside the image | BSD-3-Clause / BSD-3-Clause / MIT |
| **Inkling 2 Small** weights (thinkingmachines/Inkling-2-Small) | not included; loaded by the user | Apache-2.0 + Thinking Machines model acceptable-use policy |

Modified files: every diff in `patches/sm120/` marks its changes with comments that start with `inkling-rt` (or, for
`inkling2_vision.py`, "pigmed patch"). The method used to find and measure the changes follows the
kernel-autoresearch (kar) workflow (Apache-2.0, Gaurav Jain); no kar code is included.
