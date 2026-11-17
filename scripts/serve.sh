#!/bin/bash
# Serve Inkling 2 Small (NVFP4) on one node of 8x RTX PRO 6000 (sm_120, PCIe only) with the 2fast2inkling patches.
#
#   SIF=sglang-dev-inkling-2-small.sif MODEL=/local/nvme/Inkling-2-Small PATCH_TREE=/path/tree PORT=30000 \
#     scripts/serve.sh
#
# PATCH_TREE is the output of scripts/build_patch_tree.sh (it contains sglang/srt/...). Keep the weights on fast
# node-local disk. Apptainer notes: only bind files that already exist in the image, and do not bind a /tmp
# subdirectory (both force "underlay" mode, which fails on a stale lock file in /tmp of this image); host /tmp is
# visible in the container without a bind on most sites.
#
# Memory split (per GPU: ~47 GB weights + 1.6 GB replicated LM head; the rest is KV and short-conv state):
#   long-context text (~10k tokens per request): the defaults below (3.9M full-attention KV tokens, 341 running);
#     client concurrency ~384.
#   short-context figures only (~3k tokens per request): SWA_RATIO=0.4 MAMBA_RATIO=0.4 MAX_RUNNING=768 (not
#     benchmarked; check the server's "max_running_requests is capped" log line).
set -euo pipefail
: "${SIF:?path to the SGLang dev-inkling-2-small image}" "${MODEL:?path to the Inkling-2-Small checkpoint}"
: "${PATCH_TREE:?output dir of scripts/build_patch_tree.sh}"
CACHE=${CACHE:-$(dirname "$MODEL")/inkling-cache}; mkdir -p "$CACHE"
BINDS=()
while IFS= read -r f; do
  BINDS+=(--bind "$PATCH_TREE/sglang/$f:/sgl-workspace/sglang/python/sglang/$f")
done < <(cd "$PATCH_TREE/sglang" && find . -type f -name "*.py" | sed 's|^\./||')
exec apptainer exec --nv --cleanenv "${BINDS[@]}" --bind "$(dirname "$MODEL")" \
  --env HF_HUB_OFFLINE=1 --env XDG_CACHE_HOME="$CACHE/xdg" --env TRITON_CACHE_DIR="$CACHE/triton" \
  --env CUDA_CACHE_PATH="$CACHE/nv" --env TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" \
  --env FLASHINFER_WORKSPACE_BASE="$CACHE/flashinfer" --env TMPDIR="$CACHE" \
  --env INKRT_MOE_AR=bf16 --env INKRT_LMHEAD=replicated --env INKRT_AR_GROUP=tp \
  --env SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1 \
  "$SIF" sglang serve --model-path "$MODEL" --served-model-name "${NAME:-inkling}" --trust-remote-code \
  --tp 8 --ep-size 8 --attention-backend fa4 --disable-prefill-cuda-graph --page-size 128 --dtype bfloat16 \
  --mamba-radix-cache-strategy extra_buffer --mem-fraction-static "${MEM:-0.88}" \
  --swa-full-tokens-ratio "${SWA_RATIO:-0.2}" --mamba-full-memory-ratio "${MAMBA_RATIO:-0.15}" \
  --max-running-requests "${MAX_RUNNING:-384}" ${EXTRA_FLAGS:-} \
  --grammar-backend xgrammar --reasoning-parser inkling --tool-call-parser inkling2 \
  --enable-multimodal --enable-metrics --context-length 131072 --host "${HOST:-127.0.0.1}" --port "${PORT:-30000}"
