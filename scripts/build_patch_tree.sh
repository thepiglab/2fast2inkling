#!/bin/bash
# Build the sm_120 bind-mount tree: copy the six original files out of the SGLang Inkling image (or an extracted
# copy of its python tree), apply patches/sm120/*.diff, and write OUT/sglang/<same relative paths>.
#   scripts/build_patch_tree.sh --sif /path/sglang-dev-inkling-2-small.sif OUT
#   scripts/build_patch_tree.sh --from-dir /path/containing/sglang OUT      # e.g. /sgl-workspace/sglang/python
# Then bind every OUT/sglang/<rel> over /sgl-workspace/sglang/python/sglang/<rel> (see scripts/serve.sh).
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
mode=$1; src=$2; out=$3
files=(srt/models/inkling2.py srt/models/inkling2_mtp.py srt/models/inkling2_vision.py
       srt/models/inkling_common/inkling2_moe.py srt/models/inkling_common/inkling2_attn.py
       srt/layers/attention/flashattention_backend.py)
mkdir -p "$out"
for rel in "${files[@]}"; do
  mkdir -p "$out/sglang/$(dirname $rel)"
  case $mode in
    --sif) apptainer exec "$src" cat "/sgl-workspace/sglang/python/sglang/$rel" > "$out/sglang/$rel" ;;
    --from-dir) cp "$src/sglang/$rel" "$out/sglang/$rel" ;;
    *) echo "usage: $0 --sif IMAGE.sif OUT | --from-dir DIR OUT" >&2; exit 2 ;;
  esac
done
for d in "$here"/patches/sm120/*.diff; do
  patch -s -p1 -d "$out" < "$d"
done
echo "patched tree in $out/sglang:"; (cd "$out" && find sglang -name "*.py" | sort)
