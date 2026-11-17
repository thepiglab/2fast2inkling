#!/usr/bin/env python3
"""Build a layer-truncated Inkling 2 Small checkpoint for single-GPU crash/debug runs (outputs are meaningless).

truncate.py SRC DST --keep 0,1,2,3,4,5,46,47
Old layer KEEP[j] becomes layer j. attn_res.attn_queries.{i} belongs to layer i+1, mlp_queries.{i} to layer i; both
are remapped. MTP tensors are dropped. config.json / hf_quant_config.json / index are rewritten; other small files are
copied. Runs inside the SGLang image (torch + safetensors). Write DST only to node-local disk.
"""
import argparse, json, os, re, shutil
from collections import defaultdict

from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--keep", default="0,1,2,3,4,5,46,47")
a = ap.parse_args()
keep = [int(x) for x in a.keep.split(",")]
new_of = {old: j for j, old in enumerate(keep)}
os.makedirs(a.dst, exist_ok=True)
cfg = json.load(open(os.path.join(a.src, "config.json")))
tc = cfg["text_config"]
local_old = set(tc["local_layer_ids"])
tc["num_hidden_layers"] = len(keep)
tc["local_layer_ids"] = [j for j, old in enumerate(keep) if old in local_old]
for j, old in enumerate(keep):
    if j and (old in local_old) != (j in tc["local_layer_ids"]):
        raise SystemExit("local/global mismatch")
json.dump(cfg, open(os.path.join(a.dst, "config.json"), "w"), indent=2)

LAYER = re.compile(r"^(model\.llm\.layers\.)(\d+)(\..*)$")
AQ = re.compile(r"^(model\.llm\.attn_res\.attn_queries\.)(\d+)$")
MQ = re.compile(r"^(model\.llm\.attn_res\.mlp_queries\.)(\d+)$")


def rename(name):
    if name.startswith("model.mtp.") or name.startswith("mtp."):
        return None
    m = LAYER.match(name)
    if m:
        old = int(m.group(2))
        return f"{m.group(1)}{new_of[old]}{m.group(3)}" if old in new_of else None
    m = AQ.match(name)
    if m:  # query for layer i+1
        layer = int(m.group(2)) + 1
        return f"{m.group(1)}{new_of[layer] - 1}" if layer in new_of and new_of[layer] > 0 else None
    m = MQ.match(name)
    if m:
        old = int(m.group(2))
        return f"{m.group(1)}{new_of[old]}" if old in new_of else None
    return name


idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))
by_file = defaultdict(list)
for name, f in idx["weight_map"].items():
    by_file[f].append(name)
odd = sorted({re.sub(r"\d+", "N", n) for n in idx["weight_map"] if re.search(r"\.\d+", n)
              and not (LAYER.match(n) or AQ.match(n) or MQ.match(n) or n.startswith("model.visual"))})
print("indexed names with numbers outside layers/attn_res/visual:", odd[:20])
def shard_of(nn):
    m = LAYER.match(nn)
    return f"model-L{int(m.group(2)):02d}.safetensors" if m else "model-rest.safetensors"


expected = defaultdict(int)
for n in idx["weight_map"]:
    nn = rename(n)
    if nn is not None:
        expected[shard_of(nn)] += 1
groups, wmap = defaultdict(dict), {}
for f, names in sorted(by_file.items()):
    with safe_open(os.path.join(a.src, f), "pt") as h:
        for n in names:
            nn = rename(n)
            if nn is None:
                continue
            shard = shard_of(nn)
            groups[shard][nn] = h.get_tensor(n)
            wmap[nn] = shard
            if len(groups[shard]) == expected[shard]:  # shard complete: write it, free memory
                save_file(groups.pop(shard), os.path.join(a.dst, shard), metadata={"format": "pt"})
assert not groups, f"incomplete shards: {list(groups)}"
json.dump({"metadata": {}, "weight_map": wmap}, open(os.path.join(a.dst, "model.safetensors.index.json"), "w"))
q = json.load(open(os.path.join(a.src, "hf_quant_config.json")))
for g in q["model"]["parameter_groups"]:
    g["tensors"] = [x for x in (rename(t) for t in g["tensors"]) if x is not None]
json.dump(q, open(os.path.join(a.dst, "hf_quant_config.json"), "w"), indent=1)
for fn in os.listdir(a.src):
    if not fn.endswith(".safetensors") and fn not in ("config.json", "hf_quant_config.json", "model.safetensors.index.json"):
        p = os.path.join(a.src, fn)
        (shutil.copytree if os.path.isdir(p) else shutil.copy)(p, os.path.join(a.dst, fn))
print("kept", len(wmap), "tensors in", len(set(wmap.values())), "shards; layers", keep, "->", list(range(len(keep))),
      "local", tc["local_layer_ids"])
