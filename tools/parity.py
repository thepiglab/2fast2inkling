#!/usr/bin/env python3
"""Greedy parity probe for an OpenAI-compatible SGLang server.

collect: send a fixed prompt set (text + image, chat template, reasoning low) at temperature 0 with logprobs, save JSONL.
compare: token agreement + chosen-token logprob diffs between two collections (ref vs candidate).
Python (not Rust) because it only runs inside the SGLang image as a reference check, next to the server.
"""
import argparse, base64, json, sys, time, glob, random, concurrent.futures as cf, urllib.request, statistics as st

def post(ep, body, timeout=1800):
    r = urllib.request.Request(ep + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))

def prompts(sample, articles, figs, n_text, n_img, seed):
    rnd = random.Random(seed)
    arts = {}
    for line in open(articles):
        a = json.loads(line); arts[a.get("id") or a.get("pmcid") or a.get("article_id")] = a
    out = []
    rows = [json.loads(l) for l in open(sample)]
    rnd.shuffle(rows)
    for r in rows[:n_text]:
        txt = r.get("text") or r.get("chunk") or json.dumps(r)[:6000]
        out.append({"id": f"t{len(out)}", "messages": [{"role": "user", "content":
            "Write three search queries a clinician might type to find this passage. Answer as JSON "
            "{\"queries\": [...]}.\n\n" + txt[:6000]}]})
    imgs = sorted(glob.glob(figs + "/*.jpg")); rnd.shuffle(imgs)
    for p in imgs[:n_img]:
        u = "data:image/jpeg;base64," + base64.b64encode(open(p, "rb").read()).decode()
        out.append({"id": f"i{len(out)}", "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": u}},
            {"type": "text", "text": "Describe this figure and write two search queries for it."}]}]})
    return out

def collect(a):
    ps = prompts(a.sample, a.articles, a.figs, a.n_text, a.n_img, a.seed)
    def one(p):
        b = {"model": a.model, "messages": p["messages"], "max_tokens": a.max_tokens, "temperature": 0.0,
             "logprobs": True, "top_logprobs": 5, "chat_template_kwargs": {"reasoning_effort": "low"},
             "separate_reasoning": False}
        t = time.time(); j = post(a.endpoint, b); c = j["choices"][0]
        lp = (c.get("logprobs") or {}).get("content") or []
        return {"id": p["id"], "s": round(time.time() - t, 2), "text": c["message"].get("content"),
                "tokens": [x["token"] for x in lp], "lp": [x["logprob"] for x in lp],
                "top": [[(y["token"], y["logprob"]) for y in x.get("top_logprobs", [])] for x in lp],
                "usage": j.get("usage")}
    with cf.ThreadPoolExecutor(a.conc) as ex, open(a.out, "w") as f:
        for r in ex.map(one, ps):
            f.write(json.dumps(r) + "\n")
    print("wrote", a.out, len(ps))

def compare(a):
    R = {j["id"]: j for j in map(json.loads, open(a.ref))}; C = {j["id"]: j for j in map(json.loads, open(a.cand))}
    same, prefix, diffs, ties, n = 0, [], [], 0, 0
    for k in R:
        if k not in C: continue
        r, c = R[k], C[k]; n += 1
        m = 0
        while m < min(len(r["tokens"]), len(c["tokens"])) and r["tokens"][m] == c["tokens"][m]:
            diffs.append(abs(r["lp"][m] - c["lp"][m])); m += 1
        if m == len(r["tokens"]) == len(c["tokens"]): same += 1
        elif m < min(len(r["tokens"]), len(c["tokens"])):
            # near-tie: the ref's top-2 gap at the first divergence
            t = r["top"][m]; gap = (t[0][1] - t[1][1]) if len(t) > 1 else 99
            ties += gap < a.tie
        prefix.append(m / max(1, len(r["tokens"])))
    q = sorted(diffs)
    res = {"n": n, "identical": same, "mean_prefix_frac": round(st.mean(prefix), 4) if prefix else None,
           "divergences_at_near_tie": ties, "tie_threshold": a.tie,
           "mean_abs_dlogprob": round(st.mean(q), 5) if q else None,
           "p99_abs_dlogprob": round(q[int(0.99 * (len(q) - 1))], 5) if q else None}
    print(json.dumps(res)); json.dump(res, open(a.out, "w"))

ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
c = sp.add_parser("collect")
for k, v in [("endpoint", None), ("model", None), ("sample", None), ("articles", None), ("figs", None), ("out", None)]:
    c.add_argument("--" + k, required=True)
c.add_argument("--n-text", type=int, default=48); c.add_argument("--n-img", type=int, default=16)
c.add_argument("--max-tokens", type=int, default=256); c.add_argument("--conc", type=int, default=1)
c.add_argument("--seed", type=int, default=7)
m = sp.add_parser("compare"); m.add_argument("--ref", required=True); m.add_argument("--cand", required=True)
m.add_argument("--out", required=True); m.add_argument("--tie", type=float, default=0.5)
a = ap.parse_args(); {"collect": collect, "compare": compare}[a.cmd](a)
