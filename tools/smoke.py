#!/usr/bin/env python3
"""Smoke test of an OpenAI-compatible Inkling server: text, strict JSON schema, one image. Exit 0 all OK,
2 image failed only, 1 otherwise. (Python: runs in the SGLang image next to the server, a reference check.)"""
import base64, glob, json, sys, time, urllib.request
ep, name, figdir = sys.argv[1:4]
def chat(msgs, **kw):
    b = {"model": name, "messages": msgs, "max_tokens": 1024, "temperature": 0.0,
         "chat_template_kwargs": {"reasoning_effort": "low"}}
    b.update(kw); t = time.time()
    r = urllib.request.urlopen(urllib.request.Request(ep + "/chat/completions", json.dumps(b).encode(),
                               {"Content-Type": "application/json"}), timeout=900)
    j = json.load(r); m = j["choices"][0]["message"]
    return {"s": round(time.time() - t, 2), "content": m.get("content"),
            "reasoning": (m.get("reasoning_content") or "")[:300], "usage": j.get("usage")}
out, ok = {}, {}
def attempt(key, fn):
    try:
        out[key] = fn(); ok[key] = bool((out[key].get("content") or "").strip())
    except Exception as e:
        out[key] = {"error": repr(e)[:800]}; ok[key] = False
attempt("text", lambda: chat([{"role": "user", "content": "In one sentence: what does the PCSK9 protein do?"}]))
def js():
    r = chat([{"role": "user", "content": "Is 391 prime? Answer as JSON."}],
             response_format={"type": "json_schema", "json_schema": {"name": "a", "strict": True, "schema": {
                 "type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"],
                 "additionalProperties": False}}})
    r["parsed"] = json.loads(r["content"]); return r
attempt("json", js)
img = sorted(glob.glob(figdir + "/*.jpg"))[0]
u = "data:image/jpeg;base64," + base64.b64encode(open(img, "rb").read()).decode()
attempt("image", lambda: chat([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": u}},
                                {"type": "text", "text": "Describe this figure in two sentences."}]}]))
out["ok"] = ok; print(json.dumps(out, indent=1))
sys.exit(0 if all(ok.values()) else (2 if ok["text"] and ok["json"] else 1))
