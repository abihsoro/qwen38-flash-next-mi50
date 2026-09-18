#!/usr/bin/env python3
"""concurrency_bench.py — decode concurrency sweep for the TP8 question.

WHY: every number in the tracking docs so far is a SINGLE active request. TP8 may look mediocre on
one stream yet matter for serving several. This measures whether aggregate throughput scales with
concurrency (unused compute -> TP8 helps throughput) or saturates immediately (overhead/comm/
scheduler bound -> TP8 unlikely to help).

METHOD
  * Streaming requests, so per-token arrival times are real and p95 inter-token latency is measurable
    from data rather than inferred.
  * Each concurrent request gets a UNIQUE prompt salt, so nothing is served from the prefix cache.
    Identical prompts would share KV and flatter the numbers.
  * GPU sampled at ~1 Hz in a background thread, tagged with the phase, so utilisation and power can
    be attributed to each concurrency level rather than averaged over the sweep.
  * Greedy (temperature 0). Note the serve enables speculative decoding (MTP2), so a "token" here is
    a streamed token, and the per-token interval reflects a speculative round, not one forward pass.

CONSTRAINT WORTH KNOWING: the running preset sets --max-num-seqs 4. At concurrency 8 the excess
requests queue rather than batching, so decode-8 measures the queue, not 8-way batching. Reported
as-is; changing it would alter the operator-selected configuration.

Usage: concurrency_bench.py [--sizes 1 2 4 8] [--prompt-tokens 1000] [--gen 512] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8002"
MODEL = "qwen38-flash-next"

BASE_TEXT = (
    "A hash table maps keys to values through a bucket array, and collisions are resolved either by "
    "chaining or by open addressing with linear probing. The load factor determines how often probes "
    "are needed, and rehashing amortises the cost of growth across many insertions. "
)

_salt_lock = threading.Lock()
_salt_n = 0


def _salt() -> str:
    global _salt_n
    with _salt_lock:
        _salt_n += 1
        return f"[case {_salt_n:04d}] "


def post_json(path, payload, timeout=3600):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def build_prompt(target_tokens: int) -> tuple[str, int]:
    """Grow a unique prompt to >= target_tokens and report the achieved token count."""
    text = _salt() + BASE_TEXT * 4
    for _ in range(40):
        try:
            out = post_json("/tokenize", {"model": MODEL, "prompt": text})
            ids = out.get("tokens") or out.get("token_ids") or []
            n = len(ids)
        except Exception:
            n = 0
        if n >= target_tokens:
            return text, n
        text += " " + BASE_TEXT * 4
    return text, n


class GpuSampler(threading.Thread):
    """~1 Hz rocm-smi sampling, tagged with the current phase."""
    def __init__(self):
        super().__init__(daemon=True)
        self.phase = "init"
        self.samples = []          # (phase, ts, {gpu: (use%, power W, tj C)})
        self.stop = threading.Event()

    def run(self):
        import re
        import subprocess
        cur = re.compile(r"GPU\[(\d+)\]\s*:\s*(.*)")
        while not self.stop.is_set():
            try:
                out = subprocess.run(
                    ["rocm-smi", "--showuse", "--showpower", "--showtemp"],
                    capture_output=True, text=True, timeout=25).stdout
            except Exception:
                out = ""
            per = {}
            for line in out.splitlines():
                m = cur.match(line.strip())
                if not m:
                    continue
                idx, rest = int(m.group(1)), m.group(2)
                num = re.search(r"(-?\d+\.?\d*)", rest.split(":", 1)[-1] if ":" in rest else rest)
                v = float(num.group(1)) if num else None
                if v is None:
                    continue
                if "GPU use" in rest:
                    per.setdefault(idx, {})["use"] = v
                elif "Graphics Package Power" in rest:
                    per.setdefault(idx, {})["power"] = v
                elif "Sensor junction" in rest:
                    per.setdefault(idx, {})["tj"] = v
            if per:
                self.samples.append((self.phase, time.time(), per))
            self.stop.wait(1.0)


def one_request(prompt: str, gen: int, out: dict, key: int):
    """Streaming greedy request; records TTFT and every inter-token interval."""
    body = {"model": MODEL, "prompt": prompt, "max_tokens": gen, "temperature": 0.0,
            "top_p": 1.0, "ignore_eos": True, "seed": 0, "stream": True}
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Accept": "text/event-stream"})
    t0 = time.perf_counter()
    times, usage, metrics = [], None, None
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue
                ch = obj.get("choices") or [{}]
                txt = (ch[<bus>].get("text") or "")
                if txt:
                    times.append(time.perf_counter())
                if obj.get("usage"):
                    usage = obj["usage"]
                if obj.get("metrics"):
                    metrics = obj["metrics"]
    except Exception as e:  # noqa: BLE001
        out[key] = {"error": str(e)}
        return
    t_end = time.perf_counter()
    itls = [b - a for a, b in zip(times, times[1:])]
    out[key] = {
        "ttft_s": (times[<bus>] - t0) if times else None,
        "n_tokens": len(times),
        "t_first": times[<bus>] if times else None,
        "t_last": times[-1] if times else None,
        "wall_s": t_end - t0,
        "itl_ms": [x * 1000 for x in itls],
        "usage": usage,
        "metrics": metrics,
    }


def run_phase(n: int, prompt_tokens: int, gen: int, sampler: GpuSampler):
    prompts = [build_prompt(prompt_tokens)[<bus>] for _ in range(n)]
    achieved = build_prompt(prompt_tokens)[<bus>]
    out: dict = {}
    sampler.phase = f"decode-{n}"
    threads = [threading.Thread(target=one_request, args=(prompts[i], gen, out, i)) for i in range(n)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    sampler.phase = f"gap-{n}"

    ok = [v for v in out.values() if "error" not in v and v.get("n_tokens")]
    agg_tokens = sum(v["n_tokens"] for v in ok)
    per_req = [v["n_tokens"] / (v["t_last"] - v["t_first"]) for v in ok
               if v["t_last"] and v["t_first"] and v["t_last"] > v["t_first"]]
    pooled = [x for v in ok for x in v["itl_ms"]]
    pooled_sorted = sorted(pooled)
    p95 = pooled_sorted[int(0.95 * (len(pooled_sorted) - 1))] if pooled_sorted else None
    return {
        "phase": f"decode-{n}",
        "concurrency": n,
        "prompt_tokens_achieved": achieved,
        "gen_target": gen,
        "wall_s": round(wall, 3),
        "requests_ok": len(ok),
        "requests_failed": len(out) - len(ok),
        "aggregate_tps": round(agg_tokens / wall, 2) if wall else None,
        "per_request_tps_median": round(statistics.median(per_req), 2) if per_req else None,
        "per_request_tps_all": [round(x, 2) for x in per_req],
        "ttft_s_median": round(statistics.median([v["ttft_s"] for v in ok if v["ttft_s"]]), 3)
        if any(v["ttft_s"] for v in ok) else None,
        "itl_ms_median": round(statistics.median(pooled), 2) if pooled else None,
        "itl_ms_p95": round(p95, 2) if p95 is not None else None,
        "itl_ms_max": round(max(pooled), 2) if pooled else None,
        "tokens_total": agg_tokens,
        "errors": [v["error"] for v in out.values() if "error" in v][:4],
        "sample_metrics": next((v["metrics"] for v in ok if v.get("metrics")), None),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--prompt-tokens", type=int, default=1000)
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--json", default="<workdir>/concurrency_results.json")
    args = ap.parse_args()

    # warmup
    print("warmup...", flush=True)
    w: dict = {}
    one_request(build_prompt(args.prompt_tokens)[<bus>], 16, w, 0)

    sampler = GpuSampler()
    sampler.start()

    results = []
    for n in args.sizes:
        print(f"\n=== decode-{n}: {n} concurrent, {args.prompt_tokens} prompt, {args.gen} gen ===",
              flush=True)
        r = run_phase(n, args.prompt_tokens, args.gen, sampler)
        results.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "sample_metrics"}), flush=True)
        if r.get("sample_metrics"):
            print("  server metrics sample:", json.dumps(r["sample_metrics"])[:300], flush=True)

    sampler.stop.set()
    sampler.join(timeout=10)

    # GPU attribution per phase
    by_phase = {}
    for phase, _ts, per in sampler.samples:
        d = by_phase.setdefault(phase, {"use": [], "power": [], "tj": []})
        for g, vals in per.items():
            for k, v in vals.items():
                d[k].append(v)
    gpu = {}
    for phase, d in by_phase.items():
        gpu[phase] = {
            "n_samples": len(d["use"]),
            "use_median": round(statistics.median(d["use"]), 1) if d["use"] else None,
            "power_median": round(statistics.median(d["power"]), 1) if d["power"] else None,
            "power_max": round(max(d["power"]), 1) if d["power"] else None,
            "tj_max": round(max(d["tj"]), 1) if d["tj"] else None,
        }

    payload = {"results": results, "gpu_by_phase": gpu, "gen_target": args.gen,
               "prompt_tokens_target": args.prompt_tokens}
    with open(args.json, "w") as fh:
        json.dump(payload, fh, indent=2)

    print("\n=== SUMMARY ===")
    print(f"{'phase':>10} {'agg tok/s':>10} {'per-req med':>12} {'ITL med ms':>11} "
          f"{'ITL p95 ms':>11} {'wall s':>8} {'ERR':>4}")
    for r in results:
        print(f"{r['phase']:>10} {str(r['aggregate_tps']):>10} {str(r['per_request_tps_median']):>12} "
              f"{str(r['itl_ms_median']):>11} {str(r['itl_ms_p95']):>11} {str(r['wall_s']):>8} "
              f"{r['requests_failed']:>4}")
    print("\n=== GPU BY PHASE ===")
    for phase in sorted(gpu):
        if not phase.startswith("decode"):
            continue
        g = gpu[phase]
        print(f"  {phase:>10}  use {g['use_median']}%  power {g['power_median']}W "
              f"(max {g['power_max']})  tj max {g['tj_max']}C  n={g['n_samples']}")
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
