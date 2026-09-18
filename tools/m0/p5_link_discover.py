#!/usr/bin/env python3
"""p5_link_discover.py — find the REAL kernel->op linkage in this chrome trace.

Established so far:
  * GPU kernels are on stream tids; cpu_op events are on CPU tids -> same-tid containment can
    never match (that is why the v2 attempt scored 0%).
  * cuda_runtime events (221,038) are host-side launches and SHOULD sit on CPU tids.
  * flow events exist: ph='f'/'s', cat='ac2g', 221,038 distinct ids == the cuda_runtime count.

This dumps, for a few sample kernels, every nearby candidate with its ids so the mapping is READ
rather than assumed. Print-only; changes nothing.
"""
import glob
import gzip
import json
import os
from collections import defaultdict

FILES = [f for f in sorted(glob.glob("<workdir>/tprof/**/*.json.gz", recursive=True))
         if "rank" in os.path.basename(f)]


def valid_trace(path):
    """The first capture pass was killed mid-write, leaving 4 large but TRUNCATED gz files that
    look fine by size. Test the gzip stream instead of trusting the file size."""
    try:
        with gzip.open(path, "rt") as fh:
            while fh.read(1 << 20):
                pass
        return True
    except Exception:
        return False


good = [f for f in FILES if valid_trace(f)]
print(f"rank traces found={len(FILES)} valid={len(good)} "
      f"(invalid = truncated from the killed first pass)")
if not good:
    raise SystemExit("no VALID rank traces")
f = good[<bus>]
print(f"trace: {os.path.basename(f)} ({os.path.getsize(f)/1e6:.1f} MB)")
with gzip.open(f, "rt") as fh:
    d = json.load(fh)
evs = d.get("traceEvents", [])
print(f"events: {len(evs)}")

# index by Ev Idx
by_idx = {}
for e in evs:
    if e.get("ph") != "X":
        continue
    a = e.get("args") or {}
    i = a.get("Ev Idx")
    if i is not None:
        by_idx[i] = e

# pick a kernel from the family
target = None
for e in evs:
    if e.get("ph") == "X" and e.get("cat") == "kernel" and "copyBuffer" in (e.get("name") or ""):
        target = e
        break
if target is None:
    for e in evs:
        if e.get("ph") == "X" and e.get("cat") == "kernel":
            target = e
            break

print("\n=== SAMPLE KERNEL ===")
print(json.dumps({k: target.get(k) for k in ("name", "cat", "ph", "ts", "dur", "pid", "tid")},
                 indent=2))
print("args:", json.dumps(target.get("args"), indent=2))

ext = (target.get("args") or {}).get("External id")
print(f"\n=== what does External id {ext} point to? ===")
hit = by_idx.get(ext)
if hit:
    print("  Ev Idx match ->", json.dumps({k: hit.get(k) for k in ("name", "cat", "ts", "dur", "tid")},
                                          indent=2))
    print("  its args:", json.dumps(hit.get("args"), indent=2))
else:
    print("  no event has that Ev Idx")

# search for events whose ARGS mention the kernel's id in any field
print(f"\n=== events near the kernel's ts={target.get('ts')} on ANY tid, within +-40us ===")
t0 = target.get("ts")
rows = []
for e in evs:
    if e.get("ph") not in ("X", "s", "f"):
        continue
    ts = e.get("ts")
    if ts is None:
        continue
    if abs(ts - t0) <= 40:
        rows.append(e)
rows.sort(key=lambda e: (e.get("ts"), e.get("ph")))
print(f"  {len(rows)} events")
for e in rows[:26]:
    a = e.get("args") or {}
    ids = {k: v for k, v in a.items() if k in ("External id", "Ev Idx", "Record function id")}
    print(f"  ts={e.get('ts')} ph={e.get('ph')!r:4} cat={str(e.get('cat')):16} "
          f"tid={str(e.get('tid')):>8} id={e.get('id')} {ids} :: {str(e.get('name'))[:58]}")

# flow events: id -> name on both s and f sides
print("\n=== FLOW (ac2g) id mapping sample ===")
sid, fid = {}, {}
for e in evs:
    if e.get("cat") != "ac2g":
        continue
    if e.get("ph") == "s":
        sid[e.get("id")] = (e.get("ts"), e.get("tid"), e.get("pid"))
    elif e.get("ph") == "f":
        fid[e.get("id")] = (e.get("ts"), e.get("tid"), e.get("pid"), e.get("name"))
print(f"  starts={len(sid)} finishes={len(fid)} common ids={len(set(sid) & set(fid))}")
common = list(set(sid) & set(fid))[:4]
for cid in common:
    print(f"  id={cid} start(ts,tid,pid)={sid[cid]}  finish(ts,tid,pid,name)={fid[cid]}")
print("\n  Note: start.ts should equal the launching cpu_op's ts; finish.ts the kernel's ts.")
