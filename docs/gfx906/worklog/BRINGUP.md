# Bring-up worklog

Narrative companion to `campaign/RESULTS.md`. The full append-only log is DECISIONS.md (D001–D151).

## The arc

1. **Fabric** — the cards sit on a PEX88096 under one root complex; P2P validated byte-exact, and the
   custom wave64 one-shot all-reduce (fork T44 protocol) was carried over, validated at world=4, and
   confirmed live in the TP4 serve (+4%).
2. **TP4 resident** — the AWQ checkpoint's missing shard (the 51 B-row n-gram table) is served from
   the int4 sidecar; patched index overlay; 18.23 GiB/rank with **no weight offload**; cold load
   ~72.7 s from NVMe.
3. **Profile** — in-engine torch profiler captures; per-rank kernel time 2753.6 ms vs wall 3220.9 ms
   (0.855 attribution accepted); the kernel family ranking that drove the optimisation phase.
4. **Optimisation closures** — see `campaign/DEAD_ENDS.md`; every candidate was gated, none adopted
   except the prefill MoE tiles + MI50 router + MTP2.
5. **TP8 probe** — topology, RCCL, fresh baseline, and the concurrency sweep that settled the
   capacity-vs-throughput question.

## Environment corrections that matter (see TRAPS.md)

- `LD_LIBRARY_PATH` does **not** hide GPUs (TRAPS #17 corrects #11); the original reading was a
  stale-process confound.
- The first profiler capture pass left truncated `.json.gz` files that look valid by size (TRAPS #19).
- `rocm-smi --showclocks` prints the DPM level index *before* the value; parsing the first number
  inverts the clock conclusion (TRAPS #20).
- rccl-tests needs `git`, `hipify-perl` on PATH, and an rpath (`LD_LIBRARY_PATH`) (TRAPS #21).
- The "~47.7 tok/s" figure timed the whole HTTP request — it is not a decode rate (D148).

## The numbers worth remembering

See `campaign/MEASUREMENTS.md` for the tables. The single most useful one-liner: **decode is
latency/serialization bound, not compute or weight-bandwidth bound** — proven by flat aggregate
throughput and *falling* GPU power under concurrency (D151).
