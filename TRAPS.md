# TRAPS.md — operational traps that will otherwise waste hours

Curated, hard-won operational failures from the gfx906 bring-up. Add to this file when a fix
costs more than ~30 minutes of debugging. Each entry: the trap, the failure mode, the fix.

## 1. vLLM renames its processes via prctl — `pkill -f python3` does NOT kill them
- Trap: vLLM V1 renames EngineCore/workers to `VLLM::EngineCore`, `VLLM::Worker_TP0_EP0`, etc.
  `pkill -9 -f python3` and `pkill -f api_server` miss them entirely.
- Failure mode: a leftover engine holds ~29 GB of GPU0/GPU1 VRAM between runs. Every subsequent
  run dies at "Free memory on device cuda:0 (2.52/31.98 GiB)" — and the failure *looks*
  environmental/transient because it survives "cleanups" that didn't actually match the processes.
- Fix: `pkill -9 -f "VLLM::"` (matches the prctl names). AND every offline client must call
  `llm.shutdown()` before exit (os._exit alone leaks the engine children). D137.

## 2. Offline clients must guard main() — the PLE worker spawns via multiprocessing SPAWN
- Trap: the PLE offload worker spawns with `popen_spawn_posix`, which RE-IMPORTS the entry
  script. A module-level `LLM(...)` re-executes in every spawn child → double engine build
  (two "Initializing a V1 engine" lines), free-memory clash, and PLE-worker EOFError.
- Failure mode: seemingly environmental double-init failures that survive container reboots
  (the bug is in the script structure). Cost three goal rounds to find. D125/D127/D129.
- Fix: wrap the body in `def main():` + `if __name__ == "__main__":`.

## 3. pct push / writes into the READ-ONLY bind mounts fail SILENTLY
- Trap: `<home>/qwen38-flash-next-mi50` (<vm> root) is mounted read-only in LXC <ct>.
  `pct push` to a path under it and in-place edits appear to succeed but do NOT persist
  (compile ran the old file; no error surfaced). D134.
- Failure mode: "I added the code but it didn't run" confusion.
- Fix: edit/compile from a writable dir (`<workdir>/m2b/`, copy the needed headers alongside) or
  on the writable tree copy `<workdir>/vllm-w`. The LOCAL repo
  `<home>/<share>/70_Code/qwen38-flash-next-mi50` is authoritative.

## 4. pct-exec with inline `-c` quoting intermittently NO-OPS
- Trap: long `pct exec 300 -- bash -c "..."` chains (especially with nested quotes/heredocs)
  sometimes return empty output and do NOTHING — no error, no effect.
- Failure mode: commands that "ran" but left stale logs/files; wasted retries.
- Fix: always push a script file and run `pct exec 300 -- bash <workdir>/<file>.sh`.

## 5. rocm-smi --showuse and gpu_busy_percent read 0 on the flashed V420 cards
- Trap: `rocm-smi --showuse` and `/sys/class/drm/card*/device/gpu_busy_percent` both read ~0
  even under a VERIFIED full-occupancy saturating kernel (built + ran, real load).
- Failure mode: "the GPU is idle" conclusions; the independent-busy leg of accounting
  procedures cannot pass. D137.
- Fix: none found — treat the SM-busy counter as unavailable on this hardware/container.
  Use kernel-time accounting (profiler) with the overlap caveat instead.

## 6. Offload-degraded model emits EOS immediately — use ignore_eos for rate measurements
- Trap: the 48L TP2+UVA-offload model (garbage logits) hit EOS after ONE token, making naive
  256-token decodes return "1 token".
- Fix: `SamplingParams(..., ignore_eos=True)` for steady-state decode rate runs. D137.

## 7. GPU "GPU use (%)" parsing: the number is the LAST field, not $4
- rocm-smi output is `GPU[<bus>] : GPU use (%): 93` — `awk '{print $NF}'` gets the value;
  `awk '{print $4}'` gets "use". D137.

## 8. Heredoc python edits: `\n` inside the insert string becomes a literal newline in C
- When patching .cu/.py via a python heredoc, `\\n` in the replacement text lands in the file
  as a real newline (broken C string literals). Always verify with a compile + `sed -n` on the
  edited region. D134.

## 9. `hipMemcpyPeer` does NOT block in ROCm 7.14 — timing it gives impossible bandwidths
- Trap: in the 7.14 / core-7.14 build, `hipMemcpyPeer` returns before the transfer completes.
  Timing `t0; hipMemcpyPeer(...); t1` measures **enqueue only**.
- Failure mode: the first M0 4-card probe reported **59,473 GB/s self-copy and 93,515 GB/s peer
  copy** — ~2000x the physical ceiling of a Gen4 x16 fabric. It looked like a spectacular result.
  The byte-exact check still PASSED, because the later verification `hipMemcpy` forced completion,
  so nothing else flagged it. D049's rule catches it: a kernel cannot sustain 2000x the machine's
  measured ceiling, therefore the measurement is wrong.
- Fix: call `hipDeviceSynchronize()` on BOTH devices inside every timed region, add a block-total
  cross-check, and assert a physical plausibility bound (peer <= 40 GB/s, self <= 1200 GB/s) that
  FLAGS implausible values instead of reporting them. D140.

## 10. Re-asserting ACS on a live peer-mapped GPU path WEDGES the GPU
- Trap: re-asserting ACS (`setpci -s <hop> ECAP_ACS+6.w=001d`) on a GPU-path bridge while the ROCm
  runtime already has peer mappings established faults sibling peer traffic.
- Failure mode: the ACS negative control killed GPU3. Probe died with `unspecified launch failure`;
  kernel logged `amdgpu <pci>: GPU reset begin!` -> `smu firmware loading failed` ->
  `fw load failed` -> `VRAM is lost due to GPU reset!` -> **`GPU reset end with ret = -22`** (the
  reset itself failed). After that **no HIP device is detectable at all** — `hipGetDeviceCount`
  returns `no ROCm-capable device is detected (n=0)` for the whole container, even though
  `rocm-smi` still lists all four GPUs and KFD still reports `p2p_links_count 3`. The wedged card
  also keeps a stale ~428 MB VRAM allocation.
- Note the trap-within-the-trap: **`rocm-smi` and KFD topology keep looking healthy while HIP is
  completely dead.** A "4 GPUs visible" check is NOT a health check — probe with an actual
  `hipGetDeviceCount` + small copy per device.
- Fix: none known short of operator recovery (D105 precedent: a <source-host> power cycle). Do not run an
  ACS assertion control on a live peer-mapped path; if it is ever needed, do it with no ROCm
  process attached and no peer mappings established. D140.

## 11. Setting `LD_LIBRARY_PATH` HIDES THE GPUS in the gfx906 LXC
> **THIS ENTRY IS WRONG — see #17.** Re-measured 2026-09-17: `LD_LIBRARY_PATH=/opt/rocm/lib` gives
> `torch.cuda.device_count() == 4`, not 0. The original reading was a stale-process confound. The
> defensive `unset` in the scripts is harmless but this explanation must not be relied on.
- Trap: in the <host> container (<container>), `torch.cuda.device_count()` is **4** with
  `LD_LIBRARY_PATH` unset, and **0** with `LD_LIBRARY_PATH=/opt/rocm/lib` or
  `/opt/rocm/core-7.14/lib`.
- Failure mode: vLLM/torch silently sees no devices; looks like a driver problem.
- Fix: **do not set `LD_LIBRARY_PATH`** in the MI50 serve script. Note <source-host>'s
  `serve_tp2_48l.sh` DID set it — that line must not be carried over to <host>. D142.

## 12. `amdsmi` pre-init `sitecustomize.py` is REQUIRED or vLLM says "Failed to infer device type"
- Trap: without a `sitecustomize.py` on the PYTHONPATH doing `import amdsmi; amdsmi.amdsmi_init()`,
  vLLM dies at startup with `RuntimeError: Failed to infer device type`.
- Fix: keep `<workdir>/sitecustomize.py` (3 lines) and ensure `<workdir>` is on `PYTHONPATH`:
  the serve scripts use `PYTHONPATH=<workdir>/vllm-w:<workdir>`. D111 noted this; D142 re-confirmed.

## 13. `pct set --memory` takes **MiB** — `--memory 128` is 128 MiB, not 128 GB
- Trap: `pct set <ct> --memory 128` sets a 128 MiB container. It still *starts*; the failure shows up
  later as bizarre OOM/behaviour. `pct config` echoes the bare number, so it looks right.
- Fix: 128 GiB = **`--memory 131072`**. Verify inside with `grep MemTotal /proc/meminfo`
  (should read ~134217728 kB). The plan needs >=64 GB, and the PLE sidecar is paged into RAM.
  D142.

## 14. `pkill -9 -f api_server` inside `bash -c '...'` KILLS ITS OWN LAUNCHER
- Trap: a launcher invoked as `pct exec CT -- bash -c '... pkill -9 -f api_server; setsid nohup ...'`
  matches its own command line, kills the wrapping shell, and the `setsid` never runs. The log you
  then read is the PREVIOUS run's, so it looks like a repeat failure.
- Failure mode: "the serve failed again with the same traceback" — actually it never started.
- Fix: put the cleanup + launch in a **script file** (`bash <workdir>/start_serve.sh`), whose command
  line does not contain the pattern. Also use `pkill -9 -f 'VLLM::'` for the prctl-named engines
  (TRAPS #1). D142.

  **SECOND OCCURRENCE + the robust fix (D142 addendum 3).** This bit again in `ab_ar.sh`, where the
  cleanup `pct exec CT -- bash -c 'pkill -9 -f "VLLM::" ...'` matched its own command line, killed
  itself, and silently did nothing - so the next run launched on top of the previous serve and the
  engine died of the collision. Use the **bracket trick** so the pattern cannot match its own
  command line:
  ```bash
  pkill -9 -f 'VLLM[M]::'            # matches VLLM:: but not the literal 'VLLM[M]::'
  pkill -9 -f 'vllm.entrypoint[s]'
  ```
  Related: killing a harness JOB (`job_kill`) does **not** kill the process inside the container.
  Clean up in-container explicitly, and verify VRAM is released, before declaring a run finished.

  **THIRD, SUBTLER FORM (D143 addendum 2) — the bracket trick is necessary but NOT sufficient.**
  `pkill -9 -f 'VLLM[M]::'` is self-match-safe but matches **nothing**, because of TRAPS #1: the
  engine renames itself with `prctl`, so `VLLM::` is in the process **comm** and never in
  `/proc/PID/cmdline`. `pkill -f` therefore reports success while five processes keep holding 31 GB
  each. The pattern that actually works is a **comm match, with no `-f`**:
  ```bash
  pkill -9 -f 'vllm.entrypoint[s]'   # parent api_server: cmdline match DOES work
  pkill -9 'VLLM'                    # engine+workers: comm match (no -f) -- the one that works
  ```
  Always verify afterwards (`pgrep -c 'VLLM'` must be 0 and VRAM must drop), and make the cleanup
  script WARN if anything survived.

## 15. `libamd_smi.so` is present but its DEP DIRECTORY is not on the loader path
- Trap: amdsmi reports `Unable to find libamd_smi.so library try installing amd-smi-lib from your
  package manager` - but the library exists in three places (`/opt/rocm/lib`, `.../core-7.14/lib`,
  and the venv's own `amdsmi/` package). The real fault is that its dependencies
  (`librocm_sysdeps_nl_genl_3.so.200`, `librocm_sysdeps_mnl.so.0`, `librocm_sysdeps_nl_3.so.200`)
  live in `/opt/rocm/core-7.14/lib/rocm_sysdeps/lib`, which is never on the loader path.
- Failure mode: **silent**. The `amdsmi` `sitecustomize` shim is wrapped in `except Exception: pass`,
  so it has been failing since it was written and nothing ever reported it. vLLM does not need
  amdsmi, so nothing broke - but **any** tool that hard-requires it (notably Ray's AMD GPU
  detection) dies with a misleading error three layers away.
- Fix: put that dir on `LD_LIBRARY_PATH`:
  `export LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib/rocm_sysdeps/lib`
  Verified: amdsmi sees 4 GPUs, torch sees 4, ray sees 4.

## 16. Ray and `benchmark_moe.py` disagree about which visibility env var to use
- Trap: `Ray 2.58` raises `RuntimeError: Please use HIP_VISIBLE_DEVICES instead of
  ROCR_VISIBLE_DEVICES` (`ray/_private/accelerators/amd_gpu.py:46`) whenever `ROCR_VISIBLE_DEVICES`
  is present **and** `HIP_VISIBLE_DEVICES` is absent.
- But `vllm/benchmarks/kernels/benchmark_moe.py` **itself** does the opposite unconditionally:
  ```python
  if current_platform.is_rocm() and "HIP_VISIBLE_DEVICES" in os.environ:
      # "Ray uses ROCR_VISIBLE_DEVICES to control device accessibility."
      os.environ["ROCR_VISIBLE_DEVICES"] = os.environ["HIP_VISIBLE_DEVICES"]
      del os.environ["HIP_VISIBLE_DEVICES"]
  ```
  Written for OLD Ray (which only honoured ROCR). Against ray 2.58 it deletes the variable Ray
  requires and sets the one Ray forbids — the script fails by construction.
- Fix: gate that swap on the Ray version (>=2.9 keeps `HIP_VISIBLE_DEVICES`), and/or wrap the exec
  in `env -u ROCR_VISIBLE_DEVICES`. Note the swap happens *inside* the script, so an outer
  `unset ROCR_VISIBLE_DEVICES` is not enough.
- Related: the serve scripts use `ROCR_VISIBLE_DEVICES` (fine for torch/vLLM); the **tuner** must
  not. D143 addendum 2.

## 17. CORRECTION to #11 — `LD_LIBRARY_PATH` did NOT hide the GPUs
- TRAPS #11 asserts that `LD_LIBRARY_PATH=/opt/rocm/lib` makes `torch.cuda.device_count()` report
  **0**. Re-measured on 2026-09-17: it reports **4**, reproducibly, on the same container.
- Most likely explanation: the original reading was taken while stale `VLLM::` processes were still
  holding the GPUs (the failure mode that bit twice in this session). That is a confound, not a
  library-path effect.
- Action: the defensive `unset LD_LIBRARY_PATH` in the serve/sweep scripts is harmless and may
  stay, but **do not rely on #11's explanation**. Re-verify before citing it to justify a workaround.

## 18. Microbench traps: an unchecked launch looks exactly like a wrong answer, and this rig's
##     run-to-run variance is 5-27% per shape
- **Always check the launch status.** In D145 an experimental GEMV variant produced
  `relerr ~1.0` (i.e. zero output). That was read as "the variant is wrong" when it equally meant
  "the kernel never ran" - `hipMalloc`'d output that nothing wrote. Wrap the launch in `CHECK(...)`
  and/or zero the output buffer first, so "did not run" is distinguishable from "computed garbage".
- **Keep an arm that must reproduce the baseline.** The same experiment's `U=1` control was supposed
  to be identical to the production kernel; it failing is the *only* reason the real bug (my variant
  was written from a partial file read and modelled a simpler kernel) was caught instead of being
  reported as "this optimisation gives no gain". A candidate with no control is not evidence.
- **Variance here is large.** Measured per-shape spread between identical harness runs reached
  **27%** (`hc.down` 33.5 us -> 24.5 us) with a typical 5-8%. Any single-run comparison below
  ~10% on this rig is not evidence; the in-serve noise floor (~1%, D143 addendum) is a *different*
  and much tighter number, so do not mix the two.
- **Read the whole kernel before "fixing" it.** D145's hypothesis (no k-unrolling, so latency could
  not be hidden) came from reading lines 1-30 of a 166-line file and inferring the rest. The kernel
  already had a 4-deep predicated unroll using `__builtin_amdgcn_fdot2`. The file was short; reading
  it was the cheap option.

## 19. Torch chrome traces: kernels CANNOT be reliably attributed to ops (13.1% ceiling here), and
##     the truncated `.gz` files look valid by size
- **The linkage is structurally incomplete in this capture.** A kernel's `args["External id"]` equals
  the launching event's `Ev Idx`, but **`cuda_runtime` events carry no `Ev Idx`** - only
  `External id`. With 221,038 runtime events against 162,245 `cpu_op`s, most kernels point at an
  event that is not indexable. Measured over 317,447 family launches: **13.1% resolved**.
  `correlation` -> flow-start does not rescue it either: only **15,757** `ph='s' cat='ac2g'` events
  exist against **221,038** finishes.
- **Containment must use the LAUNCHING CPU THREAD.** GPU kernels are on stream tids; `cpu_op` events
  are on CPU tids. Same-tid containment scores exactly 0%. Route through the `cuda_runtime` event
  (which *is* on the CPU tid) instead.
- Consequence: if op-level attribution is needed, **capture a better trace** (fuller flow metadata,
  or `rocprofv3`) - do not re-analyse this one expecting more than 13.1%. The **by-name** ranking
  needs no ids and is unaffected; rank targets on that.
- **Second trap in the same directory:** the first capture pass was killed mid-write and left 4
  large, truncated `.json.gz` files. They pass any size check and fail only on decompression
  (`EOFError: Compressed file ended before the end-of-stream marker`). **Test the gzip stream, not
  the file size**, before consuming a trace directory. Three of my scripts initially analysed the
  truncated files and reported plausible-looking numbers from them.
- **Third:** `torch.profiler` names are ambiguous by category - `ev["cat"] == "kernel"` are GPU
  kernels while `cat == "cpu_op"` are host ops, and the same short name can appear in both. Always
  filter on `cat` as well as `name`.

## 20. `rocm-smi --showclocks` prints the LEVEL INDEX before the value — parsing it wrong inverts
##     the conclusion
- Output is `GPU[<bus>] : sclk clock level: 8: (1800Mhz)`. A regex taking the **first** number reads the
  level index, not the clock: D150's first monitor analysis reported a "median 6 MHz" (physically
  impossible) and therefore concluded **"clocks never reached full speed"** - the exact opposite of
  the truth (full 1800 MHz was reached, if only 12-21% of the time).
- **Read the value inside the parentheses.** The parenthesised form is the one to trust; the leading
  integer is a DPM level selector.
- Same shape of trap applies to `--showpower`: `Current Socket Graphics Package Power (W)` is a
  package figure that can spike ~50% above `power1_cap` on short transients (observed 223-230 W
  against a 150 W cap). Judge the operating point on the **median versus the cap**, not the peak.

## 21. `rccl-tests` in <container>: three build/run blockers, none of them obvious
- **`git` is NOT installed** in the container. The Makefile also invokes `git` for version info and
  prints `make[<bus>]: git: No such file or directory` (non-fatal). Fetch the source as a GitHub tarball
  instead.
- **`hipify-perl` is only at `/opt/rocm/core-7.14/bin/hipify-perl`** and is not on PATH - only
  `hipify-clang` is (`/usr/bin/hipify-clang`). rccl-tests hipifies its CUDA sources at build time,
  so without that directory on PATH the build dies with `hipify-perl: not found` (Error 127).
- **The built binaries carry no rpath.** They link `-L/opt/rocm/lib` but running them fails with
  `libhsa-runtime64.so.1: cannot open shared object file` until `LD_LIBRARY_PATH=/opt/rocm/lib` is
  set. This is the legitimate case for that variable: a locally built binary with no rpath. (TRAPS
  #11/#17 corrected a different claim - that it hides GPUs from torch. Both can be true.)
- Also note **`-b`/`-e` are BYTES**, not element counts, despite the `(elements)` column in the
  output. `-e 256M` is a 256 MiB message. This decides whether a run fits in free VRAM when a serve
  is resident (here only ~2.7 GB of 34.3 GB was free per card).

