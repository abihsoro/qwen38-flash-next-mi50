# Trap checklist (N-C5) — T1-T8 from Rev 5 + gfx906-specific findings

| Trap | Effect | Rule |
|---|---|---|
| T1 HIP graphs drop hipStreamWaitValue32 | graphed decode reads stale PLE output | NO GPU-side waits; worker DMA + per-worker counters (fork protocol) |
| T2 torch.compile trace-time freeze | int8 path never runs in the graph | opaque custom ops with fake impls; eager op registration; VLLM_DISABLE_COMPILE_CACHE=1 to clear |
| T3 cross-device signalling = memory type | logprob divergence 1e-5..1e-2 | flags in a 4 KB page in each rank's UNCACHED staging, same IPC handle; announce = one posted P2P store per peer; poll own memory with s_sleep(8) |
| T4 seq numbers from device counter | kernel args frozen at graph capture | device counter, never an argument |
| T5 boot self-test | false RCCL fallback (-25% decode) | warm untimed, then min of 3 timed repeats |
| T6 TP=4 stability | cards fall off PCIe bus | kernel cmdline N-A1; HSA_NO_SCRATCH_RECLAIM=1; NCCL_P2P_LEVEL=PXB; max batched tokens 2048 |
| T7 vision | SDPA warm-up tries 256 GiB | --language-model-only --skip-mm-profiling |
| T8 gfx9 gates | builds but WRONG results (wave32 on wave64) | validate numerically, never assume (Lane D did; the remaining on_gfx10x gates are per-task) |
| gfx906+: v_dot4_i32_i8 | compiles but garbage | int8 unpacks to fp16 + fdot2 (D026) |
| gfx906+: host reads of GPU-written coherent memory | stale CPU cache, no snoop | hipMemcpy D2H / uncached mapping (D016) |
| gfx906+: concurrent-kernel uncached-flag visibility | unreliable single-GPU | protocol validates on 4-card box (D031) |
| gfx906+: barrier over-subscription | grid barrier deadlock | blocks must fit resident capacity (D031) |
| gfx906+: hipBLASLt | no gfx906 device library | no torch._int_mm INT8 path (M6 note) |
