# Evidence for the MI50 TP4 review

Collected through read-only inspection on September 17, 2026 (America/Chicago). Line numbers below refer to the inspected source, not this excerpt file. No inference benchmarks were run.

**E1. MTP acceptance and runtime JIT**

Source: <container>:<workdir>/mtp_B.log

SHA-256: ed41904146191db168953c62538e53697d294c6712435841b9bef9419773ade1

```text
489: (Worker_TP0_EP0 pid=64527) WARNING 09-18 02:22:15 [jit_monitor.py:141] Triton kernel JIT compilation during inference: _compress_qsa_groups_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.
490: (Worker_TP0_EP0 pid=64527) WARNING 09-18 02:22:15 [jit_monitor.py:141] Triton kernel JIT compilation during inference: _qsa_mqa_paged_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.
491: (Worker_TP0_EP0 pid=64527) WARNING 09-18 02:22:15 [jit_monitor.py:141] Triton kernel JIT compilation during inference: _expand_qsa_indices_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.
492: (Worker_TP0_EP0 pid=64527) WARNING 09-18 02:22:16 [jit_monitor.py:141] Triton kernel JIT compilation during inference: _qsa_sparse_paged_gqa_splitk_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.
493: (Worker_TP0_EP0 pid=64527) WARNING 09-18 02:22:16 [jit_monitor.py:141] Triton kernel JIT compilation during inference: _qsa_merge_splitk_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.
502: (Worker_TP0_EP0 pid=64527) WARNING 09-18 02:22:17 [jit_monitor.py:141] Triton kernel JIT compilation during inference: fused_moe_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.
511: (Worker_TP0_EP0 pid=64527) WARNING 09-18 02:22:17 [jit_monitor.py:141] Triton kernel JIT compilation during inference: layer_norm_fwd_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.
521: (Worker_TP0_EP0 pid=64527) WARNING 09-18 02:22:19 [jit_monitor.py:141] Triton kernel JIT compilation during inference: _compute_local_logits_stats_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.
522: (Worker_TP0_EP0 pid=64527) WARNING 09-18 02:22:19 [jit_monitor.py:141] Triton kernel JIT compilation during inference: _rejection_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.
523: (Worker_TP0_EP0 pid=64527) WARNING 09-18 02:22:20 [jit_monitor.py:141] Triton kernel JIT compilation during inference: _resample_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.
537: (APIServer pid=63493) INFO 09-18 02:22:24 [metrics.py:120] SpecDecoding metrics: Mean acceptance length: 3.00, Accepted throughput: 0.27 tokens/s, Drafted throughput: 0.40 tokens/s, Accepted: 6 tokens, Drafted: 9 tokens, Per-position acceptance rate: 0.667, 0.667, 0.667, Avg Draft acceptance rate: 66.7%
541: (APIServer pid=63493) INFO 09-18 02:22:34 [metrics.py:120] SpecDecoding metrics: Mean acceptance length: 2.93, Accepted throughput: 17.80 tokens/s, Drafted throughput: 27.60 tokens/s, Accepted: 178 tokens, Drafted: 276 tokens, Per-position acceptance rate: 0.826, 0.630, 0.478, Avg Draft acceptance rate: 64.5%
546: (APIServer pid=63493) INFO 09-18 02:22:44 [metrics.py:120] SpecDecoding metrics: Mean acceptance length: 2.76, Accepted throughput: 20.90 tokens/s, Drafted throughput: 35.70 tokens/s, Accepted: 209 tokens, Drafted: 357 tokens, Per-position acceptance rate: 0.790, 0.529, 0.437, Avg Draft acceptance rate: 58.5%
```

**E2. Autotuner calls the wrapper**

Source: <container>:<workdir>/vllm-w/benchmarks/kernels/benchmark_moe.py

SHA-256: ee32539ae5cd78b70b6f8a1e6d56528b56da8ab61233192fc94cef1db9fc98ee


```text
276:         with override_config(config):
277:             topk_weights, topk_ids, token_expert_indices = fused_topk(
278:                 x, input_gating, topk, renormalize=not use_deep_gemm
279:             )
280: 
281:             if use_deep_gemm:
282:                 return deep_gemm_experts.apply(
283:                     x,
284:                     w1,
285:                     w2,
286:                     topk_weights,
287:                     topk_ids,
288:                     activation=MoEActivation.SILU,
289:                     global_num_experts=num_experts,
290:                     apply_router_weight_on_input=False,
291:                     expert_map=False,
292:                 )
293:             return fused_experts(
294:                 x,
295:                 w1,
296:                 w2,
297:                 topk_weights,
298:                 topk_ids,
299:                 quant_config=quant_config,
300:             )
```

**E3. Skinny path bypasses the tile configuration**

Source: <container>:<workdir>/vllm-w/vllm/model_executor/layers/fused_moe/fused_moe.py

SHA-256: 432e409f07a55300465ed15983e4616f854ab2cb50e94252ff9e3e9c931ddea3


```text
1291: def _rocm_moe_skinny_available() -> bool:
1292:     global _ROCM_MOE_SKINNY
1293:     if _ROCM_MOE_SKINNY is None:
1294:         import os
1295: 
1296:         _ROCM_MOE_SKINNY = (
1297:             current_platform.is_rocm()
1298:             and os.environ.get("VLLM_ROCM_MOE_SKINNY", "1") == "1"
1299:             and hasattr(torch.ops._rocm_C, "moe_skinny_int4_decode")
1300:         )
1301:     return _ROCM_MOE_SKINNY
1302: 
1303: 
1304: def should_moe_wna16_use_cuda(
1305:     num_valid_tokens: int, group_size: int, num_experts: int, bit: int
1306: ):
1307:     if not (current_platform.is_cuda() or current_platform.is_rocm()):
1308:         return False
1309:     if current_platform.is_rocm() and not hasattr(torch.ops._moe_C, "moe_wna16_gemm"):
1310:         # older ROCm builds do not compile the kernel
1311:         return False
1312:     return (
1313:         bit == 4
1314:         and group_size in [32, 64, 128]
1315:         and num_valid_tokens / num_experts <= 6
1316:     )
```


```text
1452:     override_config = get_config()
1453:     if override_config:
1454:         config = override_config
1455:     else:
1456:         # First try to load optimal config from the file
1457:         E, _, N = w2_shape
1458:         if dtype == "int4_w4a16":
1459:             N = N * 2
1460:         block_n = block_shape[<bus>] if block_shape else 0
1461:         block_k = block_shape[<bus>] if block_shape else 0
1462:         configs = get_moe_configs(E, N, dtype, block_n, block_k)
1463: 
1464:         if configs:
1465:             # If an optimal configuration map has been found, look up the
1466:             # optimal config
1467:             config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
1468:         else:
1469:             # Else use the default config
1470:             config = get_default_config(M, E, N, w1_shape[<bus>], top_k, dtype, block_shape)
1471:     return config
```


```text
1629:     if (
1630:         _rocm_moe_skinny_available()
1631:         and quant_config.use_int4_w4a16
1632:         and quant_config.w1_zp is None
1633:         and quant_config.block_shape is not None
1634:         and expert_map is None
1635:         and not apply_router_weight_on_input
1636:         and activation == MoEActivation.SILU
1637:         and hidden_states.dtype == torch.float16
1638:         and hidden_states.shape[<bus>] <= 8
1639:         and global_num_experts in (-1, w1.shape[<bus>])
1640:     ):
1641:         # gfx1030 decode path: wave-per-row skinny GEMV pair (~432 GB/s
1642:         # effective vs ~40 GB/s-class for the tile kernels at these M).
1643:         M, K = hidden_states.shape
1644:         topk = topk_ids.shape[<bus>]
1645:         inter = w1.shape[<bus>] // 2
1646:         act_buf = torch.empty(
1647:             (M, topk, inter), dtype=torch.float16, device=hidden_states.device
1648:         )
1649:         out = torch.empty(
1650:             (M, K), dtype=torch.float16, device=hidden_states.device
1651:         )
1652:         ops.moe_skinny_int4_decode(
1653:             hidden_states.contiguous(),
1654:             w1,
1655:             quant_config.w1_scale,
1656:             w2,
1657:             quant_config.w2_scale,
1658:             topk_weights.to(torch.float32).contiguous(),
1659:             topk_ids.to(torch.int32).contiguous(),
1660:             act_buf,
1661:             out,
1662:             quant_config.block_shape[<bus>],
1663:         )
1664:         return out
1665: 
1666:     return torch.ops.vllm.fused_experts(
```

**E4. Surviving WAVES harness differs from production and reads beyond uint4**

Source: <container>:<workdir>/gemv_waves_sweep.cu; matching defect also exists in project tools/m0/gemv_waves_sweep.cu

SHA-256: 37de2df5f93e2f5182acfe1e8d8f777565edf823a8c2dee78e10c4fd42f65953


```text
27: // --- production kernel, verbatim from tools/rdna2/gemv_f16_harness.cu (wave-per-output-row) ---
28: template <int WAVES, int MT>
29: __global__ void __launch_bounds__(WAVES * 32)
30: gemv_f16_rdna2_(const half* __restrict__ x, const half* __restrict__ w,
31:                 const half* __restrict__ bias, half* __restrict__ y,
32:                 const int N, const int K) {
33:   const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
34:   const int n = blockIdx.x * WAVES + wave;
35:   if (n >= N) return;
36:   const int K8 = K / 8;
37:   const uint4* __restrict__ wrow = reinterpret_cast<const uint4*>(w + (size_t)n * K);
38:   const uint4* __restrict__ xr = reinterpret_cast<const uint4*>(x);
39:   float acc[MT];
40: #pragma unroll
41:   for (int m = 0; m < MT; m++) acc[m] = 0.f;
42:   for (int k = lane; k < K8; k += 32) {
43:     const uint4 wv = wrow[k];
44:     const half2* wp = reinterpret_cast<const half2*>(&wv);
45: #pragma unroll
46:     for (int m = 0; m < MT; m++) {
47:       const uint4 xv = xr[(size_t)m * K8 + k];
48:       const half2* xp = reinterpret_cast<const half2*>(&xv);
49: #pragma unroll
50:       for (int j = 0; j < 8; j++) {
51:         const half2 a = __hmul2(xp[j], wp[j]);
52:         acc[m] += __low2float(a) + __high2float(a);
53:       }
54:     }
55:   }
56: #pragma unroll
57:   for (int m = 0; m < MT; m++) {
58: #pragma unroll
59:     for (int off = 16; off > 0; off >>= 1)
60:       acc[m] += __shfl_down(acc[m], off, 32);
```

**E5-A. Saved PP/TG result**

Source: <container>:<workdir>/bench_A.json

```json
{
  "tag": "A",
  "prompt_tokens": 365,
  "gen": 128,
  "reps": 3,
  "pp": 350.33,
  "tg": 53.92,
  "tg_incl_prompt": 37.47,
  "wall_1tok_s": 1.0419,
  "wall_gen_s": 3.4156,
  "sha256": "2db606c96cce89e283d660e4619561217771abb625e38cace686b029cb3555db",
  "sha256_head400": "4d05639ce7abbcc56e15db7ccbb21364dfb987b88357a5c18059e02608634a74",
  "text_head": "\nassistant\n<think>\nThe user has repeated the same sentence 13 times. This appears to be either a test, a copy-paste error, or perhaps they want me to acknowledge the content. The sentence describes a "
}
```

**E5-B. Saved PP/TG result**

Source: <container>:<workdir>/bench_B.json

```json
{
  "tag": "B",
  "prompt_tokens": 365,
  "gen": 128,
  "reps": 3,
  "pp": 306.48,
  "tg": 56.04,
  "tg_incl_prompt": 36.83,
  "wall_1tok_s": 1.1909,
  "wall_gen_s": 3.4752,
  "sha256": "2db606c96cce89e283d660e4619561217771abb625e38cace686b029cb3555db",
  "sha256_head400": "4d05639ce7abbcc56e15db7ccbb21364dfb987b88357a5c18059e02608634a74",
  "text_head": "\nassistant\n<think>\nThe user has repeated the same sentence 13 times. This appears to be either a test, a copy-paste error, or perhaps they want me to acknowledge the content. The sentence describes a "
}
```

**E5-Aprime. Saved PP/TG result**

Source: <container>:<workdir>/bench_Aprime.json

```json
{
  "tag": "Aprime",
  "prompt_tokens": 365,
  "gen": 128,
  "reps": 3,
  "pp": 350.03,
  "tg": 52.64,
  "tg_incl_prompt": 36.84,
  "wall_1tok_s": 1.0428,
  "wall_gen_s": 3.4743,
  "sha256": "2db606c96cce89e283d660e4619561217771abb625e38cace686b029cb3555db",
  "sha256_head400": "4d05639ce7abbcc56e15db7ccbb21364dfb987b88357a5c18059e02608634a74",
  "text_head": "\nassistant\n<think>\nThe user has repeated the same sentence 13 times. This appears to be either a test, a copy-paste error, or perhaps they want me to acknowledge the content. The sentence describes a "
}
```

**E6. Request benchmark timing and hardcoded configuration**

Source: <mount>/smb/<lan-ip>/<share>/70_Code/qwen38-flash-next-mi50/tools/m0/tp4_gate.py

SHA-256: 478a578a02204a1fad2a6c02c7cfa685596ce37e5baeae0487eef8ca986c4681


```text
41:         "max_tokens": maxtok,
42:         "temperature": 0.0,
43:         "top_p": 1.0,
44:         "ignore_eos": True,          # TRAPS #6: EOS must not truncate a rate run
45:         "seed": 0,
46:     }
47:     t0 = time.perf_counter()
48:     out = post("/v1/completions", body)
49:     wall = time.perf_counter() - t0
50:     text = out["choices"][<bus>]["text"]
51:     usage = out.get("usage", {})
52:     ct = usage.get("completion_tokens")
53:     return {
54:         "tag": tag,
55:         "completion_tokens": ct,
56:         "wall_s": round(wall, 4),
57:         "tok_per_s": round(ct / wall, 3) if ct and wall else None,
58:         "first_64_tokens": text[:400],
59:         "sha256": hashlib.sha256(text.encode()).hexdigest(),
60:         "text_len_chars": len(text),
61:     }
62: 
63: 
64: def main():
65:     max_tokens = int(sys.argv[<bus>]) if len(sys.argv) > 1 else 64
66:     reps = int(sys.argv[<bus>]) if len(sys.argv) > 2 else 3
```


```text
94:         "correctness_a": a, "correctness_b": b,
95:         "baseline_runs": runs,
96:         "baseline_median_tok_per_s": med,
97:         "label": "BASELINE ONLY - not a tuned number",
98:     }
99:     with open("<workdir>/tp4_gate.json", "w") as fh:
100:         json.dump(result, fh, indent=2)
101:     print("\nwrote <workdir>/tp4_gate.json")
102: 
103: 
104: if __name__ == "__main__":
105:     main()
```

**E7. Prompt construction and subtractive timing**

Source: <mount>/smb/<lan-ip>/<share>/70_Code/qwen38-flash-next-mi50/tools/m0/bench_pp_tg.py

SHA-256: b612e1f03b89adce66f46a134f47ad46dc77c10eebd1b0999eb4da68815b9998


```text
46:         except Exception:
47:             ids = []
48:         if len(ids) >= n_tokens and ids:
49:             # rebuild from the exact prefix token count using character slicing is not exact;
50:             # instead grow a repeated unit and accept the achieved count (reported, not assumed).
51:             return text
52:         text += BASE_TEXT
53:     return text
54: 
55: 
56: def measure(prompt: str, max_tokens: int, reps: int):
57:     walls, toks = [], []
58:     text = None
59:     for _ in range(reps):
60:         body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
61:                 "temperature": 0.0, "top_p": 1.0, "ignore_eos": True, "seed": 0}
62:         t0 = time.perf_counter()
```


```text
90:     # warmup (these also warm the prefix cache; noted so PP is read with that in mind)
91:     measure(prompt, 1, 1)
92:     measure(prompt, 8, 1)
93: 
94:     w1, t1, _ = measure(prompt, 1, args.reps)
95:     wg, tg, text = measure(prompt, args.gen, args.reps)
96: 
97:     pp = (n_prompt / w1) if (n_prompt > 0 and w1 > 0) else None
98:     gen_time = wg - w1
99:     tg_rate = (args.gen / gen_time) if gen_time > 0 else None
100:     tg_wall = (args.gen / wg) if wg > 0 else None
101: 
102:     digest = hashlib.sha256((text or "").encode()).hexdigest()
103:     head = hashlib.sha256(((text or "")[:400]).encode()).hexdigest()
104: 
105:     print(f"  wall(1 tok)   = {w1:.4f} s   -> PP{n_prompt} = {pp:.1f} tok/s"
106:           if pp else "  PP unavailable")
107:     print(f"  wall({args.gen} tok) = {wg:.4f} s   -> generation {gen_time:.4f} s")
108:     print(f"  TG{args.gen} = {tg_rate:.2f} tok/s (prompt time subtracted)"
109:           if tg_rate else "  TG unavailable")
110:     print(f"  TG{args.gen} (incl. prompt) = {tg_wall:.2f} tok/s" if tg_wall else "")
111:     print(f"  completion tokens = {tg}")
```

**E8. Current GPU locality, link and power caps**

Read from root@<host-ip> at 2026-09-18T02:43:34.575236+00:00.

```text
/sys/class/drm/card1/device
1
16.0 GT/s PCIe
16
178000000
/sys/class/drm/card2/device
1
16.0 GT/s PCIe
16
178000000
/sys/class/drm/card3/device
1
16.0 GT/s PCIe
16
178000000
/sys/class/drm/card4/device
1
16.0 GT/s PCIe
16
178000000
```

**E9. Container CPU and memory-node allowance**

Read from root@<container-ip> at 2026-09-18T02:43:34.794756+00:00.

```text
Cpus_allowed_list:	3,8,11-14,18-19,22-23,25,30-33,36,38-41,47,54-55,57-58,64,68,71,76-77,79,81-82,84,89,93,102,104,109-110,115,121-123,128-129,131-132,134,136,140,144-145,147,152,156,158-159,161-162,167-168,179,183
Mems_allowed_list:	0-1
```

**E10. INT8 gate on the installed source**

Read from root@<container-ip> at 2026-09-18T02:43:34.996883+00:00.

```text
    31	
    32	def enabled() -> bool:
    33	    global _ENABLED
    34	    if _ENABLED is None:
    35	        from vllm.platforms import current_platform
    36	
    37	        on = os.getenv("VLLM_RDNA_DENSE_INT8", "0") == "1" and current_platform.is_rocm()
    38	        if on:
    39	            from vllm.platforms.rocm import on_gfx10x
    40	
    41	            on = on_gfx10x()
    42	        _ENABLED = on
    43	    return _ENABLED
```

**E11. Production wave64 GEMV implementation**

Read from root@<container-ip> at 2026-09-18T02:43:35.203685+00:00.

```text
  1015	template <int WAVES, int MT>
  1016	__global__ void __launch_bounds__(WAVES * 64)
  1017	gemv_f16_rdna2_(const half* __restrict__ x, const half* __restrict__ w,
  1018	                const half* __restrict__ bias, half* __restrict__ y,
  1019	                const int N, const int K) {
  1020	  const int wave = threadIdx.x / 64, lane = threadIdx.x % 64;
  1021	  const int n = blockIdx.x * WAVES + wave;
  1022	  if (n >= N) return;
  1023	  const int K8 = K / 8;
  1024	  const uint4* __restrict__ wrow =
  1025	      reinterpret_cast<const uint4*>(w + (size_t)n * K);
  1026	  const uint4* __restrict__ xr = reinterpret_cast<const uint4*>(x);
  1027	  float acc[MT];
  1028	#pragma unroll
  1029	  for (int m = 0; m < MT; m++) acc[m] = 0.f;
  1030	  for (int i = lane; i < K8; i += 64 * 4) {
  1031	    uint4 wq[<bus>];
  1032	#pragma unroll
  1033	    for (int u = 0; u < 4; u++) {
  1034	      const int idx = i + 64 * u;
  1035	      wq[u] = (idx < K8) ? wrow[idx] : make_uint4(0, 0, 0, 0);
  1036	    }
  1037	#pragma unroll
  1038	    for (int u = 0; u < 4; u++) {
  1039	      const int idx = i + 64 * u;
  1040	      if (idx < K8) {
  1041	        const half2* wh = reinterpret_cast<const half2*>(&wq[u]);
  1042	#pragma unroll
  1043	        for (int m = 0; m < MT; m++) {
  1044	          const uint4 xq = xr[(size_t)m * K8 + idx];
  1045	          const half2* xh = reinterpret_cast<const half2*>(&xq);
  1046	          float a = acc[m];
  1047	          a = __builtin_amdgcn_fdot2(wh[<bus>], xh[<bus>], a, false);
  1048	          a = __builtin_amdgcn_fdot2(wh[<bus>], xh[<bus>], a, false);
  1049	          a = __builtin_amdgcn_fdot2(wh[<bus>], xh[<bus>], a, false);
  1050	          a = __builtin_amdgcn_fdot2(wh[<bus>], xh[<bus>], a, false);
  1051	          acc[m] = a;
  1052	        }
  1053	      }
  1054	    }
  1055	  }
  1056	#pragma unroll
  1057	  for (int m = 0; m < MT; m++) {
  1058	#pragma unroll
  1059	    for (int off = 32; off >= 1; off >>= 1) acc[m] += __shfl_xor(acc[m], off);
  1060	  }
  1061	  if (lane == 0) {
  1062	    const float b = bias ? __half2float(bias[n]) : 0.f;
  1063	#pragma unroll
  1064	    for (int m = 0; m < MT; m++)
  1065	      y[(size_t)m * N + n] = __float2half(acc[m] + b);
  1066	  }
  1067	}
  1068	
  1069	template <int MT>
  1070	static void gemv_f16_rdna2_launch(const half* x, const half* w,
  1071	                                  const half* bias, half* y, int N, int K,
  1072	                                  cudaStream_t s) {
  1073	  // Fewer waves per block for small N so the grid still covers all CUs.
  1074	  if (N >= 1152)
  1075	    gemv_f16_rdna2_<8, MT><<<(N + 7) / 8, 512, 0, s>>>(x, w, bias, y, N, K);
  1076	  else if (N >= 576)
  1077	    gemv_f16_rdna2_<4, MT><<<(N + 3) / 4, 256, 0, s>>>(x, w, bias, y, N, K);
  1078	  else if (N >= 288)
  1079	    gemv_f16_rdna2_<2, MT><<<(N + 1) / 2, 128, 0, s>>>(x, w, bias, y, N, K);
  1080	  else
  1081	    gemv_f16_rdna2_<1, MT><<<N, 64, 0, s>>>(x, w, bias, y, N, K);
  1082	}
```

**E12. Serving expert path returns through the custom decode kernel**

Read from root@<container-ip> at 2026-09-18T02:43:35.432785+00:00.

```text
   253	        if (
   254	            self.quant_config.use_int4_w4a16
   255	            and hidden_states.dtype == torch.float16
   256	            and hidden_states.shape[<bus>] <= 8
   257	            and activation == MoEActivation.SILU
   258	            and not apply_router_weight_on_input
   259	            and self.quant_config.w1_zp is None
   260	            and self.block_shape is not None
   261	        ):
   262	            from vllm.model_executor.layers.fused_moe.fused_moe import (
   263	                _rocm_moe_skinny_available,
   264	            )
   265	
   266	            if _rocm_moe_skinny_available():
   267	                # gfx1030 decode path: wave-per-row skinny GEMV pair. Writes
   268	                # the fully topk-combined result into `output`, matching the
   269	                # invoke+moe_sum tail below.
   270	                M = hidden_states.shape[<bus>]
   271	                topk = topk_ids.shape[<bus>]
   272	                # T46: ids/weights go to the kernel as produced (int64/fp32 or
   273	                # fp16) and the EP expert_map is applied in-kernel: no index,
   274	                # convert or copy launches on the decode path.
   275	                inter = w1.size(1) // 2
   276	                act_buf = torch.empty(
   277	                    (M, topk, inter),
   278	                    dtype=torch.float16,
   279	                    device=hidden_states.device,
   280	                )
   281	                ops.moe_skinny_int4_decode(
   282	                    hidden_states,
   283	                    w1,
   284	                    self.quant_config.w1_scale,
   285	                    w2,
   286	                    self.quant_config.w2_scale,
   287	                    topk_weights.contiguous(),
   288	                    topk_ids.contiguous(),
   289	                    act_buf,
   290	                    output,
   291	                    self.block_shape[<bus>],
   292	                    expert_map,
   293	                )
   294	                return
   295	
   296	        # Check constraints.
   297	        if self.quant_config.use_int4_w4a16:
```

**Project decision references**

- DECISIONS.md D142 addendum 3, lines 4105–4165: custom AR comparison.
- D143, lines 4167–4259: ledger, wall accounting and collective ceiling.
- D144, lines 4416–4504: tuner and rejected JSON.
- D145, lines 4506–4629: WAVES, dense sizing and unroll claims.
- D147, lines 4828–4903: MTP and PP/TG conclusions.
- PROGRESS.md: current status, superseding entries and remaining contradictory gate rows.

These are references into the project supplied by the user; they have not been edited.
