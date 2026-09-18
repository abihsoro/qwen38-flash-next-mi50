#!/usr/bin/env python3
"""capability_check.py — N-C9: fail-loud capability banner (I1 enforcement).

Prints the state of each optimized path and exits non-zero if a REQUESTED
path is unavailable. The harness refuses to write results rows when a
requested path is inactive — a silently-degraded run (the silent-RCCL-
fallback / inductor-miscompile class) must NOT produce a measurement.

Paths detected on the current hardware + config:
  skinny_gemv   gemv_f16_rdna2 op (wave64 port; validated N-D1..D4)
  w4_moe        moe_skinny_int4_decode op (EP-aware int4 MoE; N-D3/D036)
  p2p_ar        rdna_ar_* ops (custom one-shot all-reduce; world=1 on 1 card)
  ple_offload   VLLM_PLE_CPU_OFFLOAD env + ple_offload_wait op registered
  int8_dense    gemv_i8_rdna2 op (FLAG: N-D5 deferred — unvalidated on gfx906)
  qsa           qwen4_exp_qsa_with_output op (the model's QSA transaction)
  graph_mode    cudagraphs active (compile ON + cudagraph != NONE)

Usage:
  capability_check.py --require skinny_gemv,w4_moe,ple_offload
  (prints the banner + a machine-readable state to stdout)
"""
import argparse
import json
import os
import sys

PATHS = [
    "skinny_gemv", "w4_moe", "p2p_ar", "ple_offload", "int8_dense",
    "qsa", "graph_mode",
]


def detect() -> dict:
    """Return {path: {"ok": bool, "note": str}} on this machine+config."""
    out = {}

    # --- ops that load with vllm._custom_ops ---
    try:
        import vllm._custom_ops as c  # noqa: F401
        ops_loaded = True
    except Exception as e:  # noqa: BLE001
        ops_loaded = False
        ops_err = str(e)[-160:]

    def has_op(name):
        if not ops_loaded:
            return False
        try:
            import vllm._custom_ops as c2
            return hasattr(c2, name)
        except Exception:  # noqa: BLE001
            return False

    out["skinny_gemv"] = {
        "ok": has_op("gemv_f16_rdna2"),
        "note": "gemv_f16_rdna2 (wave64; N-D1..D4 validated)" if has_op(
            "gemv_f16_rdna2") else f"gemv_f16_rdna2 missing ({ops_err if not ops_loaded else ''})",
    }
    out["w4_moe"] = {
        "ok": has_op("moe_skinny_int4_decode"),
        "note": "moe_skinny_int4_decode (EP-aware; N-D3/D036)" if has_op(
            "moe_skinny_int4_decode") else "moe_skinny_int4_decode missing",
    }
    ar_ok = all(has_op(n) for n in
                ("rdna_ar_init", "rdna_ar_connect", "rdna_ar_can",
                 "rdna_ar_all_reduce"))
    out["p2p_ar"] = {
        "ok": ar_ok,
        "note": "rdna_ar_* registered (world=1 degenerate PASS, N-D3)"
        if ar_ok else "rdna_ar_* incomplete",
    }
    out["int8_dense"] = {
        "ok": has_op("gemv_i8_rdna2"),
        "note": ("gemv_i8_rdna2 present but UNVALIDATED on gfx906 "
                 "(N-D5 deferred) - require with care")
        if has_op("gemv_i8_rdna2") else "gemv_i8_rdna2 missing",
    }

    # --- model ops that register on import (import FIRST: the module-init
    # order matters - vllm._custom_ops pulls a chain that can leave qsa
    # partially initialized if imported later) ---
    try:
        import vllm.models.qwen4_exp.amd.qsa as _q  # noqa: F401
        import torch
        qsa_ok = hasattr(torch.ops.vllm, "qwen4_exp_qsa_with_output")
        qsa_err = ""
    except Exception as e:  # noqa: BLE001
        qsa_ok = False
        qsa_err = str(e)[-200:]
    out["qsa"] = {
        "ok": qsa_ok,
        "note": "qwen4_exp_qsa_with_output registered" if qsa_ok else
        ("engine-time op; standalone import blocked by module-init order "
         "(verified at serve; N-C8 prefill ladder passed) - require only in "
         "the serve wrapper" + (f" | {qsa_err}" if qsa_err else "")),
    }

    # --- PLE offload: env + the wait op ---
    try:
        import vllm.model_executor.layers.ple_offload_layer as _p  # noqa: F401
        import torch
        ple_op = hasattr(torch.ops.vllm, "ple_offload_wait")
    except Exception as e:  # noqa: BLE001
        ple_op = False
        ple_err = str(e)[-160:]
    env_ok = bool(os.environ.get("VLLM_PLE_CPU_OFFLOAD")) and bool(
        os.environ.get("VLLM_PLE_QUANT_DIR"))
    out["ple_offload"] = {
        "ok": ple_op and env_ok,
        "note": (f"ple_offload_wait op={'yes' if ple_op else 'NO'} "
                 f"env={'set' if env_ok else 'UNSET (VLLM_PLE_CPU_OFFLOAD/VLLM_PLE_QUANT_DIR)'}"),
    }

    # --- graph mode: compile + cudagraph active ---
    cc_mode = os.environ.get("VLLM_COMPILE_MODE") or ""
    graph = "off"
    out["graph_mode"] = {
        "ok": graph == "on",
        "note": "graph mode is a serve-config property; pass --graph on/off",
    }
    return out


def banner(state: dict, required: list) -> int:
    print("=== capability banner (fail-loud, N-C9) ===")
    code = 0
    for p in PATHS:
        s = state.get(p, {"ok": False, "note": "unknown"})
        req = " [REQUIRED]" if p in required else ""
        mark = "ACTIVE" if s["ok"] else "INACTIVE"
        print(f"  {p:12s} {mark:8s}{req}  {s.get('note', '')}")
        if p in required and not s["ok"]:
            code = 1
    if code:
        print("FAIL-LOUD: a REQUESTED path is unavailable - refusing to run.")
    return code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--require", default="",
                    help="comma list of paths that MUST be active")
    ap.add_argument("--graph", default="off", choices=["on", "off"])
    ap.add_argument("--write-state", default="",
                    help="write the machine-readable state to this json")
    args = ap.parse_args()
    required = [p for p in args.require.split(",") if p]
    state = detect()
    state["graph_mode"]["ok"] = args.graph == "on"
    state["graph_mode"]["note"] = f"cudagraphs={'on' if args.graph=='on' else 'off'}"
    if args.write_state:
        os.makedirs(os.path.dirname(args.write_state) or ".", exist_ok=True)
        with open(args.write_state, "w") as f:
            json.dump({"required": required, "state": state}, f, indent=2)
    return banner(state, required)


if __name__ == "__main__":
    raise SystemExit(main())
