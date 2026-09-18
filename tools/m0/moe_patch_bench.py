#!/usr/bin/env python3
"""moe_patch_bench.py — minimal compatibility patch to vLLM's MoE tuner for this model.

WHY: benchmark_moe.py:get_model_params() assumes Mixtral-style attribute names. This model's text
config class (Qwen4ExpTextConfig) exposes:
    num_experts = 512              (not num_local_experts)
    moe_intermediate_size = 640    (not intermediate_size)
so the tuner dies with AttributeError before it tunes anything. Verified against the model config.

The patch prefers the ORIGINAL upstream names and only falls back, so it is a no-op for the
Mixtral/Llama4 models the script was written for.

Run INSIDE the container. Idempotent: re-running detects the marker and does nothing.
"""
import shutil
import sys
from pathlib import Path

B = Path("<workdir>/vllm-w/benchmarks/kernels/benchmark_moe.py")
MARK = "# --- LOCAL PATCH (MI50 port)"

OLD = """        # Support for llama4
        config = config.get_text_config()
        # Default: Mixtral.
        E = config.num_local_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
        hidden_size = config.hidden_size
"""

NEW = """        # Support for llama4
        config = config.get_text_config()
        # Default: Mixtral.
        # --- LOCAL PATCH (MI50 port) --------------------------------------------------
        # Qwen4ExpTextConfig exposes 'num_experts' (global) instead of Mixtral's
        # 'num_local_experts', and 'moe_intermediate_size' instead of 'intermediate_size'.
        # Upstream names are tried FIRST so this is a no-op for Mixtral/Llama4.
        # Note: E here must be the GLOBAL expert count; main() divides by tp_size under
        # --enable-expert-parallel, giving 512//4 = 128, and the saved filename becomes
        # E=128,N=640,device_name=<gpu>,dtype=int4_w4a16.json - the name the serve looks up.
        E = getattr(config, "num_local_experts", None)
        if E is None:
            E = getattr(config, "num_experts", None)
        if E is None:
            raise AttributeError(
                "config exposes neither num_local_experts nor num_experts"
            )
        topk = config.num_experts_per_tok
        # This script tunes MoE EXPERTS, so the MoE expert width is the correct value.
        # Qwen4ExpTextConfig carries BOTH: intermediate_size = 5632 (dense FFN width) and
        # moe_intermediate_size = 640 (expert width). Taking intermediate_size first yields
        # N=5632 and writes a filename the serve never looks up (the serve derives N from the
        # MoE layer's w2.shape[<bus>] = 640). So prefer the MoE width, then fall back.
        intermediate_size = getattr(config, "moe_intermediate_size", None)
        if intermediate_size is None:
            intermediate_size = getattr(config, "intermediate_size", None)
        if intermediate_size is None:
            raise AttributeError(
                "config exposes neither moe_intermediate_size nor intermediate_size"
            )
        hidden_size = config.hidden_size
        # --- END LOCAL PATCH -----------------------------------------------------------
"""


OLD2 = """    if current_platform.is_rocm() and "HIP_VISIBLE_DEVICES" in os.environ:
        # Ray will set ROCR_VISIBLE_DEVICES for device visibility
        logger.warning(
            "Ray uses ROCR_VISIBLE_DEVICES to control device accessibility."
            "Replacing HIP_VISIBLE_DEVICES with ROCR_VISIBLE_DEVICES."
        )
        val = os.environ["HIP_VISIBLE_DEVICES"]
        os.environ["ROCR_VISIBLE_DEVICES"] = val
        del os.environ["HIP_VISIBLE_DEVICES"]
"""

NEW2 = """    if current_platform.is_rocm() and "HIP_VISIBLE_DEVICES" in os.environ:
        # --- LOCAL PATCH (MI50 port) --------------------------------------------------
        # The original code ALWAYS swapped HIP_VISIBLE_DEVICES -> ROCR_VISIBLE_DEVICES,
        # because OLD Ray only honoured ROCR. Modern Ray is the exact opposite: see
        # ray/_private/accelerators/amd_gpu.py:40-48, which RAISES
        #   "Please use HIP_VISIBLE_DEVICES instead of ROCR_VISIBLE_DEVICES"
        # whenever ROCR is present AND HIP is absent. So the unconditional swap makes the
        # script fail against ray 2.58 by construction - it deletes the very variable Ray
        # now requires and sets the one Ray now rejects.
        # Gate on the Ray version: modern Ray keeps HIP_VISIBLE_DEVICES untouched.
        import ray as _ray_mod
        _rv = tuple(int(p) for p in _ray_mod.__version__.split(".")[:2] if p.isdigit())
        if _rv >= (2, 9):
            logger.info(
                "Ray %s honours HIP_VISIBLE_DEVICES; leaving it in place (MI50 port patch).",
                _ray_mod.__version__,
            )
        else:
            logger.warning(
                "Ray uses ROCR_VISIBLE_DEVICES to control device accessibility."
                "Replacing HIP_VISIBLE_DEVICES with ROCR_VISIBLE_DEVICES."
            )
            val = os.environ["HIP_VISIBLE_DEVICES"]
            os.environ["ROCR_VISIBLE_DEVICES"] = val
            del os.environ["HIP_VISIBLE_DEVICES"]
        # --- END LOCAL PATCH -----------------------------------------------------------
"""


def main() -> int:
    src = B.read_text()
    backup = str(B) + ".orig-mi50patch"
    # If a previous (possibly wrong) patch is present, restore the pristine file first so the
    # patch can be re-applied with corrected logic. Idempotent either way.
    if MARK in src:
        if Path(backup).exists():
            shutil.copy2(backup, B)
            src = B.read_text()
            print("restored pristine file from backup; re-applying corrected patch")
        else:
            print("already patched and no backup available - nothing to do")
            return 0
    if OLD not in src:
        print("ANCHOR NOT FOUND - refusing to patch blind")
        print("expected block:")
        print(OLD)
        return 2

    shutil.copy2(B, backup)
    out = src.replace(OLD, NEW, 1)
    if OLD2 not in out:
        print("SECOND ANCHOR NOT FOUND (ray HIP/ROCR swap) - refusing to patch blind")
        return 3
    out = out.replace(OLD2, NEW2, 1)
    B.write_text(out)
    print(f"patched {B} (both hunks)")
    print(f"backup  {backup}")

    # show the result
    out = B.read_text().splitlines()
    for i, line in enumerate(out, 1):
        if "LOCAL PATCH" in line:
            for j in range(max(1, i - 3), min(len(out), i + 26)):
                print(f"  {j}: {out[j-1]}")
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
