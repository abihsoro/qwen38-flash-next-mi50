#!/usr/bin/env python3
"""ray_torch_probe.py — is ray+torch fundamentally broken here, or only under GPU assignment?

Three cases, in order of increasing GPU involvement:
  A) ray task that does NOT import torch            -> is ray itself OK?
  B) ray task that imports torch, NO num_gpus       -> does torch import crash in a worker?
  C) ray task with num_gpus=1 that imports torch    -> is it the GPU assignment?
This separates "ray is unusable with this torch" from "ray's GPU pinning is unusable".
"""
import os
import sys

import ray

print("driver: ROCR_VISIBLE_DEVICES =", os.environ.get("ROCR_VISIBLE_DEVICES", "<unset>"))
print("driver: HIP_VISIBLE_DEVICES  =", os.environ.get("HIP_VISIBLE_DEVICES", "<unset>"))
print("driver: importing torch...")
import torch  # noqa: E402
print("driver: torch", torch.__version__, "devices", torch.cuda.device_count())

ray.init(ignore_reinit_error=True, include_dashboard=False, logging_level=40)
print("driver: ray initialized; GPUs seen =", int(ray.available_resources().get("GPU", 0)))
print()


@ray.remote
def a_no_torch():
    return "ray works without torch"


@ray.remote
def b_import_torch():
    import torch
    return f"torch {torch.__version__} devices={torch.cuda.device_count()}"


@ray.remote(num_gpus=1)
def c_import_torch_gpu():
    import torch
    return f"torch {torch.__version__} devices={torch.cuda.device_count()}"


for name, fn in (("A no-torch", a_no_torch), ("B torch no-gpu", b_import_torch),
                 ("C torch num_gpus=1", c_import_torch_gpu)):
    try:
        out = ray.get(fn.remote(), timeout=180)
        print(f"  {name:20} OK   -> {out}")
    except Exception as e:
        print(f"  {name:20} FAIL -> {type(e).__name__}: {str(e)[:140]}")

ray.shutdown()
print("\nprobe done")
sys.exit(0)
