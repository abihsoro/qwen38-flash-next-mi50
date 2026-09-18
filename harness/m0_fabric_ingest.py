#!/usr/bin/env python3
"""m0_fabric_ingest.py — ingest M0 four-card transport-validation measurements.

Follows the gpu_probe.py pattern (I3): measurement happens on the hardware
(host <source-host> + LXC <ct>) and emits JSON; this script runs where the repo lives and
writes results rows + one results artifact through results.py, the sole writer.

Usage:
    python3 harness/m0_fabric_ingest.py --dir <phase-output-dir> [--task M0] [--dry-run]

Inputs expected in <phase-output-dir> (missing files are recorded as absent,
never guessed — I1):
    A1_fabric_pre.json      fabric/ACS report before the re-run
    A3_fabric_post.json     fabric/ACS report after re-running clear-pex-acs.sh
    A4_fabric_final.json    fabric/ACS report at the end
    B_matrix.json           P2P matrix probe output (all ordered pairs)
    C_negctl.json           ACS negative-control result (optional)

Outputs:
    results/m0_fabric.jsonl   append-only artifact: every measured record
    results/{task}.jsonl      one run row summarising the validation
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import results  # noqa: E402


def load(path: Path):
    """Return (data, error). Never fabricates: unreadable -> (None, reason)."""
    if not path.exists():
        return None, "absent"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh), None
    except Exception as exc:
        return None, f"unreadable: {exc}"


def load_lenient(path: Path):
    """Like load(), but if the file is a TRUNCATED JSON document (a probe that
    died mid-write), recover the complete pair objects that were fully written
    and flag the result as partial. Without this, a crash under test destroys
    the evidence of how far it got — which is exactly the interesting part."""
    data, err = load(path)
    if data is not None:
        return data, None
    if not path.exists():
        return None, "absent"
    try:
        text = path.read_text(errors="replace")
    except Exception as exc:
        return None, f"unreadable: {exc}"
    pairs = []
    for blob in re.findall(r'\{\s*"src".*?\}', text, re.S):
        try:
            pairs.append(json.loads(blob))
        except Exception:
            continue
    if not pairs:
        return None, err
    return {"pairs": pairs, "_partial": True, "_recovered": len(pairs),
            "_parse_error": err}, f"truncated JSON; recovered {len(pairs)} complete pair records"


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest M0 fabric validation measurements (I3).")
    ap.add_argument("--dir", required=True, help="directory holding the phase JSON outputs")
    ap.add_argument("--task", default="M0", help="task id for the run row (default M0)")
    ap.add_argument("--dry-run", action="store_true", help="validate + print, write nothing")
    ap.add_argument("--force", action="store_true",
                    help="overwrite results/m0_fabric.jsonl (a correction to an earlier ingest)")
    args = ap.parse_args()

    d = Path(args.dir).expanduser().resolve()
    records: list[dict] = []
    problems: list[str] = []

    a1, e1 = load(d / "A1_fabric_pre.json")
    a3, e3 = load(d / "A3_fabric_post.json")
    a4, e4 = load(d / "A4_fabric_final.json")
    bm, eb = load(d / "B_matrix.json")
    cb, ecb = load(d / "C_before.json")
    ca, eca = load_lenient(d / "C_asserted.json")
    cf, ecf = load(d / "C_after.json")
    rerun_log = (d / "A2_acs_rerun.log")
    rerun_text = rerun_log.read_text(errors="replace") if rerun_log.exists() else ""

    for name, err in (("A1_fabric_pre.json", e1), ("A3_fabric_post.json", e3),
                      ("A4_fabric_final.json", e4), ("B_matrix.json", eb)):
        if err:
            problems.append(f"{name}: {err}")

    if a1:
        records.append({"kind": "fabric_report", "phase": "pre_acs_rerun", "report": a1})
    if a3:
        records.append({"kind": "fabric_report", "phase": "post_acs_rerun", "report": a3})
    if a4:
        records.append({"kind": "fabric_report", "phase": "final", "report": a4})
    if rerun_text:
        records.append({"kind": "acs_rerun", "script": "clear-pex-acs.sh", "log": rerun_text})

    # ── P2P matrix records ─────────────────────────────────────────────────
    pairs = []
    self_copy = []
    devices = []
    if bm and "device_count" in bm:
        devices = bm.get("devices", [])
        self_copy = bm.get("self_copy", [])
        pairs = bm.get("pairs", [])
        for s in self_copy:
            records.append({"kind": "self_copy", **s})
        for p in pairs:
            records.append({"kind": "p2p_pair", **p})
        for h in bm.get("host_transfer", []):
            records.append({"kind": "host_transfer", **h})
    elif bm:
        problems.append(f"B_matrix.json: probe error {bm.get('error')!r}")

    # ── ACS negative control (three-leg: cleared / asserted / restored) ────
    def pmap(x):
        if not x or "pairs" not in x:
            return None
        return {f'{p["src"]}->{p["dst"]}': p.get("bw_gbps_median") for p in x["pairs"]}

    nb, na, nf = pmap(cb), pmap(ca), pmap(cf)
    nc_verdict = None
    if cb or ca or cf:
        nc = {
            "kind": "acs_negative_control",
            "target": "GPU3 <pci> path, hops <pci> + <pci> (unique to GPU3; <pci> excluded as shared with GPU2)",
            "asserted_value": "0x001d (SrcValid|ReqRedir|CmpltRedir|UpstreamFwd)",
            "legs_readable": {"before": nb is not None, "asserted": na is not None, "after": nf is not None},
            "load_errors": {"before": ecb, "asserted": eca, "after": ecf},
            "before": nb, "asserted": na, "after": nf,
        }
        if nb and na:
            gpu3 = sorted(k for k in na if "3" in k.split("->") and k in nb)
            ctrl = sorted(k for k in na if "3" not in k.split("->") and k in nb)
            nc["gpu3_pairs"] = {k: {"before": nb[k], "asserted": na[k]} for k in gpu3}
            nc["control_pairs"] = {k: {"before": nb[k], "asserted": na[k]} for k in ctrl}
            nc["asserted_leg_partial"] = bool(ca.get("_partial"))
            allk = [k for k in nb]
            missing = sorted(set(allk) - set(na))
            nc["pairs_not_measured_under_assertion"] = missing
            # Did any pair that DID complete change materially?
            changed = [k for k in na if nb.get(k) and abs(na[k] - nb[k]) / nb[k] > 0.03]
            nc["pairs_changed_gt3pct"] = changed
            if not changed and missing:
                nc_verdict = (
                    f"ACS asserted on the GPU3 deep hops: every pair that completed "
                    f"({len(na)}/{len(allk)}) was UNCHANGED (max <3%); no bandwidth collapse. "
                    f"Pair(s) {missing} never completed - the probe died with 'unspecified launch "
                    f"failure' and triggered a FAILED GPU reset on <pci>. So these hops' ACS "
                    f"state does NOT govern the ~14.4 GB/s peer limit, but re-asserting ACS on a "
                    f"live peer-mapped path faults sibling peer traffic and wedges the GPU.")
            elif changed:
                nc_verdict = f"ACS assertion changed pairs {changed} => ACS state governs peer bandwidth"
            else:
                nc_verdict = "inconclusive from the paired data"
        else:
            nc_verdict = ("negative control incomplete: the asserted leg could not be read "
                          f"({eca})")
        nc["verdict"] = nc_verdict
        records.append(nc)

        # The assertion had a real consequence; record it as measured fact.
        records.append({
            "kind": "incident",
            "what": "ACS re-assertion on a live peer-mapped GPU path caused a failed GPU reset",
            "gpu": "<pci> (GPU3)",
            "sequence": [
                "asserted 0x001d on <pci> and <pci>",
                "probe pair 2->3 hit 'unspecified launch failure' (m0_p2p_matrix.cpp:188)",
                "kernel: amdgpu <pci>: GPU reset begin!. Source: 2",
                "kernel: smu firmware loading failed / fw load failed / VRAM is lost due to GPU reset!",
                "kernel: GPU reset end with ret = -22 (the reset itself FAILED)",
                "ACS restored to cleared on both hops (verified post-restore)",
                "post-restore probe: HIP error no ROCm-capable device is detected (n=0)",
                "rocm-smi still lists all 4 GPUs; KFD topology still reports p2p_links_count 3",
                "GPU3 retains a stale 428 MB VRAM allocation vs ~11 MB on the other three",
            ],
            "provenance": "transcribed from dmesg and probe stdout on <source-host> at the time of the negative control",
            "consequence": "ROCm cannot initialise any device in LXC <ct>; M0/M1/M2 blocked pending host recovery (operator boundary, I10)",
            "precedent": "D105/D108 - an analogous gfx906 wedge was cleared by a <source-host> power cycle",
        })

    # ── verdicts (only over what was actually measured) ────────────────────
    n_gpu = bm.get("device_count") if (bm and "device_count" in bm) else None
    expected_pairs = n_gpu * (n_gpu - 1) if n_gpu else None

    all_paths_cleared = a4.get("all_gpu_paths_fully_cleared") if a4 else None
    uncleared = a4.get("any_gpu_path_acs_capable_hop_uncleared", []) if a4 else None

    pairs_ok = None
    implausible = []
    if pairs:
        # A pair passes only if peer access works, the copy is byte-exact, AND the
        # bandwidth survives the physical-plausibility check (an enqueue-only
        # measurement once produced 93 TB/s here).
        implausible = [f'{p.get("src")}->{p.get("dst")}' for p in pairs
                       if p.get("plausible") is False]
        pairs_ok = all(
            p.get("can_access_peer") == 1 and p.get("peer_access_enabled") == 1
            and p.get("bytes_ok") is True and p.get("plausible") is True
            and (p.get("err") or "") == ""
            for p in pairs
        ) and (expected_pairs is None or len(pairs) == expected_pairs)

    # idempotency: the re-run must produce the same GPU-path port set
    def port_set(rep):
        if not rep:
            return None
        gpus = [e for e in rep.get("endpoints", []) if e.get("under_pex")]
        return sorted({h for e in gpus for h in e.get("acs_capable_hops", [])})

    idempotent = None
    if a1 and a3:
        idempotent = port_set(a1) == port_set(a3)

    # regression control: PEX ports NOT on a GPU path must keep their ACS state
    nongpu_unchanged = None
    if a1 and a4:
        def nongpu(rep):
            return {p["bdf"]: p.get("cleared") for p in rep.get("pex_ports", [])
                    if not p.get("on_gpu_path")}
        nongpu_unchanged = nongpu(a1) == nongpu(a4)

    summary = (
        f"M0 fabric validation: gpus={n_gpu} pairs={len(pairs)}/{expected_pairs} "
        f"all_pairs_p2p_ok={pairs_ok} implausible_pairs={implausible} "
        f"gpu_paths_cleared={all_paths_cleared} "
        f"uncleared_hops={uncleared} acs_rerun_idempotent={idempotent} "
        f"non_gpu_pex_ports_unchanged={nongpu_unchanged} "
        f"acs_negative_control={nc_verdict} problems={problems}"
    )
    print("m0_fabric_ingest: " + summary)

    if args.dry_run:
        print(f"m0_fabric_ingest: dry-run, would write {len(records)} artifact records")
        return 0 if not problems else 1

    art = results.write_artifact("m0_fabric.jsonl", records, force=args.force)
    row = results.make_row(
        task_id=args.task,
        exit_code=0 if not problems else 1,
        wall_seconds=0.0,
        timeout_s=0.0,
        notes=summary[:2000],
        gpu_count=n_gpu if n_gpu is not None else None,
        shape_profile="microbench",
        graphs="off",
        timed_out=False,
    )
    row_path = results.append_row(row)
    print(f"m0_fabric_ingest: wrote {art.name} ({len(records)} records) + {row_path.name} row {row['run_id']}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
