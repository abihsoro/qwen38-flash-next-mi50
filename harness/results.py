"""results.py — the sole writer of results/ (invariant I3).

Run rows conform to harness/results_schema.json (schema v2, D048): every field is
present on every row (null when not measured, never a guess - I1). Environment
fields are snapshotted from config/environment.lock at write time (I8);
git_commit/dirty_flag are read fresh. steps_per_second is the acceptance metric;
tokens_per_second is recorded and never governs (D048/I5). v1 rows remain
readable (schema_version 1) but are inadmissible for acceptance.

Only harness code may call these functions. The executor never hand-writes, edits,
or synthesises a results row (I3). If a number is not measured, the value is
"not measured" or null — never a guess (I1).
"""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
ENV_LOCK = ROOT / "config" / "environment.lock"
SCHEMA_PATH = ROOT / "harness" / "results_schema.json"

NOT_MEASURED = "not measured"

# Row fields that snapshot the environment fingerprint (I8).
ENV_FIELDS = [
    "rocm_version", "torch_version", "triton_commit", "vllm_commit",
    "kernel_version", "gfx_target", "guest_kernel", "pve_version", "gpu_count",
]

# Fields that are null unless a task supplies them (config or measurement).
MEASUREMENT_FIELDS = [
    "tp", "ep", "mtp", "power_cap_w", "observed_power_w", "gpu_clock_mhz",
    "hbm_clock_mhz", "temp_c", "soak_seconds", "context", "gen_tokens",
    "repeat_index", "tokens_per_second", "prefill_tps", "ttft_ms",
    "kernel_config", "correctness_hash",
    # schema v2 (D048)
    "steps_per_second", "accepted_tokens_per_step", "forward_steps",
    "decode_wall_seconds", "shape_profile", "graphs", "dtype", "ple_source",
    "ple_offload", "thermally_valid", "clock_deviation_pct", "timed_out",
    "tp4_projection_factor",
]


def _load_json(path: Path, default):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def load_env_lock() -> dict:
    """Load config/environment.lock; {} if missing or unreadable."""
    return _load_json(ENV_LOCK, {})


def load_schema() -> dict:
    """Load the results row schema."""
    return _load_json(SCHEMA_PATH, {})


def git_state() -> tuple[str, bool]:
    """(HEAD commit, dirty_flag) for the project repo, measured fresh (I1)."""
    try:
        head = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        commit = head.stdout.strip() if head.returncode == 0 else NOT_MEASURED
        status = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        )
        dirty = status.returncode == 0 and bool(status.stdout.strip())
        return commit, dirty
    except Exception:
        return NOT_MEASURED, True


def make_row(task_id: str, exit_code: int, wall_seconds: float, timeout_s: float,
             notes: str = "", **extra) -> dict:
    """Build a complete 34-field run row. Unknown extra keys are rejected."""
    env = load_env_lock()
    commit, dirty = git_state()
    row = {
        "run_id": uuid.uuid4().hex,
        "task_id": task_id,
        "git_commit": commit,
        "dirty_flag": dirty,
        "schema_version": 2,
    }
    for field in ENV_FIELDS:
        row[field] = env.get(field, NOT_MEASURED)
    for field in MEASUREMENT_FIELDS:
        row[field] = None
    row.update({
        "exit_code": exit_code,
        "wall_seconds": wall_seconds,
        "timeout_s": timeout_s,
        "notes": notes,
    })
    required_paths = extra.pop("required_paths", None)
    row.update(extra)
    validate_row(row)
    if required_paths:
        row["required_paths"] = list(required_paths)
    return row


def validate_row(row: dict) -> None:
    """Raise ValueError unless row conforms to results_schema.json.

    v2 rows (schema_version == 2) must carry every schema property (null when
    not measured). Rows without schema_version (the pre-D048 writer, v1) stay
    readable: only the fields they carry must be schema-valid (D048 migration:
    do not backfill what was not measured - I1).
    """
    schema = load_schema()
    props = schema.get("properties", {})
    is_v2 = row.get("schema_version") == 2
    if is_v2:
        for field in props:
            if field not in row:
                raise ValueError(f"row missing required field: {field}")
    for key in row:
        if key not in props:
            if key == "tps" and not is_v2:
                continue  # legacy v1 field (renamed to tokens_per_second in v2)
            raise ValueError(f"row has unknown field: {key}")
    for field in schema.get("required", []):
        if field == "schema_version" and not is_v2:
            continue  # pre-v2 rows carry no stamp; readable, inadmissible (D048)
        if row.get(field) in (None, ""):
            raise ValueError(f"required field is null/empty: {field}")
    # v2 conditional enforcement (D048/I5): performance rows, MTP rows,
    # tp4_rank0 projection; thermally_valid/full-profile admissibility is the
    # acceptance calculator's job (this stays the validation layer).
    if row.get("schema_version") == 2:
        if row.get("shape_profile") == "tp4_rank0" and row.get("tp4_projection_factor") is None:
            raise ValueError("shape_profile tp4_rank0 requires tp4_projection_factor")
        if row.get("mtp") is not None and row.get("mtp", 0) >= 1 and row.get("accepted_tokens_per_step") is None:
            raise ValueError("mtp >= 1 rows require accepted_tokens_per_step (I5/D048)")
    for field, spec in props.items():
        if field not in row:
            if not is_v2:
                continue  # v1 rows carry only their own fields
            raise ValueError(f"row missing required field: {field}")
        value = row[field]
        if value is None:
            continue
        allowed = spec.get("type")
        if isinstance(allowed, list):
            ok = any(_type_ok(value, t) for t in allowed)
        else:
            ok = _type_ok(value, allowed)
        if not ok:
            raise ValueError(f"field {field}: value {value!r} not of type {allowed}")
        if "enum" in spec and value not in spec["enum"]:
            raise ValueError(f"field {field}: {value!r} not in {spec['enum']}")


def _type_ok(value, type_name: str) -> bool:
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "null":
        return value is None
    return True


def _existing_run_ids(task_id: str) -> set:
    path = RESULTS_DIR / f"{task_id}.jsonl"
    if not path.exists():
        return set()
    ids = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["run_id"])
            except Exception:
                continue
    return ids


def _capability_gate(row: dict, req: list | None) -> None:
    """N-C9 fail-loud: refuse a row whose REQUIRED capability is inactive.

    The row's required_paths (a make_row param) names optimized paths that
    must be ACTIVE for the measurement to mean what it claims (I1: no claim
    without artifact). The gate consults state/capabilities.json (written by
    tools/capability/capability_check.py at serve start); a missing state
    file means no banner ran -> refuse (a silently-degraded run must not
    produce a measurement).
    """
    if not req:
        return
    state_path = ROOT / "state" / "capabilities.json"
    if not state_path.exists():
        raise ValueError(
            "capability gate: no state/capabilities.json (run "
            "tools/capability/capability_check.py --write-state first) - "
            f"refusing to record {row['task_id']} (I1/N-C9)")
    state = json.loads(state_path.read_text())
    for p in req:
        if p not in state.get("state", {}):
            raise ValueError(f"capability gate: unknown path {p!r}")
        if not state["state"][p]["ok"]:
            raise ValueError(
                f"capability gate: REQUIRED path {p!r} is inactive "
                f"({state['state'][p].get('note', '')}) - refusing to record "
                f"{row['task_id']} (a silently-degraded run must not produce a "
                f"measurement, N-C9/I1)")


def append_row(row: dict) -> Path:
    """Append a validated row to results/{task_id}.jsonl (append-only, I3).

    Refuses to write a row whose run_id already exists in that file, and
    (N-C9) refuses when a required capability is inactive.
    """
    required_paths = row.pop("required_paths", None)
    validate_row(row)
    _capability_gate(row, required_paths)
    task_id = row["task_id"]
    if row["run_id"] in _existing_run_ids(task_id):
        raise ValueError(f"run_id {row['run_id']} already present in results/{task_id}.jsonl")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{task_id}.jsonl"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()
        os_fsync(fh)
    return path


def os_fsync(fh) -> None:
    import os
    try:
        os.fsync(fh.fileno())
    except OSError:
        pass


def write_artifact(relpath: str, records: list, force: bool = False) -> Path:
    """Write harness-produced task artifacts under results/ (e.g. profile_s4.jsonl).

    results/ is append-only per the work order: refuse to overwrite an existing
    file unless force=True (which still requires a DECISIONS.md trail if it is a
    re-measurement of a golden/benchmark artifact — see I4/H6).
    """
    path = RESULTS_DIR / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing artifact: {path} (append-only)")
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
        fh.flush()
        os_fsync(fh)
    return path


def read_rows(task_id: str) -> list:
    """Read all rows for a task (executor may read results freely, I3)."""
    path = RESULTS_DIR / f"{task_id}.jsonl"
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
