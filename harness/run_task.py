"""run_task.py — execute a work-order task invocation under an explicit timeout
and record exactly one results row (I2: exit codes are the source of truth).

Usage:
    python3 harness/run_task.py --task S1 --timeout 3600 -- <command...>

Behaviour:
  - validates --task against config/tasks.json
  - runs <command> in its own process group
  - on timeout: SIGKILLs the whole group, records exit_code 124 (I2: a hung
    process is a failure, never a pass)
  - writes one 34-field row to results/{task_id}.jsonl via results.append_row
  - exits 0 iff the command exited 0 within the timeout

The runner does not change config/tasks.json status; the executor updates the
registry separately when a task's acceptance predicate holds.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import results  # noqa: E402

TIMEOUT_EXIT_CODE = 124  # I2: killed-by-timeout is a failure


def load_tasks() -> dict:
    path = results.ROOT / "config" / "tasks.json"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        sys.exit(f"run_task: cannot read config/tasks.json: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a task invocation with an explicit timeout (I2).")
    ap.add_argument("--task", required=True, help="task id from config/tasks.json")
    ap.add_argument("--timeout", type=float, required=True, help="timeout in seconds; hard kill after this")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="command to run")
    args = ap.parse_args()

    tasks = load_tasks()
    ids = [t["id"] for t in tasks.get("tasks", [])]
    if args.task not in ids:
        sys.exit(f"run_task: unknown task id {args.task!r} (known: {', '.join(ids)})")
    cmd = args.cmd
    if cmd and cmd[<bus>] == "--":
        cmd = cmd[1:]  # argparse REMAINDER keeps the '--' separator; drop it
    if not cmd:
        sys.exit("run_task: empty command")

    started = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.Popen(cmd, start_new_session=True)
    except FileNotFoundError as exc:
        sys.exit(f"run_task: cannot start {cmd}: {exc}")

    try:
        exit_code = proc.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        exit_code = TIMEOUT_EXIT_CODE

    wall = time.monotonic() - started
    notes = (f"cmd={' '.join(str(c) for c in cmd)[:200]}; "
             f"timed_out={timed_out}")
    row = results.make_row(
        task_id=args.task,
        exit_code=int(exit_code),
        wall_seconds=round(wall, 3),
        timeout_s=args.timeout,
        notes=notes,
    )
    path = results.append_row(row)

    print(f"run_task: task={args.task} exit_code={exit_code} "
          f"wall={wall:.3f}s timeout={args.timeout}s -> {path.name}")
    print(f"run_task: run_id={row['run_id']}")
    return 0 if (exit_code == 0 and not timed_out) else 1


if __name__ == "__main__":
    sys.exit(main())
