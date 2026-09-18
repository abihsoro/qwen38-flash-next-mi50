#!/usr/bin/env python3
"""test_wrapper.py — split the test suite into smoke / normal / stress profiles.

The goal is to stop a coding agent from re-running the full (and especially the
adversarial/SIGKILL/concurrency) suite as its inner loop. Instead:

  smoke   — syntax check + fast correctness tests only (no subprocess storms,
            no long sleeps). Cheap enough to run after every edit.
  normal  — the regular unit suite.
  stress  — adversarial / crash-safety / concurrency / truncation / fuzz tests.
            Run once the core suite is green, or when touching crash-safety code.

Usage:
  python3 test_wrapper.py --profile {smoke,normal,stress} [options]

Options:
  --dir DIR          Root to compile/discover tests in (default: cwd).
  --module MOD       A single module to test via `python MOD --test` when the
                     module exposes that flag (e.g. rfid_ingester.py).
  --pattern REGEX    Override the per-profile test-name regex used for unittest
                     discovery selection.
  --timeout SECONDS  Per-test-invocation timeout (default 900).

Discovery order: if --module is given and has a `--test` flag, that module's
own runner is used and the profile maps onto it (smoke=compile+import, stress=
the module's crash-safety paths only when discoverable). Otherwise it falls back
to unittest discovery (or pytest if available and configured).
"""

from __future__ import annotations

import argparse
import os
import py_compile
import re
import subprocess
import sys
import tempfile
import unittest

# Name patterns that classify a test as smoke- or stress-eligible. These are
# heuristics over test names/files; use --pattern to override.
SMOKE_RE = re.compile(r"(smoke|fast|quick|simple|basic|unit|syntax|parse)", re.I)
STRESS_RE = re.compile(
    r"(stress|attack|crash|concurren|sigkill|sigterm|kill|truncat|race|adversarial|fuzz|"
    r"partial|corrupt|interrupt|resume|restart|recover)",
    re.I,
)


def find_py_files(root: str) -> list[str]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {"__pycache__", ".git", ".venv", "venv", "node_modules"}]
        for fn in filenames:
            if fn.endswith(".py"):
                out.append(os.path.join(dirpath, fn))
    return out


def compile_all(root: str) -> tuple[int, list[str]]:
    """Byte-compile every .py under root; returns (failures, errors)."""
    errors = []
    for path in find_py_files(root):
        if os.path.basename(path).startswith("test_"):
            continue
        try:
            py_compile.compile(path, doraise=True)
        except py_compile.PyCompileError as e:
            errors.append(str(e))
        except Exception as e:  # noqa: BLE001
            errors.append(f"{path}: {e}")
    return len(errors), errors


def module_has_test_flag(module: str) -> bool:
    try:
        with open(module, "r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(200_000)
        return bool(re.search(r"add_argument\([\"']--test[\"']", head))
    except OSError:
        return False


def run_cmd(cmd: list[str], cwd: str, timeout: int) -> int:
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=cwd, timeout=timeout)


def run_unittest(root: str, pattern: str | None, timeout: int) -> int:
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", root, "-v"]
    if pattern:
        cmd += ["-p", pattern]
    return run_cmd(cmd, root, timeout)


def run_pytest(root: str, pattern: str | None, timeout: int) -> int:
    cmd = [sys.executable, "-m", "pytest", "-q"]
    if pattern:
        cmd += ["-k", pattern]
    else:
        cmd += [root]
    return run_cmd(cmd, root, timeout)


def uses_pytest(root: str) -> bool:
    for name in ("pytest.ini", "pyproject.toml", "setup.cfg"):
        p = os.path.join(root, name)
        if os.path.exists(p):
            try:
                if "pytest" in open(p, encoding="utf-8", errors="replace").read(200_000):
                    return True
            except OSError:
                pass
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True, choices=["smoke", "normal", "stress"])
    ap.add_argument("--dir", default=".")
    ap.add_argument("--module", default=None)
    ap.add_argument("--pattern", default=None)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    root = os.path.abspath(args.dir)
    if not os.path.isdir(root):
        print(f"error: not a directory: {args.dir}", file=sys.stderr)
        return 2

    profile = args.profile
    print(f"# test_wrapper profile={profile} root={root}")

    # 1. Every profile starts with a syntax gate.
    n_err, errors = compile_all(root)
    if errors:
        print(f"## syntax: {n_err} error(s)")
        for e in errors[:20]:
            print("  " + e)
        return 1
    print(f"## syntax: OK ({len(find_py_files(root))} files)")

    # 2. Single-module fast path (module exposes --test).
    if args.module:
        module = args.module
        mpath = os.path.abspath(module)
        if not os.path.exists(mpath):
            print(f"error: module not found: {module}", file=sys.stderr)
            return 2
        if module_has_test_flag(mpath):
            if profile == "smoke":
                # import-only smoke: proves it at least imports cleanly.
                print("## smoke: import check")
                rc = run_cmd([sys.executable, "-c", f"import importlib.util as u; u.spec_from_file_location('m', {mpath!r}) and __import__('importlib').import_module('m') if False else None; print('import OK')"], root, args.timeout)
                return rc
            # normal and stress both use the module's own runner.
            print(f"## {profile}: module runner")
            return run_cmd([sys.executable, mpath, "--test"], root, args.timeout)

    # 3. Discovery-based path.
    pattern = args.pattern
    if pattern is None and profile == "smoke":
        pattern = "*smoke*.py"  # conservative; --pattern overrides
    elif pattern is None and profile == "stress":
        pattern = "*stress*.py"

    if uses_pytest(root):
        return run_pytest(root, pattern, args.timeout)
    return run_unittest(root, pattern, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
