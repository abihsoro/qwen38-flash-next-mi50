#!/usr/bin/env python3
"""context_pack.py — collapse the "read the repo one sed/grep at a time" pattern.

One invocation returns a compact, bounded map of a codebase so an agent can
orient itself without dozens of serial `sed`/`grep`/`ls` calls over (possibly
slow) storage. It is read-only, standard-library-only, and exits non-zero on
a missing target.

Usage:
  python3 context_pack.py [DIR] [options]

Options:
  --grep PATTERN      One or more literal-or-regex patterns to search source
                      files for; prints matching lines with file:line.
                      Repeat for multiple patterns.
  --max-files N       Cap the file tree listing (default 60, 0 = unlimited).
  --max-symbols N     Cap the def/class symbol list (default 200, 0 = unlimited).
  --max-grep N        Cap grep hits per pattern (default 60, 0 = unlimited).
  --json              Emit a single JSON object instead of human text.

Output sections (text mode):
  [tree]    path  size  (largest files first)
  [symbols] def/class/test definitions
  [tests]   test functions / TestCase classes / methods
  [grep]    per-pattern matching lines

The point of the caps is bounded output: an agent should get the shape of the
tree, the key symbols, the test names, and a few targeted grep hits — not a
full file dump — in a single tool call.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

# Directories and file suffixes skipped while walking. These are noise for a
# coding agent and rarely worth listing or grepping.
IGNORE_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".tox", ".nox", "node_modules", ".venv", "venv", "env",
    "dist", "build", ".eggs", "*.egg-info", ".next", ".cache", "wandb",
}
IGNORE_SUFFIXES = {
    ".pyc", ".pyo", ".so", ".o", ".a", ".class", ".jar", ".war", ".zip",
    ".tar", ".gz", ".bz2", ".xz", ".7z", ".whl", ".egg", ".png", ".jpg",
    ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".pptx", ".docx", ".xlsx",
    ".bin", ".exe", ".dll", ".dylib", ".lock", ".min.js", ".min.css",
    ".map",
}
# Files we will extract symbols from and grep.
TEXT_SUFFIXES = {
    ".py", ".pyi", ".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs", ".sh",
    ".bash", ".go", ".rs", ".c", ".h", ".cpp", ".hpp", ".cc", ".java",
    ".rb", ".php", ".md", ".rst", ".txt", ".yaml", ".yml", ".toml", ".json",
    ".sql", ".cs", ".swift", ".kt",
}
PY_SUFFIXES = {".py", ".pyi"}

# Matches "def name(", "async def name(", "class Name". Leading indentation is
# captured so nested definitions can be shown with a stable depth hint.
SYMBOL_RE = re.compile(r"^(?P<indent>\s*)(?:(?:async\s+)?def|class)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b")
TEST_FUNC_RE = re.compile(r"^\s*(?:async\s+)?def\s+(test_[A-Za-z0-9_]+)\b")
TEST_CLASS_RE = re.compile(r"^\s*class\s+(Test[A-Za-z0-9_]*|[A-Za-z0-9_]*Tests?)\b")
# Lines inside a TestCase class that look like test methods.
TEST_METHOD_RE = re.compile(r"^\s{4,}(?:async\s+)?def\s+(test_[A-Za-z0-9_]+)\b")


def _should_skip_dir(name: str) -> bool:
    if name in IGNORE_DIRS:
        return True
    for pat in IGNORE_DIRS:
        if pat.endswith(".egg-info") and name.endswith(".egg-info"):
            return True
    return False


def _should_skip_file(name: str) -> bool:
    if name.startswith("."):
        return False  # dotfiles like .gitignore are fine to list
    for suf in IGNORE_SUFFIXES:
        if name.endswith(suf):
            return True
    return False


def walk(root: str):
    """Yield (relpath, size) for files worth listing, largest first is handled later."""
    entries = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not _should_skip_dir(d))
        for fn in sorted(filenames):
            if _should_skip_file(fn):
                continue
            full = os.path.join(dirpath, fn)
            try:
                size = os.path.getsize(full)
            except OSError:
                size = 0
            rel = os.path.relpath(full, root)
            entries.append((rel, size, full))
    return entries


def extract_symbols(full: str, rel: str) -> list[str]:
    """Return 'rel:line  name' for def/class lines (only text source files)."""
    if not any(rel.endswith(s) for s in TEXT_SUFFIXES):
        return []
    out = []
    try:
        with open(full, "r", encoding="utf-8", errors="replace") as fh:
            for lineno, line in enumerate(fh, 1):
                m = SYMBOL_RE.match(line)
                if m:
                    depth = len(m.group("indent")) // 4
                    out.append(f"{rel}:{lineno}  {'  ' * depth}{m.group('name')}")
    except OSError:
        pass
    return out


def extract_tests(full: str, rel: str) -> list[str]:
    """Return test-related defs: functions, TestCase classes, and test methods."""
    if not any(rel.endswith(s) for s in PY_SUFFIXES):
        return []
    out = []
    in_test_class = False
    try:
        with open(full, "r", encoding="utf-8", errors="replace") as fh:
            for lineno, line in enumerate(fh, 1):
                mc = TEST_CLASS_RE.match(line)
                if mc:
                    out.append(f"{rel}:{lineno}  class {mc.group(1)}")
                    in_test_class = True
                    continue
                if in_test_class and TEST_METHOD_RE.match(line):
                    m = TEST_METHOD_RE.match(line)
                    out.append(f"{rel}:{lineno}    {m.group(1)}")
                    continue
                mf = TEST_FUNC_RE.match(line)
                if mf:
                    out.append(f"{rel}:{lineno}  {mf.group(1)}")
                if line.strip() and not line.startswith((" ", "\t", "#")):
                    in_test_class = False
    except OSError:
        pass
    return out


def grep_hits(full: str, rel: str, pattern: re.Pattern[str]) -> list[str]:
    if not any(rel.endswith(s) for s in TEXT_SUFFIXES):
        return []
    out = []
    try:
        with open(full, "r", encoding="utf-8", errors="replace") as fh:
            for lineno, line in enumerate(fh, 1):
                if pattern.search(line):
                    out.append(f"{rel}:{lineno}: {line.rstrip()[:160]}")
    except OSError:
        pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", nargs="?", default=".", help="directory to pack (default: cwd)")
    ap.add_argument("--grep", action="append", default=[], help="regex to search; repeatable")
    ap.add_argument("--max-files", type=int, default=60, help="file tree cap (0=unlimited)")
    ap.add_argument("--max-symbols", type=int, default=200, help="symbol cap (0=unlimited)")
    ap.add_argument("--max-grep", type=int, default=60, help="grep hits per pattern (0=unlimited)")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args()

    root = os.path.abspath(args.dir)
    if not os.path.isdir(root):
        print(f"error: not a directory: {args.dir}", file=sys.stderr)
        return 1

    entries = walk(root)
    entries.sort(key=lambda e: (-e[<bus>], e[<bus>]))  # largest files first

    tree = []
    for rel, size, _full in entries:
        if args.max_files and len(tree) >= args.max_files:
            tree.append(f"... ({len(entries) - len(tree)} more files)")
            break
        tree.append(f"{size:>10d}  {rel}")

    symbols: list[str] = []
    tests: list[str] = []
    for rel, _size, full in entries:
        if args.max_symbols and len(symbols) >= args.max_symbols:
            break
        symbols.extend(extract_symbols(full, rel))
        tests.extend(extract_tests(full, rel))
    symbols = symbols[: args.max_symbols] if args.max_symbols else symbols

    grep_map = {}
    for pat in args.grep:
        try:
            rx = re.compile(pat)
        except re.error as e:
            print(f"error: bad regex {pat!r}: {e}", file=sys.stderr)
            return 2
        hits: list[str] = []
        for rel, _size, full in entries:
            if args.max_grep and len(hits) >= args.max_grep:
                hits.append(f"... (truncated at {args.max_grep})")
                break
            hits.extend(grep_hits(full, rel, rx))
        grep_map[pat] = hits[: args.max_grep] if args.max_grep else hits

    if args.json:
        payload = {
            "dir": root,
            "file_count": len(entries),
            "tree": tree,
            "symbols": symbols,
            "tests": tests,
            "grep": grep_map,
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(f"# context pack: {root}  ({len(entries)} files)")
    print("\n## tree (largest first)")
    print("\n".join(tree) if tree else "(none)")
    print("\n## symbols")
    print("\n".join(symbols) if symbols else "(none)")
    print("\n## tests")
    print("\n".join(tests) if tests else "(none)")
    for pat, hits in grep_map.items():
        print(f"\n## grep /{pat}/")
        print("\n".join(hits) if hits else "(no hits)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
