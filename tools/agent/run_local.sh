#!/usr/bin/env bash
#
# run_local.sh — run a job on local ext4, then sync final artifacts back.
#
# The DSH agent's workspace (<home>/<share>) is a CIFS/SMB mount to
# //<lan-ip>/<share>. A coding agent doing hundreds of small
# reads/writes there hits multi-minute stalls (a 119 KB `sed` was logged at
# 152s). <home>/work is local ext4: do the hot loop there, sync the result
# back once at the end.
#
# Usage:
#   run_local.sh <job-name> <origin-dir> <command...>
#
#   Example:
#     run_local.sh rfid <home>/<share>/70_Code \
#       "python3 rfid_ingester.py --test"
#
#   Steps:
#     1. rsync  <origin-dir>/  ->  $WORK_ROOT/<job-name>/   (caches/.git skipped)
#     2. run    <command...>   with cwd = $WORK_ROOT/<job-name>
#     3. rsync  $WORK_ROOT/<job-name>/  ->  <origin-dir>/   (final artifacts back)
#
# Environment:
#   WORK_ROOT  local scratch root (default <home>/work)
#   SYNC_BACK  set to 0 to skip the final rsync back (e.g. when iterating)
#
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-<home>/work}"

usage() {
  sed -n '2,30p' "$0"
  exit 2
}

if [ "$#" -lt 3 ]; then
  usage
fi

JOB="$1"
ORIGIN="$2"
shift 2

if [ ! -d "$ORIGIN" ]; then
  echo "error: origin is not a directory: $ORIGIN" >&2
  exit 2
fi

ORIGIN="$(cd "$ORIGIN" && pwd)"
LOCAL="$WORK_ROOT/$JOB"

# Caches and VCS noise are neither needed for the hot loop nor wanted in the
# sync-back pass. Everything else (source, tests, small data files) travels.
EXCLUDES=(
  --exclude=.git --exclude=.hg --exclude=.svn
  --exclude=__pycache__ --exclude='*.pyc'
  --exclude=.venv --exclude=venv --exclude=env
  --exclude=node_modules
  --exclude=.pytest_cache --exclude=.mypy_cache --exclude=.ruff_cache --exclude=.tox
  --exclude=dist --exclude=build
)

echo "== copy: $ORIGIN -> $LOCAL =="
mkdir -p "$LOCAL"
rsync -a "${EXCLUDES[@]}" "$ORIGIN"/ "$LOCAL"/

echo "== run: $*  (cwd=$LOCAL) =="
set +e
# bash -c "$*" so the command may be passed either as separate args
# (`... python3 x.py --test`) or as one quoted string with pipes/redirections
# (`... "python3 x.py --test 2>&1 | tail"`).
( cd "$LOCAL" && bash -c "$*" )
RC=$?
set -e

if [ "${SYNC_BACK:-1}" = "1" ]; then
  echo "== sync back: $LOCAL -> $ORIGIN =="
  rsync -a "${EXCLUDES[@]}" "$LOCAL"/ "$ORIGIN"/
else
  echo "== SYNC_BACK=0: skipping sync back =="
fi

echo "== done rc=$RC (local copy kept at $LOCAL) =="
exit "$RC"
