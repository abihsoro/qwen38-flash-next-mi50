#!/usr/bin/env bash
# bootstrap_negative.sh — negative controls for harness/verify_bootstrap.py.
#
# Principle (Work Order Rev 4, S3): an oracle that never fails is not an
# oracle. This script proves the bootstrap checker can fail:
#   1. positive control: a pristine copy of the repo must PASS the checker
#   2. negative control: the same copy with config/tolerances.yaml removed
#      must FAIL the checker
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[<bus>]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "[1/2] positive control: pristine copy must pass"
cp -r "$ROOT" "$TMP/proj"
(cd "$TMP/proj" && python3 harness/verify_bootstrap.py >/dev/null 2>&1)
if [ $? -eq 0 ]; then
  echo "  PASS (pristine copy accepted)"
else
  echo "  FAIL (pristine copy rejected)"
  exit 1
fi

echo "[2/2] negative control: missing config/tolerances.yaml must fail"
rm -f "$TMP/proj/config/tolerances.yaml"
(cd "$TMP/proj" && python3 harness/verify_bootstrap.py >/dev/null 2>&1)
if [ $? -ne 0 ]; then
  echo "  PASS (broken tree rejected)"
else
  echo "  FAIL (broken tree accepted)"
  exit 1
fi

echo "bootstrap_negative: all controls passed"
exit 0
