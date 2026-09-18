#!/usr/bin/env bash
# run_health.sh — host-side (<source-host>): push the health probe into LXC <ct> and run it.
set -uo pipefail
pct push 300 <workdir>/m0v/health.cpp <workdir>/m0v/health.cpp
pct push 300 <workdir>/m0v/health_run.sh <workdir>/m0v/health_run.sh
pct exec 300 -- bash <workdir>/m0v/health_run.sh
