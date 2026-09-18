#!/usr/bin/env bash
set -euo pipefail
preset=${1:-balanced}
case "$preset" in
  stop)
    exec ssh root@<container-ip> '<home>/gfx906-venv/bin/python3 <tuning>/serve_preset.py --stop'
    ;;
  balanced|mtp0|reference)
    ssh root@<host-ip> 'python3 <tuning>/set_power.py'
    exec ssh root@<container-ip> "<home>/gfx906-venv/bin/python3 <tuning>/serve_preset.py $preset"
    ;;
  *) printf '%s\n' 'Usage: launch.sh [balanced|mtp0|reference|stop]' >&2; exit 2 ;;
esac
